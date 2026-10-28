// JNI bridge between the Kotlin app and the BPDQ2-patched whisper.cpp.
#include <jni.h>
#include <android/log.h>

#include <atomic>
#include <cmath>
#include <deque>
#include <mutex>
#include <string>

#include "whisper.h"
#include "ggml.h"

#define TAG "BPDQWhisper"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

JavaVM *g_vm = nullptr;
std::atomic<bool> g_abort{false};

// Keep the last whisper/ggml log lines so the UI can explain a failed load.
std::mutex g_log_mutex;
std::deque<std::string> g_log_tail;
std::string g_log_partial;

void log_callback(ggml_log_level level, const char *text, void *) {
    if (!text) return;
    const int prio = level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR
                   : level == GGML_LOG_LEVEL_WARN  ? ANDROID_LOG_WARN
                   : ANDROID_LOG_INFO;
    __android_log_write(prio, "whisper", text);

    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_partial += text;
    size_t pos;
    while ((pos = g_log_partial.find('\n')) != std::string::npos) {
        g_log_tail.emplace_back(g_log_partial.substr(0, pos));
        g_log_partial.erase(0, pos + 1);
        while (g_log_tail.size() > 60) g_log_tail.pop_front();
    }
}

JNIEnv *env_for_this_thread() {
    JNIEnv *env = nullptr;
    if (g_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_OK) return env;
#if defined(__ANDROID__)
    if (g_vm->AttachCurrentThread(&env, nullptr) == JNI_OK) return env;   // NDK: JNIEnv**
#else
    if (g_vm->AttachCurrentThread(reinterpret_cast<void **>(&env), nullptr) == JNI_OK) return env; // desktop JDK: void**
#endif
    return nullptr;
}

struct CallbackCtx {
    jobject listener;
    jmethodID on_progress;
    jmethodID on_segment;
};

void progress_cb(whisper_context *, whisper_state *, int progress, void *user) {
    auto *cb = static_cast<CallbackCtx *>(user);
    JNIEnv *env = env_for_this_thread();
    if (!env) return;
    env->CallVoidMethod(cb->listener, cb->on_progress, (jint) progress);
    if (env->ExceptionCheck()) env->ExceptionClear();
}

void new_segment_cb(whisper_context *ctx, whisper_state *, int n_new, void *user) {
    auto *cb = static_cast<CallbackCtx *>(user);
    JNIEnv *env = env_for_this_thread();
    if (!env) return;
    const int n = whisper_full_n_segments(ctx);
    for (int i = n - n_new; i < n; ++i) {
        const char *text = whisper_full_get_segment_text(ctx, i);
        jstring jtext = env->NewStringUTF(text ? text : "");
        env->CallVoidMethod(cb->listener, cb->on_segment,
                            (jlong) whisper_full_get_segment_t0(ctx, i),
                            (jlong) whisper_full_get_segment_t1(ctx, i),
                            jtext);
        env->DeleteLocalRef(jtext);
        if (env->ExceptionCheck()) env->ExceptionClear();
    }
}

// Called by ggml from its compute thread 0; must not touch the JVM.
bool abort_cb(void *) { return g_abort.load(std::memory_order_relaxed); }
bool encoder_begin_cb(whisper_context *, whisper_state *, void *) { return !g_abort.load(); }

} // namespace

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
    g_vm = vm;
    whisper_log_set(log_callback, nullptr);
    return JNI_VERSION_1_6;
}

JNIEXPORT jlong JNICALL
Java_com_bpdq_whisper_engine_WhisperNative_initContext(JNIEnv *env, jclass, jstring jpath) {
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_log_tail.clear();
        g_log_partial.clear();
    }
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false; // CPU backend only on Android
    LOGI("loading model %s", path);
    whisper_context *ctx = whisper_init_from_file_with_params(path, cparams);
    env->ReleaseStringUTFChars(jpath, path);
    if (!ctx) LOGE("whisper_init_from_file_with_params failed");
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT void JNICALL
Java_com_bpdq_whisper_engine_WhisperNative_freeContext(JNIEnv *, jclass, jlong ptr) {
    if (ptr) whisper_free(reinterpret_cast<whisper_context *>(ptr));
}

JNIEXPORT jstring JNICALL
Java_com_bpdq_whisper_engine_WhisperNative_systemInfo(JNIEnv *env, jclass) {
    return env->NewStringUTF(whisper_print_system_info());
}

JNIEXPORT jstring JNICALL
Java_com_bpdq_whisper_engine_WhisperNative_recentLog(JNIEnv *env, jclass) {
    std::string out;
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        for (const auto &line : g_log_tail) { out += line; out += '\n'; }
        out += g_log_partial;
    }
    return env->NewStringUTF(out.c_str());
}

JNIEXPORT void JNICALL
Java_com_bpdq_whisper_engine_WhisperNative_requestAbort(JNIEnv *, jclass) {
    g_abort.store(true);
}

// Returns 0 on success, 1 if cancelled, negative on failure.
JNIEXPORT jint JNICALL
Java_com_bpdq_whisper_engine_WhisperNative_transcribe(
        JNIEnv *env, jclass, jlong ptr, jfloatArray jaudio, jint n_threads,
        jstring jlanguage, jboolean translate, jint audio_ctx, jobject listener) {
    auto *ctx = reinterpret_cast<whisper_context *>(ptr);
    if (!ctx) return -1;
    g_abort.store(false);

    jclass lcls = env->GetObjectClass(listener);
    CallbackCtx cb{
        listener,
        env->GetMethodID(lcls, "onProgress", "(I)V"),
        env->GetMethodID(lcls, "onSegment", "(JJLjava/lang/String;)V"),
    };
    if (!cb.on_progress || !cb.on_segment) return -2;

    const char *lang = env->GetStringUTFChars(jlanguage, nullptr);
    std::string language(lang ? lang : "en");
    env->ReleaseStringUTFChars(jlanguage, lang);

    // Same decoding setup as the model author's launcher: greedy, no temperature fallback.
    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.n_threads        = n_threads > 0 ? n_threads : 4;
    p.language         = language.c_str();
    p.detect_language  = false;
    p.translate        = translate;
    p.no_context       = true;
    p.single_segment   = false;
    p.print_progress   = false;
    p.print_realtime   = false;
    p.print_special    = false;
    p.print_timestamps = false;
    p.temperature_inc  = 0.0f;
    p.audio_ctx        = audio_ctx > 0 ? audio_ctx : 0;

    p.progress_callback              = progress_cb;
    p.progress_callback_user_data    = &cb;
    p.new_segment_callback           = new_segment_cb;
    p.new_segment_callback_user_data = &cb;
    p.abort_callback                 = abort_cb;
    p.abort_callback_user_data       = nullptr;
    p.encoder_begin_callback         = encoder_begin_cb;
    p.encoder_begin_callback_user_data = nullptr;

    jfloat *audio = env->GetFloatArrayElements(jaudio, nullptr);
    const jsize n_samples = env->GetArrayLength(jaudio);

    whisper_reset_timings(ctx);
    LOGI("whisper_full: %d samples, %d threads, audio_ctx=%d, lang=%s",
         (int) n_samples, p.n_threads, p.audio_ctx, language.c_str());
    const int rc = whisper_full(ctx, p, audio, n_samples);
    env->ReleaseFloatArrayElements(jaudio, audio, JNI_ABORT);
    whisper_print_timings(ctx);

    if (g_abort.load()) return 1;
    return rc == 0 ? 0 : -3;
}

} // extern "C"
