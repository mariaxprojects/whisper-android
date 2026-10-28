package com.bpdq.whisper.engine

/** Receives callbacks from native code while a transcription runs. */
interface TranscribeListener {
    fun onProgress(percent: Int)
    /** [t0] and [t1] are in units of 10 ms, as whisper.cpp reports them. */
    fun onSegment(t0: Long, t1: Long, text: String)
}

/** Raw JNI bindings. Use [WhisperEngine] instead of calling these directly. */
internal object WhisperNative {
    @Volatile private var loaded = false

    /** Loads libbpdqwhisper.so. Call only after [CpuSupport.check] has passed. */
    @Synchronized
    fun ensureLoaded() {
        if (!loaded) {
            System.loadLibrary("bpdqwhisper")
            loaded = true
        }
    }

    @JvmStatic external fun initContext(modelPath: String): Long
    @JvmStatic external fun freeContext(ptr: Long)
    @JvmStatic external fun systemInfo(): String
    @JvmStatic external fun recentLog(): String
    @JvmStatic external fun requestAbort()

    /** Returns 0 on success, 1 if cancelled, negative on failure. */
    @JvmStatic external fun transcribe(
        ptr: Long,
        audio: FloatArray,
        nThreads: Int,
        language: String,
        translate: Boolean,
        audioCtx: Int,
        listener: TranscribeListener,
    ): Int
}
