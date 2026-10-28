package com.bpdq.whisper.engine

import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors

data class Segment(val startMs: Long, val endMs: Long, val text: String)

data class TranscribeOptions(
    val threads: Int,
    val language: String = "en",
    val translate: Boolean = false,
    /** 0 = full 30 s encoder window (the model author's setting). */
    val audioCtx: Int = 0,
)

sealed interface TranscribeOutcome {
    data class Done(val segments: List<Segment>) : TranscribeOutcome
    data class Cancelled(val segments: List<Segment>) : TranscribeOutcome
    data class Failed(val message: String) : TranscribeOutcome
}

/**
 * Owns one whisper.cpp context. whisper.cpp contexts must not be used from
 * more than one thread at a time, so every call runs on a private thread.
 */
class WhisperEngine private constructor(
    private var ptr: Long,
    private val dispatcher: ExecutorCoroutineDispatcher,
) {
    suspend fun transcribe(
        audio: FloatArray,
        options: TranscribeOptions,
        progressCallback: (Int) -> Unit,
        segmentCallback: (Segment) -> Unit,
    ): TranscribeOutcome = withContext(dispatcher) {
        check(ptr != 0L) { "Engine was released" }
        val segments = ArrayList<Segment>()
        val listener = object : TranscribeListener {
            override fun onProgress(percent: Int) {
                progressCallback(percent)
            }
            override fun onSegment(t0: Long, t1: Long, text: String) {
                val s = Segment(t0 * 10, t1 * 10, text)
                segments += s
                segmentCallback(s)
            }
        }
        when (val rc = WhisperNative.transcribe(
            ptr, audio, options.threads, options.language, options.translate, options.audioCtx, listener
        )) {
            0 -> TranscribeOutcome.Done(segments)
            1 -> TranscribeOutcome.Cancelled(segments)
            else -> TranscribeOutcome.Failed("Transcription failed (code $rc).\n" + WhisperNative.recentLog().takeLast(1500))
        }
    }

    /** Safe to call from any thread; the running transcription stops at the next step. */
    fun cancel() = WhisperNative.requestAbort()

    suspend fun release() {
        withContext(dispatcher) {
            if (ptr != 0L) {
                WhisperNative.freeContext(ptr)
                ptr = 0L
            }
        }
        dispatcher.close()
    }

    companion object {
        suspend fun load(model: File): WhisperEngine {
            val dispatcher = Executors.newSingleThreadExecutor { r ->
                Thread(r, "whisper-engine").apply { priority = Thread.MAX_PRIORITY }
            }.asCoroutineDispatcher()
            val ptr = withContext(dispatcher) {
                WhisperNative.ensureLoaded()
                WhisperNative.initContext(model.absolutePath)
            }
            if (ptr == 0L) {
                val log = withContext(dispatcher) { WhisperNative.recentLog() }
                dispatcher.close()
                throw IllegalStateException(
                    "The runtime could not load the model file.\n\n" + log.takeLast(1500)
                )
            }
            return WhisperEngine(ptr, dispatcher)
        }

        fun systemInfo(): String {
            WhisperNative.ensureLoaded()
            return WhisperNative.systemInfo()
        }
    }
}
