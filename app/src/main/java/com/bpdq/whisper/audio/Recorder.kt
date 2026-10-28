package com.bpdq.whisper.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.IOException
import kotlin.math.abs
import kotlin.math.max

/**
 * Records 16 kHz mono PCM from the microphone until the calling coroutine is
 * cancelled or [stop] is called, then returns the samples as floats.
 */
class Recorder {
    @Volatile private var stopRequested = false

    fun stop() { stopRequested = true }

    /** [onLevel] receives a 0..1 peak level about 20 times a second; [onElapsedMs] the recording length. */
    @SuppressLint("MissingPermission") // the UI asks for RECORD_AUDIO before calling this
    suspend fun record(onLevel: (Float) -> Unit, onElapsedMs: (Long) -> Unit): FloatArray =
        withContext(Dispatchers.IO) {
            stopRequested = false
            val minBuf = AudioRecord.getMinBufferSize(
                WHISPER_SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            if (minBuf <= 0) throw IOException("This phone can't record 16 kHz mono audio")
            val chunk = WHISPER_SAMPLE_RATE / 20 // 50 ms
            val rec = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                WHISPER_SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                max(minBuf, chunk * 2 * 4),
            )
            if (rec.state != AudioRecord.STATE_INITIALIZED) {
                rec.release()
                throw IOException("Couldn't open the microphone")
            }
            val out = FloatList()
            val shorts = ShortArray(chunk)
            val floats = FloatArray(chunk)
            val maxSamples = AudioDecoder.MAX_SECONDS * WHISPER_SAMPLE_RATE
            try {
                rec.startRecording()
                while (isActive && !stopRequested && out.size < maxSamples) {
                    val n = rec.read(shorts, 0, chunk)
                    if (n < 0) throw IOException("Microphone read failed ($n)")
                    var peak = 0
                    for (i in 0 until n) {
                        floats[i] = shorts[i] / 32768f
                        peak = max(peak, abs(shorts[i].toInt()))
                    }
                    out.addAll(floats, 0, n)
                    onLevel(peak / 32768f)
                    onElapsedMs(out.size * 1000L / WHISPER_SAMPLE_RATE)
                }
            } finally {
                try { rec.stop() } catch (_: Exception) {}
                rec.release()
            }
            out.toArray()
        }
}
