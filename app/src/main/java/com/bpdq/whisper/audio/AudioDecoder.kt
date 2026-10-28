package com.bpdq.whisper.audio

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.IOException
import java.nio.ByteOrder

const val WHISPER_SAMPLE_RATE = 16_000

/**
 * Decodes any audio (or the audio track of a video) that Android can play
 * into 16 kHz mono float samples, which is what Whisper expects.
 */
object AudioDecoder {
    /** Longer inputs would need several hundred MB of RAM; cap them. */
    const val MAX_SECONDS = 60 * 60

    suspend fun decode(context: Context, uri: Uri, onProgress: (Float) -> Unit = {}): FloatArray =
        withContext(Dispatchers.Default) {
            val extractor = MediaExtractor()
            try {
                extractor.setDataSource(context, uri, null)
                val track = (0 until extractor.trackCount).firstOrNull {
                    extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
                } ?: throw IOException("No audio track found in that file")
                extractor.selectTrack(track)
                val format = extractor.getTrackFormat(track)
                val mime = format.getString(MediaFormat.KEY_MIME)!!
                val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) else -1L

                var sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                var encoding = AudioFormat.ENCODING_PCM_16BIT

                val codec = try {
                    MediaCodec.createDecoderByType(mime)
                } catch (e: Exception) {
                    throw IOException("This phone can't decode $mime audio", e)
                }
                val out = FloatList()
                var resampler: Resampler? = null
                var resamplerRate = -1
                var mono = FloatArray(8192)
                val maxOutSamples = MAX_SECONDS.toLong() * WHISPER_SAMPLE_RATE

                try {
                    codec.configure(format, null, null, 0)
                    codec.start()
                    val info = MediaCodec.BufferInfo()
                    var inputDone = false
                    var outputDone = false
                    while (!outputDone) {
                        ensureActive()
                        if (!inputDone) {
                            val inIdx = codec.dequeueInputBuffer(10_000)
                            if (inIdx >= 0) {
                                val inBuf = codec.getInputBuffer(inIdx)!!
                                val n = extractor.readSampleData(inBuf, 0)
                                if (n < 0) {
                                    codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                    inputDone = true
                                } else {
                                    codec.queueInputBuffer(inIdx, 0, n, extractor.sampleTime, 0)
                                    if (durationUs > 0) onProgress((extractor.sampleTime.toFloat() / durationUs).coerceIn(0f, 1f))
                                    extractor.advance()
                                }
                            }
                        }
                        val outIdx = codec.dequeueOutputBuffer(info, 10_000)
                        when {
                            outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                                val f = codec.outputFormat
                                sampleRate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                                channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                                if (f.containsKey(MediaFormat.KEY_PCM_ENCODING)) encoding = f.getInteger(MediaFormat.KEY_PCM_ENCODING)
                            }
                            outIdx >= 0 -> {
                                val ob = codec.getOutputBuffer(outIdx)!!
                                ob.position(info.offset)
                                ob.limit(info.offset + info.size)
                                ob.order(ByteOrder.nativeOrder())

                                val frames: Int
                                when (encoding) {
                                    AudioFormat.ENCODING_PCM_FLOAT -> {
                                        val fb = ob.asFloatBuffer()
                                        frames = fb.remaining() / channels
                                        if (mono.size < frames) mono = FloatArray(frames)
                                        for (i in 0 until frames) {
                                            var s = 0f
                                            for (c in 0 until channels) s += fb.get(i * channels + c)
                                            mono[i] = s / channels
                                        }
                                    }
                                    else -> { // 16-bit PCM (the decoder default)
                                        val sb = ob.asShortBuffer()
                                        frames = sb.remaining() / channels
                                        if (mono.size < frames) mono = FloatArray(frames)
                                        for (i in 0 until frames) {
                                            var s = 0
                                            for (c in 0 until channels) s += sb.get(i * channels + c)
                                            mono[i] = s / (32768f * channels)
                                        }
                                    }
                                }
                                if (resampler == null || resamplerRate != sampleRate) {
                                    resampler?.finish()
                                    resampler = Resampler(sampleRate, WHISPER_SAMPLE_RATE, out)
                                    resamplerRate = sampleRate
                                }
                                resampler.push(mono, frames)
                                codec.releaseOutputBuffer(outIdx, false)

                                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                                if (out.size >= maxOutSamples) outputDone = true
                            }
                        }
                    }
                    resampler?.finish()
                } finally {
                    try { codec.stop() } catch (_: Exception) {}
                    codec.release()
                }
                if (out.size == 0) throw IOException("The file contained no audio samples")
                onProgress(1f)
                val samples = out.toArray()
                if (samples.size > maxOutSamples) samples.copyOf(maxOutSamples.toInt()) else samples
            } finally {
                extractor.release()
            }
        }
}
