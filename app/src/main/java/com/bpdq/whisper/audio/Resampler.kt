package com.bpdq.whisper.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Streaming band-limited resampler (windowed sinc, table driven).
 * Feed mono input with [push]; output samples are appended to [out].
 * Call [finish] once at the end to flush the tail.
 */
class Resampler(private val inRate: Int, private val outRate: Int, private val out: FloatList) {
    private val step = inRate.toDouble() / outRate             // input samples per output sample
    private val cutoff = 0.5 * min(1.0, 1.0 / step) * 0.92     // cycles per input sample, below output Nyquist
    private val halfWidth = (8 * max(1.0, step)).toInt() + 1   // taps on each side, in input samples
    private val phases = 128                                   // table resolution per input sample
    private val table: FloatArray = FloatArray(halfWidth * phases + 2).also { t ->
        for (i in t.indices) {
            val d = i.toDouble() / phases
            val x = 2.0 * cutoff * d
            val sinc = if (x == 0.0) 1.0 else sin(PI * x) / (PI * x)
            val w = if (d >= halfWidth) 0.0 else 0.42 + 0.5 * cos(PI * d / halfWidth) + 0.08 * cos(2 * PI * d / halfWidth)
            t[i] = (2.0 * cutoff * sinc * w).toFloat()
        }
    }

    // Sliding window of input; buf[0] corresponds to absolute input index bufStart.
    private var buf = FloatArray(1 shl 15)
    private var bufLen = 0
    private var bufStart = 0L
    private var nextOut = 0L                      // index of the next output sample

    private val passthrough = inRate == outRate

    fun push(src: FloatArray, n: Int) {
        if (passthrough) { out.addAll(src, 0, n); return }
        if (bufLen + n > buf.size) {
            // Drop input that no future output sample can reach.
            val keepFrom = (floor(nextOut * step).toLong() - halfWidth - 1).coerceAtLeast(bufStart)
            val drop = (keepFrom - bufStart).toInt()
            if (drop > 0) {
                System.arraycopy(buf, drop, buf, 0, bufLen - drop)
                bufLen -= drop
                bufStart += drop
            }
            if (bufLen + n > buf.size) buf = buf.copyOf(max(buf.size * 2, bufLen + n))
        }
        System.arraycopy(src, 0, buf, bufLen, n)
        bufLen += n
        produce(final = false)
    }

    fun finish() {
        if (passthrough) return
        produce(final = true)
    }

    private fun produce(final: Boolean) {
        val available = bufStart + bufLen
        val totalOut = if (final) (available * outRate + inRate - 1) / inRate else Long.MAX_VALUE
        while (nextOut < totalOut) {
            val t = nextOut * step
            val center = floor(t).toLong()
            if (!final && center + halfWidth >= available) break
            val frac = t - center
            var acc = 0.0f
            val lo = center - halfWidth + 1
            val hi = center + halfWidth
            var j = lo
            while (j <= hi) {
                if (j >= bufStart && j < available) {
                    val d = kotlin.math.abs(j - t) * phases
                    val i = d.toInt()
                    if (i < table.size - 1) {
                        val f = (d - i).toFloat()
                        val h = table[i] + (table[i + 1] - table[i]) * f
                        acc += buf[(j - bufStart).toInt()] * h
                    }
                }
                j++
            }
            out.add(acc)
            nextOut++
        }
    }
}
