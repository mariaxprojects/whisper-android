package com.bpdq.whisper.engine

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File

/**
 * The native library is compiled for ARMv8.2-A with FP16 and dot-product
 * instructions. Running it on an older core would crash with SIGILL, so we
 * check /proc/cpuinfo first and show a message instead.
 */
object CpuSupport {
    private const val TAG = "CpuSupport"

    sealed interface Result {
        data object Supported : Result
        data class Unsupported(val reason: String) : Result
    }

    fun check(): Result {
        if (!Build.SUPPORTED_ABIS.contains("arm64-v8a")) {
            return Result.Unsupported(
                "This app needs a 64-bit ARM phone. This device reports: " +
                    Build.SUPPORTED_ABIS.joinToString()
            )
        }
        val features = readFeatures()
        if (features == null) {
            // Some kernels hide /proc/cpuinfo; Android 10+ phones are ARMv8.2 in practice.
            Log.w(TAG, "Could not read CPU features; assuming support")
            return Result.Supported
        }
        val hasFp16 = "fphp" in features && "asimdhp" in features
        val hasDot = "asimddp" in features
        return if (hasFp16 && hasDot) Result.Supported else Result.Unsupported(
            "This phone's processor is too old for the model runtime " +
                "(it needs ARMv8.2 half-precision and dot-product instructions, " +
                "found on Cortex-A55/A75 and newer)."
        )
    }

    private fun readFeatures(): Set<String>? = try {
        File("/proc/cpuinfo").readLines()
            .filter { it.startsWith("Features") }
            .flatMap { it.substringAfter(':').trim().split(Regex("\\s+")) }
            .toSet()
            .takeIf { it.isNotEmpty() }
    } catch (e: Exception) {
        Log.w(TAG, "Couldn't read /proc/cpuinfo", e)
        null
    }

    /** Total RAM in bytes. */
    fun totalRam(context: Context): Long {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val info = ActivityManager.MemoryInfo()
        am.getMemoryInfo(info)
        return info.totalMem
    }

    /**
     * A sensible default thread count: the number of "big" cores (those faster
     * than the slowest cluster), at least 2 and at most 8.
     */
    fun defaultThreadCount(): Int {
        val n = Runtime.getRuntime().availableProcessors()
        val freqs = (0 until n).mapNotNull { i ->
            try {
                File("/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq").readText().trim().toLong()
            } catch (e: Exception) {
                null
            }
        }
        val big = if (freqs.size == n && freqs.distinct().size > 1) {
            val min = freqs.min()
            freqs.count { it > min }
        } else {
            n / 2
        }
        return big.coerceIn(2, 8)
    }
}
