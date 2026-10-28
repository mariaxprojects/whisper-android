package com.bpdq.whisper.model

import android.content.Context
import android.net.Uri
import android.os.StatFs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/** The packed BPDQ2 model published by the model author (same file the Mac/iOS runtime uses). */
object ModelInfo {
    const val REPO = "diegogranziol/whisper-large-v3-turbo-bpdq-2bit-qat"
    const val URL = "https://huggingface.co/$REPO/resolve/main/metal/whisper-bpdq2.bin"
    const val SHA256 = "1fbdc4bd81d721bf67511824f4926f4f6c0adba1669cfa0ab0927476285432c7"
    const val SIZE_BYTES = 364_741_703L
    const val FILE_NAME = "whisper-bpdq2.bin"
}

sealed interface ModelProgress {
    data class Transferring(val bytes: Long, val total: Long, val bytesPerSecond: Long) : ModelProgress
    data class Verifying(val bytes: Long, val total: Long) : ModelProgress
}

class ModelManager(context: Context) {
    private val dir = File(context.filesDir, "models").apply { mkdirs() }
    val modelFile = File(dir, ModelInfo.FILE_NAME)
    private val partFile = File(dir, ModelInfo.FILE_NAME + ".part")
    private val verifiedMarker = File(dir, ModelInfo.FILE_NAME + ".verified")

    /** True once the file is present and its SHA-256 has been checked. */
    fun isReady(): Boolean =
        modelFile.isFile &&
            modelFile.length() == ModelInfo.SIZE_BYTES &&
            verifiedMarker.isFile &&
            verifiedMarker.readText().trim() == markerFor(modelFile)

    fun partialBytes(): Long = if (partFile.isFile) partFile.length() else 0L

    fun delete() {
        modelFile.delete(); partFile.delete(); verifiedMarker.delete()
    }

    /** Downloads (resuming a previous partial download if there is one) and verifies the model. */
    suspend fun download(onProgress: (ModelProgress) -> Unit) = withContext(Dispatchers.IO) {
        if (isReady()) return@withContext
        ensureFreeSpace(ModelInfo.SIZE_BYTES - partialBytes())

        var attempt = 0
        while (true) {
            ensureActive()
            try {
                downloadOnce(onProgress)
                break
            } catch (e: IOException) {
                // Flaky mobile connection: retry a few times, keeping what we have.
                if (++attempt >= 4) throw IOException("Download failed: ${e.message}", e)
                delay(1500L * attempt)
            }
        }
        verifyAndInstall(partFile, onProgress)
    }

    /** Copies a model file the user picked (e.g. downloaded on a computer) and verifies it. */
    suspend fun importFrom(context: Context, uri: Uri, onProgress: (ModelProgress) -> Unit) =
        withContext(Dispatchers.IO) {
            ensureFreeSpace(ModelInfo.SIZE_BYTES)
            partFile.delete()
            val input = context.contentResolver.openInputStream(uri)
                ?: throw IOException("Couldn't open the selected file")
            input.use { inp ->
                FileOutputStream(partFile).use { out -> copyWithProgress(inp, out, 0L, onProgress) }
            }
            if (partFile.length() != ModelInfo.SIZE_BYTES) {
                val got = partFile.length()
                partFile.delete()
                throw IOException(
                    "That file is ${got} bytes; the model should be exactly ${ModelInfo.SIZE_BYTES} bytes " +
                        "(metal/whisper-bpdq2.bin from ${ModelInfo.REPO})."
                )
            }
            verifyAndInstall(partFile, onProgress)
        }

    /** Opens [ModelInfo.URL], following redirects by hand so the Range header reaches the CDN too. */
    private fun openFollowingRedirects(rangeStart: Long): HttpURLConnection {
        var url = URL(ModelInfo.URL)
        repeat(10) {
            val conn = (url.openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 20_000
                readTimeout = 30_000
                setRequestProperty("User-Agent", "BPDQWhisper-Android/1.0")
                if (rangeStart > 0) setRequestProperty("Range", "bytes=$rangeStart-")
            }
            val code = conn.responseCode
            if (code !in 300..399) return conn
            val loc = conn.getHeaderField("Location")
            conn.disconnect()
            if (loc == null) throw IOException("Redirect without Location from ${url.host}")
            url = URL(url, loc)
        }
        throw IOException("Too many redirects")
    }

    private suspend fun downloadOnce(onProgress: (ModelProgress) -> Unit) = withContext(Dispatchers.IO) {
        var existing = partialBytes()
        if (existing >= ModelInfo.SIZE_BYTES) return@withContext

        val conn = openFollowingRedirects(existing)
        when (val code = conn.responseCode) {
            HttpURLConnection.HTTP_PARTIAL -> Unit
            HttpURLConnection.HTTP_OK -> if (existing > 0) {
                // Server ignored the range; start over.
                partFile.delete(); existing = 0
            }
            416 -> { conn.disconnect(); return@withContext } // already complete
            401, 403 -> { conn.disconnect(); throw IOException("Hugging Face refused the download (HTTP $code)") }
            else -> { conn.disconnect(); throw IOException("HTTP $code from ${conn.url.host}") }
        }
        conn.inputStream.use { inp ->
            FileOutputStream(partFile, existing > 0).use { out ->
                copyWithProgress(inp, out, existing, onProgress)
            }
        }
        conn.disconnect()
    }

    private suspend fun copyWithProgress(
        inp: InputStream, out: FileOutputStream, startBytes: Long, onProgress: (ModelProgress) -> Unit,
    ) = withContext(Dispatchers.IO) {
        val buf = ByteArray(1 shl 16)
        var total = startBytes
        var lastReport = 0L
        var windowStart = System.nanoTime()
        var windowBytes = 0L
        var speed = 0L
        while (true) {
            ensureActive()
            val n = inp.read(buf)
            if (n < 0) break
            out.write(buf, 0, n)
            total += n
            windowBytes += n
            val now = System.nanoTime()
            if (now - windowStart > 1_000_000_000L) {
                speed = windowBytes * 1_000_000_000L / (now - windowStart)
                windowStart = now; windowBytes = 0
            }
            if (now - lastReport > 150_000_000L) {
                onProgress(ModelProgress.Transferring(total, ModelInfo.SIZE_BYTES, speed))
                lastReport = now
            }
        }
        out.fd.sync()
    }

    private suspend fun verifyAndInstall(file: File, onProgress: (ModelProgress) -> Unit) = withContext(Dispatchers.IO) {
        val digest = MessageDigest.getInstance("SHA-256")
        val buf = ByteArray(1 shl 20)
        var done = 0L
        file.inputStream().use { inp ->
            while (true) {
                ensureActive()
                val n = inp.read(buf)
                if (n < 0) break
                digest.update(buf, 0, n)
                done += n
                onProgress(ModelProgress.Verifying(done, file.length()))
            }
        }
        val hex = digest.digest().joinToString("") { "%02x".format(it) }
        if (hex != ModelInfo.SHA256) {
            file.delete()
            throw IOException("Checksum mismatch: the file is corrupted or not the expected model. Please try again.")
        }
        modelFile.delete()
        if (!file.renameTo(modelFile)) throw IOException("Couldn't move the model into place")
        verifiedMarker.writeText(markerFor(modelFile))
    }

    private fun markerFor(f: File) = "${ModelInfo.SHA256}:${f.length()}:${f.lastModified()}"

    private fun ensureFreeSpace(needed: Long) {
        val free = StatFs(dir.path).availableBytes
        if (free < needed + 50L * 1024 * 1024) {
            throw IOException(
                "Not enough free storage: need about ${(needed shr 20) + 50} MB, have ${free shr 20} MB."
            )
        }
    }
}
