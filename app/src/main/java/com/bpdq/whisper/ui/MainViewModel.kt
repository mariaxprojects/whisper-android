package com.bpdq.whisper.ui

import android.app.Application
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.bpdq.whisper.audio.AudioDecoder
import com.bpdq.whisper.audio.Recorder
import com.bpdq.whisper.audio.WHISPER_SAMPLE_RATE
import com.bpdq.whisper.engine.CpuSupport
import com.bpdq.whisper.engine.Segment
import com.bpdq.whisper.engine.TranscribeOptions
import com.bpdq.whisper.engine.TranscribeOutcome
import com.bpdq.whisper.engine.WhisperEngine
import com.bpdq.whisper.model.ModelManager
import com.bpdq.whisper.model.ModelProgress
import com.bpdq.whisper.model.ModelInfo
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.math.ceil

sealed interface ModelUi {
    data object Checking : ModelUi
    data class Missing(val partialBytes: Long) : ModelUi
    data class Downloading(val bytes: Long, val total: Long, val bytesPerSecond: Long) : ModelUi
    data class Verifying(val fraction: Float) : ModelUi
    data object Ready : ModelUi
}

sealed interface EngineUi {
    data object Idle : EngineUi
    data object Loading : EngineUi
    data class Ready(val systemInfo: String) : EngineUi
    data class Failed(val message: String) : EngineUi
}

sealed interface JobUi {
    data object Idle : JobUi
    data class Recording(val elapsedMs: Long, val level: Float) : JobUi
    data class Decoding(val fraction: Float) : JobUi
    data class Transcribing(val percent: Int, val startedAtMs: Long) : JobUi
}

data class Settings(
    val threads: Int,
    val maxThreads: Int,
    val fastMode: Boolean,
    val timestamps: Boolean,
    val autoLanguage: Boolean,
)

data class RunStats(val audioSeconds: Double, val wallSeconds: Double, val threads: Int, val fastMode: Boolean, val cancelled: Boolean)

data class UiState(
    val cpuProblem: String? = null,
    val ramWarning: String? = null,
    val model: ModelUi = ModelUi.Checking,
    val engine: EngineUi = EngineUi.Idle,
    val job: JobUi = JobUi.Idle,
    val sourceLabel: String? = null,
    val audioSeconds: Double = 0.0,
    val hasAudio: Boolean = false,
    val segments: List<Segment> = emptyList(),
    val lastRun: RunStats? = null,
    val message: String? = null,
    val settings: Settings,
) {
    val busy: Boolean get() = job !is JobUi.Idle
    val engineReady: Boolean get() = engine is EngineUi.Ready
}

class MainViewModel(app: Application) : AndroidViewModel(app) {
    private val prefs = app.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val models = ModelManager(app)
    private val recorder = Recorder()

    private var engine: WhisperEngine? = null
    private var audio: FloatArray? = null
    private var pendingShare: Uri? = null
    private var downloadJob: Job? = null
    private var workJob: Job? = null

    private val _state = MutableStateFlow(
        UiState(
            settings = Settings(
                threads = prefs.getInt("threads", CpuSupport.defaultThreadCount()),
                maxThreads = Runtime.getRuntime().availableProcessors().coerceAtLeast(2),
                fastMode = prefs.getBoolean("fastMode", false),
                timestamps = prefs.getBoolean("timestamps", false),
                autoLanguage = prefs.getBoolean("autoLanguage", false),
            )
        )
    )
    val state: StateFlow<UiState> = _state.asStateFlow()

    init {
        when (val cpu = CpuSupport.check()) {
            is CpuSupport.Result.Unsupported -> _state.update { it.copy(cpuProblem = cpu.reason) }
            CpuSupport.Result.Supported -> {
                val ramGb = CpuSupport.totalRam(app) / 1e9
                if (ramGb < 3.5) {
                    _state.update {
                        it.copy(ramWarning = "This phone has %.1f GB of RAM. The model needs about 0.6–0.8 GB while running, so close other apps if it gets killed.".format(ramGb))
                    }
                }
                if (models.isReady()) {
                    _state.update { it.copy(model = ModelUi.Ready) }
                    loadEngine()
                } else {
                    _state.update { it.copy(model = ModelUi.Missing(models.partialBytes())) }
                }
            }
        }
    }

    // ---------------------------------------------------------------- model

    fun downloadModel() {
        if (downloadJob?.isActive == true) return
        downloadJob = viewModelScope.launch {
            _state.update { it.copy(message = null, model = ModelUi.Downloading(models.partialBytes(), ModelInfo.SIZE_BYTES, 0)) }
            try {
                models.download(::onModelProgress)
                _state.update { it.copy(model = ModelUi.Ready) }
                loadEngine()
            } catch (e: CancellationException) {
                _state.update { it.copy(model = ModelUi.Missing(models.partialBytes())) }
                throw e
            } catch (e: Exception) {
                _state.update { it.copy(model = ModelUi.Missing(models.partialBytes()), message = e.message ?: e.toString()) }
            }
        }
    }

    fun cancelDownload() {
        downloadJob?.cancel()
    }

    fun importModel(uri: Uri) {
        if (downloadJob?.isActive == true) return
        downloadJob = viewModelScope.launch {
            _state.update { it.copy(message = null, model = ModelUi.Downloading(0, ModelInfo.SIZE_BYTES, 0)) }
            try {
                models.importFrom(getApplication<Application>(), uri, ::onModelProgress)
                _state.update { it.copy(model = ModelUi.Ready) }
                loadEngine()
            } catch (e: CancellationException) {
                _state.update { it.copy(model = ModelUi.Missing(models.partialBytes())) }
                throw e
            } catch (e: Exception) {
                _state.update { it.copy(model = ModelUi.Missing(models.partialBytes()), message = e.message ?: e.toString()) }
            }
        }
    }

    fun deleteModel() {
        if (state.value.busy) return
        viewModelScope.launch {
            engine?.release()
            engine = null
            models.delete()
            _state.update { it.copy(engine = EngineUi.Idle, model = ModelUi.Missing(0)) }
        }
    }

    private fun onModelProgress(p: ModelProgress) {
        _state.update {
            it.copy(
                model = when (p) {
                    is ModelProgress.Transferring -> ModelUi.Downloading(p.bytes, p.total, p.bytesPerSecond)
                    is ModelProgress.Verifying -> ModelUi.Verifying(p.bytes.toFloat() / p.total.coerceAtLeast(1))
                }
            )
        }
    }

    // --------------------------------------------------------------- engine

    private fun loadEngine() {
        if (engine != null || state.value.engine == EngineUi.Loading) return
        _state.update { it.copy(engine = EngineUi.Loading) }
        viewModelScope.launch {
            try {
                val e = WhisperEngine.load(models.modelFile)
                engine = e
                _state.update { it.copy(engine = EngineUi.Ready(WhisperEngine.systemInfo())) }
                pendingShare?.let { uri -> pendingShare = null; openAudio(uri) }
            } catch (e: Throwable) {
                _state.update { it.copy(engine = EngineUi.Failed(e.message ?: e.toString())) }
            }
        }
    }

    fun retryEngine() {
        _state.update { it.copy(engine = EngineUi.Idle) }
        loadEngine()
    }

    // ---------------------------------------------------------------- audio

    fun startRecording() {
        if (state.value.busy || !state.value.engineReady) return
        workJob = viewModelScope.launch {
            _state.update { it.copy(message = null, job = JobUi.Recording(0, 0f)) }
            try {
                val samples = recorder.record(
                    onLevel = { lvl -> _state.update { s -> (s.job as? JobUi.Recording)?.let { s.copy(job = it.copy(level = lvl)) } ?: s } },
                    onElapsedMs = { ms -> _state.update { s -> (s.job as? JobUi.Recording)?.let { s.copy(job = it.copy(elapsedMs = ms)) } ?: s } },
                )
                if (samples.size < WHISPER_SAMPLE_RATE / 2) {
                    _state.update { it.copy(job = JobUi.Idle, message = "That recording was too short.") }
                    return@launch
                }
                setAudio(samples, "Recording")
                transcribeCurrent()
            } catch (e: CancellationException) {
                _state.update { it.copy(job = JobUi.Idle) }
                throw e
            } catch (e: Exception) {
                _state.update { it.copy(job = JobUi.Idle, message = e.message ?: e.toString()) }
            }
        }
    }

    fun stopRecording() = recorder.stop()

    fun openAudio(uri: Uri) {
        if (!state.value.engineReady) { pendingShare = uri; return }
        if (state.value.busy) return
        workJob = viewModelScope.launch {
            _state.update { it.copy(message = null, job = JobUi.Decoding(0f)) }
            try {
                val samples = AudioDecoder.decode(getApplication<Application>(), uri) { f ->
                    _state.update { s -> if (s.job is JobUi.Decoding) s.copy(job = JobUi.Decoding(f)) else s }
                }
                setAudio(samples, displayName(uri))
                if (samples.size.toLong() >= AudioDecoder.MAX_SECONDS.toLong() * WHISPER_SAMPLE_RATE) {
                    _state.update { it.copy(message = "Only the first ${AudioDecoder.MAX_SECONDS / 60} minutes will be transcribed.") }
                }
                transcribeCurrent()
            } catch (e: CancellationException) {
                _state.update { it.copy(job = JobUi.Idle) }
                throw e
            } catch (e: Exception) {
                _state.update { it.copy(job = JobUi.Idle, message = "Couldn't read that file: ${e.message ?: e}") }
            }
        }
    }

    /** Runs the model again on the current audio, e.g. after changing settings. */
    fun retranscribe() {
        if (state.value.busy || audio == null || !state.value.engineReady) return
        workJob = viewModelScope.launch { transcribeCurrent() }
    }

    fun cancelTranscription() {
        engine?.cancel()
    }

    private fun setAudio(samples: FloatArray, label: String) {
        audio = samples
        _state.update {
            it.copy(
                hasAudio = true,
                sourceLabel = label,
                audioSeconds = samples.size.toDouble() / WHISPER_SAMPLE_RATE,
                segments = emptyList(),
                lastRun = null,
            )
        }
    }

    private suspend fun transcribeCurrent() {
        val e = engine ?: return
        val samples = audio ?: return
        val s = state.value.settings
        val seconds = samples.size.toDouble() / WHISPER_SAMPLE_RATE
        val audioCtx = if (s.fastMode) fastAudioCtx(seconds) else 0
        val started = System.currentTimeMillis()
        _state.update { it.copy(segments = emptyList(), lastRun = null, job = JobUi.Transcribing(0, started)) }

        val outcome = try {
            e.transcribe(
                samples,
                TranscribeOptions(threads = s.threads, language = if (s.autoLanguage) "auto" else "en", audioCtx = audioCtx),
                progressCallback = { p -> _state.update { st -> (st.job as? JobUi.Transcribing)?.let { st.copy(job = it.copy(percent = p)) } ?: st } },
                segmentCallback = { seg -> _state.update { st -> st.copy(segments = st.segments + seg) } },
            )
        } catch (ex: Exception) {
            TranscribeOutcome.Failed(ex.message ?: ex.toString())
        }
        val wall = (System.currentTimeMillis() - started) / 1000.0
        _state.update {
            when (outcome) {
                is TranscribeOutcome.Done -> it.copy(job = JobUi.Idle, segments = outcome.segments,
                    lastRun = RunStats(seconds, wall, s.threads, audioCtx > 0, cancelled = false))
                is TranscribeOutcome.Cancelled -> it.copy(job = JobUi.Idle, segments = outcome.segments,
                    lastRun = RunStats(seconds, wall, s.threads, audioCtx > 0, cancelled = true))
                is TranscribeOutcome.Failed -> it.copy(job = JobUi.Idle, message = outcome.message)
            }
        }
    }

    /**
     * Shorter encoder window for clips under 30 s. whisper.cpp always encodes a
     * full 30 s window (1500 frames) otherwise, which dominates the run time
     * for short clips. Experimental: can reduce accuracy.
     */
    private fun fastAudioCtx(seconds: Double): Int {
        if (seconds >= 29.0) return 0
        val frames = ceil(seconds / 30.0 * 1500).toInt() + 128
        val aligned = ((frames + 63) / 64) * 64
        return aligned.coerceIn(256, 1500).let { if (it >= 1500) 0 else it }
    }

    // ------------------------------------------------------------- settings

    fun updateSettings(transform: (Settings) -> Settings) {
        _state.update { it.copy(settings = transform(it.settings)) }
        val s = state.value.settings
        prefs.edit()
            .putInt("threads", s.threads)
            .putBoolean("fastMode", s.fastMode)
            .putBoolean("timestamps", s.timestamps)
            .putBoolean("autoLanguage", s.autoLanguage)
            .apply()
    }

    fun dismissMessage() = _state.update { it.copy(message = null) }

    fun showMessage(text: String) = _state.update { it.copy(message = text) }

    private fun displayName(uri: Uri): String {
        return try {
            getApplication<Application>().contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
        } catch (e: Exception) {
            null
        } ?: "Audio file"
    }

    @OptIn(DelicateCoroutinesApi::class)
    override fun onCleared() {
        engine?.cancel()
        workJob?.cancel()
        val e = engine
        engine = null
        // Free the ~0.5 GB native context off the main thread, after any run has stopped.
        if (e != null) GlobalScope.launch { e.release() }
    }
}

fun formatClock(ms: Long): String {
    val totalSec = ms / 1000
    val h = totalSec / 3600
    val m = (totalSec % 3600) / 60
    val s = totalSec % 60
    return if (h > 0) "%d:%02d:%02d".format(h, m, s) else "%d:%02d".format(m, s)
}

/** The transcript as plain text, with or without "[start → end]" timestamps. */
fun transcriptOf(state: UiState): String {
    val segs = state.segments
    return if (state.settings.timestamps) {
        segs.joinToString("\n") { "[${formatClock(it.startMs)} → ${formatClock(it.endMs)}] ${it.text.trim()}" }
    } else {
        segs.joinToString("") { it.text }.trim()
    }
}
