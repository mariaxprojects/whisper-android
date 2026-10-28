@file:OptIn(ExperimentalMaterial3Api::class)

package com.bpdq.whisper.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LargeFloatingActionButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.bpdq.whisper.model.ModelInfo
import kotlinx.coroutines.delay
import kotlin.math.roundToInt

@Composable
fun MainScreen(vm: MainViewModel) {
    val state by vm.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    var showSettings by remember { mutableStateOf(false) }

    // Keep the screen on while something long is running, so the phone doesn't sleep mid-way.
    val view = LocalView.current
    val keepOn = state.busy || state.model is ModelUi.Downloading || state.model is ModelUi.Verifying
    DisposableEffect(keepOn) {
        view.keepScreenOn = keepOn
        onDispose { view.keepScreenOn = false }
    }

    val pickAudio = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(vm::openAudio)
    }
    val pickModel = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let(vm::importModel)
    }
    val micPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) vm.startRecording() else vm.showMessage("Microphone access is needed to record. You can still open audio files.")
    }
    val onMic = {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            vm.startRecording()
        } else {
            micPermission.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Whisper 2-bit", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                        Text(
                            "large-v3-turbo · runs on this phone",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                },
                actions = {
                    if (state.cpuProblem == null) {
                        IconButton(onClick = { showSettings = true }) {
                            Icon(Icons.Filled.Settings, contentDescription = "Settings")
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
            )
        },
        bottomBar = {
            if (state.cpuProblem == null && state.engineReady) {
                ControlBar(
                    state = state,
                    onMic = onMic,
                    onStop = vm::stopRecording,
                    onCancel = vm::cancelTranscription,
                    onOpenFile = { pickAudio.launch(arrayOf("audio/*", "video/*")) },
                )
            }
        },
    ) { padding ->
        Column(
            Modifier
                .padding(padding)
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            val cpuProblem = state.cpuProblem
            state.message?.let { NoticeCard(title = null, body = it, isError = true, onDismiss = vm::dismissMessage) }

            if (cpuProblem != null) {
                NoticeCard(title = "This phone can't run the model", body = cpuProblem, isError = true)
            } else if (state.model !is ModelUi.Ready) {
                ModelSetupCard(
                    model = state.model,
                    onDownload = vm::downloadModel,
                    onCancel = vm::cancelDownload,
                    onImport = { pickModel.launch(arrayOf("*/*")) },
                )
                state.ramWarning?.let { NoticeCard(title = null, body = it, isError = false) }
            } else when (val engine = state.engine) {
                EngineUi.Idle, EngineUi.Loading -> LoadingCard()
                is EngineUi.Failed -> EngineErrorCard(engine.message, onRetry = vm::retryEngine, onRedownload = vm::deleteModel)
                is EngineUi.Ready -> TranscriptArea(
                    state = state,
                    transcript = transcriptOf(state),
                    onRerun = vm::retranscribe,
                )
            }
            Spacer(Modifier.height(24.dp))
        }
    }

    if (showSettings) {
        ModalBottomSheet(onDismissRequest = { showSettings = false }) {
            SettingsSheet(
                settings = state.settings,
                engine = state.engine,
                modelReady = state.model is ModelUi.Ready,
                busy = state.busy,
                onChange = vm::updateSettings,
                onDeleteModel = { vm.deleteModel(); showSettings = false },
            )
        }
    }
}

// ------------------------------------------------------------------ model setup

@Composable
private fun ModelSetupCard(model: ModelUi, onDownload: () -> Unit, onCancel: () -> Unit, onImport: () -> Unit) {
    val totalMb = ModelInfo.SIZE_BYTES / 1_000_000
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLowest),
        shape = RoundedCornerShape(20.dp),
        modifier = Modifier.fillMaxWidth().animateContentSize(),
    ) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("One-time setup", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.secondary)
            Text("Get the speech model", style = MaterialTheme.typography.headlineSmall)
            Text(
                "Whisper large-v3-turbo, compressed to 2 bits per weight ($totalMb MB). " +
                    "It downloads once from Hugging Face and is checked against its published fingerprint. " +
                    "After that, transcription works offline and audio never leaves the phone.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            when (model) {
                is ModelUi.Downloading -> {
                    val frac = model.bytes.toFloat() / model.total.coerceAtLeast(1)
                    LinearProgressIndicator(progress = { frac }, modifier = Modifier.fillMaxWidth().height(6.dp))
                    val eta = if (model.bytesPerSecond > 0) (model.total - model.bytes) / model.bytesPerSecond else -1
                    Text(
                        buildString {
                            append("${model.bytes / 1_000_000} of ${model.total / 1_000_000} MB")
                            if (model.bytesPerSecond > 0) append(" · %.1f MB/s".format(model.bytesPerSecond / 1e6))
                            if (eta >= 0) append(" · ${formatEta(eta)} left")
                        },
                        style = MonoStyle,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    OutlinedButton(onClick = onCancel) { Text("Pause") }
                }
                is ModelUi.Verifying -> {
                    LinearProgressIndicator(progress = { model.fraction }, modifier = Modifier.fillMaxWidth().height(6.dp))
                    Text("Checking the file's fingerprint…", style = MonoStyle, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                is ModelUi.Missing -> {
                    Button(onClick = onDownload, modifier = Modifier.fillMaxWidth().height(52.dp)) {
                        Icon(AppIcons.Download, contentDescription = null, modifier = Modifier.size(20.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(
                            if (model.partialBytes > 0) "Resume download (${model.partialBytes / 1_000_000} MB done)"
                            else "Download $totalMb MB"
                        )
                    }
                    Text(
                        "Wi-Fi recommended. You can pause and resume.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                    TextButton(onClick = onImport) { Text("Already have whisper-bpdq2.bin? Import it") }
                }
                else -> Unit
            }
        }
    }
}

private fun formatEta(seconds: Long): String = when {
    seconds < 60 -> "${seconds}s"
    seconds < 3600 -> "${seconds / 60} min"
    else -> "%dh %02dm".format(seconds / 3600, (seconds % 3600) / 60)
}

@Composable
private fun LoadingCard() {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLowest),
        shape = RoundedCornerShape(20.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(Modifier.padding(20.dp), verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(Modifier.size(28.dp), strokeWidth = 3.dp)
            Spacer(Modifier.width(16.dp))
            Column {
                Text("Loading the model", style = MaterialTheme.typography.titleMedium)
                Text(
                    "Reading 365 MB into memory. This takes a few seconds.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun EngineErrorCard(message: String, onRetry: () -> Unit, onRedownload: () -> Unit) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
        shape = RoundedCornerShape(20.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("The model didn't load", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.onErrorContainer)
            SelectionContainer {
                Text(message, style = MonoStyle, color = MaterialTheme.colorScheme.onErrorContainer)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = onRetry) { Text("Try again") }
                OutlinedButton(onClick = onRedownload) { Text("Delete model") }
            }
        }
    }
}

@Composable
private fun NoticeCard(title: String?, body: String, isError: Boolean, onDismiss: (() -> Unit)? = null) {
    val bg = if (isError) MaterialTheme.colorScheme.errorContainer else MaterialTheme.colorScheme.secondaryContainer
    val fg = if (isError) MaterialTheme.colorScheme.onErrorContainer else MaterialTheme.colorScheme.onSecondaryContainer
    Card(colors = CardDefaults.cardColors(containerColor = bg), shape = RoundedCornerShape(16.dp), modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(start = 16.dp, top = 12.dp, bottom = 12.dp, end = 4.dp), verticalAlignment = Alignment.Top) {
            Column(Modifier.weight(1f).padding(top = 4.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                title?.let { Text(it, style = MaterialTheme.typography.titleMedium, color = fg) }
                Text(body, style = MaterialTheme.typography.bodyMedium, color = fg)
            }
            if (onDismiss != null) {
                IconButton(onClick = onDismiss) { Icon(Icons.Filled.Close, contentDescription = "Dismiss", tint = fg) }
            }
        }
    }
}

// --------------------------------------------------------------- transcript

@Composable
private fun TranscriptArea(state: UiState, transcript: String, onRerun: () -> Unit) {
    val job = state.job

    if (!state.hasAudio && job is JobUi.Idle) {
        EmptyState()
        return
    }

    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLowest),
        shape = RoundedCornerShape(20.dp),
        modifier = Modifier.fillMaxWidth().animateContentSize(),
    ) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            if (job is JobUi.Recording) {
                RecordingPanel(job)
            } else {
                TranscriptBody(state, transcript, onRerun)
            }
        }
    }
}

@Composable
private fun TranscriptBody(state: UiState, transcript: String, onRerun: () -> Unit) {
    val context = LocalContext.current
    val clipboard = LocalClipboardManager.current
    val job = state.job
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            // Header: what we're transcribing
            if (state.hasAudio) {
                Text(
                    "${state.sourceLabel ?: "Audio"} · ${formatClock((state.audioSeconds * 1000).toLong())}",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            when (job) {
                is JobUi.Decoding -> {
                    LinearProgressIndicator(progress = { job.fraction }, modifier = Modifier.fillMaxWidth())
                    Text("Reading the audio file…", style = MonoStyle, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                is JobUi.Transcribing -> TranscribingStatus(job, state.audioSeconds, waitingForText = state.segments.isEmpty())
                else -> Unit
            }

            // The transcript itself
            if (state.segments.isNotEmpty()) {
                SelectionContainer {
                    if (state.settings.timestamps) {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            state.segments.forEach { seg ->
                                Row {
                                    Text(
                                        formatClock(seg.startMs),
                                        style = MonoStyle,
                                        color = MaterialTheme.colorScheme.secondary,
                                        modifier = Modifier.width(56.dp).padding(top = 5.dp),
                                    )
                                    Text(seg.text.trim(), style = MaterialTheme.typography.bodyLarge)
                                }
                            }
                        }
                    } else {
                        Text(transcript, style = MaterialTheme.typography.bodyLarge)
                    }
                }
            } else if (job is JobUi.Idle && state.lastRun != null) {
                Text("No speech was recognised.", style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }

            // Stats + actions after a run
            state.lastRun?.let { run ->
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Text(runSummary(run), style = MonoStyle, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (job is JobUi.Idle && state.hasAudio) {
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically) {
                    ActionChip(AppIcons.Copy, "Copy", enabled = transcript.isNotBlank()) {
                        clipboard.setText(AnnotatedString(transcript))
                        Toast.makeText(context, "Copied", Toast.LENGTH_SHORT).show()
                    }
                    ActionChip(Icons.Filled.Share, "Share", enabled = transcript.isNotBlank()) {
                        val send = Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, transcript)
                        context.startActivity(Intent.createChooser(send, "Share transcript"))
                    }
                    ActionChip(AppIcons.Replay, "Run again", enabled = true, onClick = onRerun)
                }
            }
    }
}

private fun runSummary(run: RunStats): String {
    val ratio = run.wallSeconds / run.audioSeconds.coerceAtLeast(0.01)
    val speed = if (ratio <= 1.0) "%.1f× faster than real time".format(1 / ratio.coerceAtLeast(0.001))
    else "%.1f× slower than real time".format(ratio)
    return buildString {
        if (run.cancelled) append("Stopped early · ")
        append("%.1f s of audio in %.1f s · %s · %d threads".format(run.audioSeconds, run.wallSeconds, speed, run.threads))
        if (run.fastMode) append(" · fast mode")
    }
}

@Composable
private fun ActionChip(icon: ImageVector, label: String, enabled: Boolean, onClick: () -> Unit) {
    TextButton(onClick = onClick, enabled = enabled) {
        Icon(icon, contentDescription = null, modifier = Modifier.size(18.dp))
        Spacer(Modifier.width(6.dp))
        Text(label)
    }
}

@Composable
private fun TranscribingStatus(job: JobUi.Transcribing, audioSeconds: Double, waitingForText: Boolean) {
    var now by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(job.startedAtMs) {
        while (true) { now = System.currentTimeMillis(); delay(500) }
    }
    val elapsed = (now - job.startedAtMs).coerceAtLeast(0)
    if (job.percent in 1..99) {
        LinearProgressIndicator(progress = { job.percent / 100f }, modifier = Modifier.fillMaxWidth())
    } else {
        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
    }
    Text(
        buildString {
            append("Transcribing · ${formatClock(elapsed)}")
            if (job.percent in 1..99) append(" · ${job.percent}%")
        },
        style = MonoStyle,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
    if (waitingForText) {
        Text(
            if (audioSeconds > 30) "Audio is processed in 30-second windows; text appears as each window finishes."
            else "Text appears when the clip has been processed. On most phones that takes longer than the clip itself.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun RecordingPanel(job: JobUi.Recording) {
    val history = remember { mutableStateListOf<Float>() }
    LaunchedEffect(job.level, job.elapsedMs) {
        history.add(job.level)
        while (history.size > 48) history.removeAt(0)
    }
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.fillMaxWidth()) {
        Text("Listening", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.secondary)
        Spacer(Modifier.height(4.dp))
        Text(formatClock(job.elapsedMs), style = MaterialTheme.typography.displaySmall, fontWeight = FontWeight.Light)
        Spacer(Modifier.height(16.dp))
        val barColor = MaterialTheme.colorScheme.secondary
        val idleColor = MaterialTheme.colorScheme.outlineVariant
        Canvas(Modifier.fillMaxWidth().height(56.dp)) {
            val n = 48
            val gap = 3.dp.toPx()
            val w = (size.width - gap * (n - 1)) / n
            for (i in 0 until n) {
                val idx = history.size - n + i
                val lvl = if (idx >= 0) history[idx] else 0f
                // Perceptual-ish scaling so normal speech fills the meter.
                val h = (kotlin.math.sqrt(lvl.coerceIn(0f, 1f)) * size.height).coerceAtLeast(3.dp.toPx())
                drawRoundRect(
                    color = if (idx >= 0) barColor else idleColor,
                    topLeft = Offset(i * (w + gap), (size.height - h) / 2),
                    size = Size(w, h),
                    cornerRadius = CornerRadius(w / 2, w / 2),
                )
            }
        }
        Spacer(Modifier.height(12.dp))
        Text(
            "Tap stop when you're done. Transcription starts automatically.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun EmptyState() {
    Column(
        Modifier.fillMaxWidth().padding(top = 48.dp, start = 8.dp, end = 8.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Speak, or open a recording.", style = MaterialTheme.typography.headlineSmall)
        Text(
            "Tap the microphone to record, or open any audio or video file. You can also share a voice memo " +
                "to this app from another app.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            "Everything runs on this phone's processor. It's a large model, so expect it to be slower than real time on most phones; " +
                "\"fast mode\" in settings helps with short clips.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

// --------------------------------------------------------------- bottom bar

@Composable
private fun ControlBar(state: UiState, onMic: () -> Unit, onStop: () -> Unit, onCancel: () -> Unit, onOpenFile: () -> Unit) {
    val job = state.job
    Surface(color = MaterialTheme.colorScheme.surfaceContainer, tonalElevation = 2.dp) {
        Box(
            Modifier.fillMaxWidth().navigationBarsPadding().padding(horizontal = 24.dp, vertical = 14.dp),
            contentAlignment = Alignment.Center,
        ) {
            AnimatedVisibility(visible = job is JobUi.Idle, modifier = Modifier.align(Alignment.CenterStart)) {
                OutlinedButton(onClick = onOpenFile) {
                    Icon(AppIcons.Folder, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("Open file")
                }
            }
            when (job) {
                is JobUi.Recording -> LargeFloatingActionButton(
                    onClick = onStop,
                    containerColor = MaterialTheme.colorScheme.secondary,
                    contentColor = MaterialTheme.colorScheme.onSecondary,
                    shape = RoundedCornerShape(28.dp),
                ) { Icon(AppIcons.Stop, contentDescription = "Stop recording", modifier = Modifier.size(36.dp)) }

                is JobUi.Transcribing, is JobUi.Decoding -> OutlinedButton(
                    onClick = onCancel,
                    enabled = job is JobUi.Transcribing,
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error),
                    modifier = Modifier.height(56.dp),
                ) {
                    Icon(Icons.Filled.Close, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Stop transcribing")
                }

                JobUi.Idle -> Box(Modifier.align(Alignment.CenterEnd)) {
                    LargeFloatingActionButton(
                        onClick = onMic,
                        containerColor = MaterialTheme.colorScheme.primary,
                        contentColor = MaterialTheme.colorScheme.onPrimary,
                        shape = RoundedCornerShape(28.dp),
                    ) { Icon(AppIcons.Mic, contentDescription = "Record", modifier = Modifier.size(36.dp)) }
                }
            }
        }
    }
}

// ------------------------------------------------------------------ settings

@Composable
private fun SettingsSheet(
    settings: Settings,
    engine: EngineUi,
    modelReady: Boolean,
    busy: Boolean,
    onChange: ((Settings) -> Settings) -> Unit,
    onDeleteModel: () -> Unit,
) {
    Column(
        Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = 24.dp).padding(bottom = 32.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        Text("Settings", style = MaterialTheme.typography.headlineSmall)

        Column {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("CPU threads", style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
                Text("${settings.threads}", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.secondary)
            }
            Slider(
                value = settings.threads.toFloat(),
                onValueChange = { v -> onChange { it.copy(threads = v.roundToInt()) } },
                valueRange = 1f..settings.maxThreads.toFloat(),
                steps = (settings.maxThreads - 2).coerceAtLeast(0),
            )
            Text(
                "Usually fastest at the number of performance cores. More isn't always better on phones.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        SwitchRow(
            title = "Fast mode for short clips",
            body = "Encodes only as much of the 30-second window as the clip needs. Much faster for short recordings, " +
                "but experimental: the model wasn't evaluated this way, so accuracy may drop.",
            checked = settings.fastMode,
        ) { v -> onChange { it.copy(fastMode = v) } }

        SwitchRow(
            title = "Show timestamps",
            body = "Show when each segment starts. Copy and share include them too.",
            checked = settings.timestamps,
        ) { v -> onChange { it.copy(timestamps = v) } }

        SwitchRow(
            title = "Detect language automatically",
            body = "Off = English. This 2-bit model was recovery-trained and tested on English only; other languages may be poor.",
            checked = settings.autoLanguage,
        ) { v -> onChange { it.copy(autoLanguage = v) } }

        HorizontalDivider()

        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Model", style = MaterialTheme.typography.titleMedium)
            Text(
                "${ModelInfo.REPO}\nmetal/${ModelInfo.FILE_NAME} · ${ModelInfo.SIZE_BYTES / 1_000_000} MB · MIT licence",
                style = MonoStyle, color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (engine is EngineUi.Ready) {
                Text(engine.systemInfo.trim(), style = MonoStyle, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (modelReady) {
                OutlinedButton(
                    onClick = onDeleteModel,
                    enabled = !busy,
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error),
                ) { Text("Delete model (frees ${ModelInfo.SIZE_BYTES / 1_000_000} MB)") }
            }
        }
    }
}

@Composable
private fun SwitchRow(title: String, body: String, checked: Boolean, onChecked: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f).padding(end = 16.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(body, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(checked = checked, onCheckedChange = onChecked)
    }
}
