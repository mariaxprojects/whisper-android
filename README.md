# Whisper 2-bit for Android

An Android app that runs [`diegogranziol/whisper-large-v3-turbo-bpdq-2bit-qat`](https://huggingface.co/diegogranziol/whisper-large-v3-turbo-bpdq-2bit-qat)
entirely on your phone: record speech or open any audio/video file, and get a transcript without anything leaving the device.

- **Record** with the microphone, **open** an audio or video file, or **share** a voice memo to the app from another app.
- Text appears segment by segment as it's recognised; copy or share it when done.
- The 365 MB model downloads once from Hugging Face (resumable, checked against its published SHA-256), then everything works offline.

## What to expect

This is Whisper **large-v3-turbo**, a big model, running on the phone's CPU. The 2-bit format keeps it small (≈365 MB on disk, ≈0.5–0.8 GB of RAM while running), but the compute is still heavy:

- Whisper always encodes audio in **30-second windows**, so even a 3-second clip costs a full window.
- Rough estimate for a recent flagship phone: around real-time speed (a 30 s window in very roughly 30–60 s); mid-range phones will be several times slower. **These are estimates; the app has not yet been run on a physical phone.** The app shows the real speed after each run.
- **Fast mode** (Settings) encodes only as much of the window as a short clip needs, which can make short recordings several times faster. It is experimental: the model author didn't evaluate it that way.
- The 2-bit model was recovery-trained and tested on **English**. Auto language detection is available but other languages may be poor.

**Requirements:** a 64-bit Android phone with Android 9 or newer and an ARMv8.2 CPU (Cortex-A55/A75 or newer, roughly 2018 onwards), about 400 MB of free storage and ideally 4 GB+ of RAM. The app checks the CPU on launch and explains if it can't run.

## Get the APK (no developer tools needed)

The repository includes a GitHub Actions workflow that builds the APK on GitHub's servers.

1. Create a free account at [github.com](https://github.com) if you don't have one, then create a **new repository** (it can be private).
2. Upload this whole folder to it (on the repository page: **Add file → Upload files**, drag in the unzipped folder contents, including the hidden `.github` folder; or use `git push`).
3. Open the **Actions** tab. The *Build APK* workflow starts automatically (or click **Run workflow**). It takes about 10–15 minutes the first time.
4. When it finishes, open the **Releases** section of the repository *on your phone* and tap **BPDQWhisper.apk**.
   (The APK is also attached to the workflow run as an artifact.)

> If uploading through the website, make sure `.github/workflows/build-apk.yml` arrived. Some systems hide folders starting with a dot; you can also create that file through **Add file → Create new file** and paste its contents.

### Install it

1. Tap the downloaded APK. Android will ask you to allow your browser (or Files app) to **install unknown apps**; allow it.
2. Open **Whisper 2-bit**, tap **Download 365 MB** (Wi-Fi recommended), wait for it to finish and verify, and you're ready.

Later builds install over the top and keep the downloaded model, because every build is signed with the same key (`app/sideload.keystore`).

## Build it yourself

**Android Studio:** open this folder, let Gradle sync, plug in your phone (USB debugging on) and press Run. The first build downloads whisper.cpp from GitHub and the model's runtime patch from Hugging Face, so it needs internet access, and `git` must be on your PATH (it applies the patch).

**Command line:** with the Android SDK installed and `ANDROID_HOME` set:

```sh
./gradlew assembleRelease
adb install -r app/build/outputs/apk/release/app-release.apk
```

## Using the app

| | |
|---|---|
| **Microphone** | Tap to record, tap the square to stop. Transcription starts automatically. |
| **Open file** | Any audio or video Android can play (m4a, mp3, wav, ogg, opus, flac, mp4…). Up to 60 minutes. |
| **Share to app** | From a voice recorder, messaging app or file manager, share an audio file to *Transcribe with Whisper 2-bit*. |
| **Stop transcribing** | Stops after the current step and keeps the text recognised so far. |
| **Run again** | Re-transcribes the same audio, e.g. after changing settings. |

**Settings:** CPU threads (defaults to the phone's fast cores), fast mode for short clips, timestamps, automatic language detection, and deleting the model to free space. The settings sheet also shows the native runtime's CPU feature report.

The screen stays on while downloading or transcribing, since Android may pause work when the phone sleeps. If the download is interrupted it resumes where it stopped.

## How it works

The model isn't a standard Whisper file. Its author ships a custom 2-bit format ("BPDQ2": 256 weights per 72-byte block, four FP16 levels per block plus per-matrix column permutations) and a patch to whisper.cpp that adds it. Stock whisper.cpp, Transformers or Q2_K loaders can't read it. The author's prebuilt runtime is for Apple devices only, so this project builds the patched runtime for Android:

1. **`app/src/main/cpp/prepare_whisper.cmake`** (runs during the native build)
   - downloads whisper.cpp at commit `5670d5c0bbcb148feabef84400a07cfca9aa3b30` (the exact commit the patch targets);
   - downloads `runtime-source/whisper-bpdq2.patch` from the model repo and refuses to continue unless its SHA-256 is `0967c05e…b688` (the hash published in the repo's `SHA256SUMS` and `release_manifest.json`);
   - applies it, then swaps the patch's scalar CPU kernel for a NEON one (below). If that swap can't find what it expects, it leaves the author's kernel in place and prints a warning, so the build still succeeds.
2. **`bpdq2_fast.inc`** — an ARM NEON dot product for the 2-bit format. It decodes four weights at a time with a table lookup into the block's four levels, and handles **2×2 tiles** (two weight rows × two audio frames) so each block is decoded once and used twice. It's registered with `nrows = 2`, which ggml's matrix multiply uses automatically for the encoder's shapes.
3. **`jni_bridge.cpp`** — loads the model, runs `whisper_full` with the author's settings (greedy, no temperature fallback, English), streams segments and progress to the UI, and supports cancelling.
4. **Kotlin app** (`app/src/main/java/...`) — model download/verification, recording, file decoding with a band-limited resampler to 16 kHz mono, and the Compose UI.

The app downloads `metal/whisper-bpdq2.bin` (364,741,703 bytes, SHA-256 `1fbdc4bd…32c7`). Despite the folder name, it's the runtime's native model file and isn't Apple-specific; the patched runtime loads it on CPU.

### What was tested before release

- The NEON kernel against the reference scalar loop (900 random cases at Whisper's matrix widths, single-row and 2×2 paths, through an ARM-NEON emulator), plus compilation for a real `aarch64-linux-android` target with `-Wall -Wextra -Werror`.
- The fetch → verify → patch → kernel-swap → build chain end to end on a Linux host, using a stand-in patch with the same structure as the real one; the resulting library exports every JNI entry point.
- `ggml_mul_mat` with the 2-bit type through the modified CPU backend at Whisper's real encoder shapes (1500 frames, and 384 in fast mode), confirming the 2×2 path is taken and results match a double-precision reference.
- The resampler at 8–96 kHz (exact lengths, unity gain at 1 kHz, about −80 dB leakage of a 12 kHz tone).
- Type-checking of the non-UI Kotlin code against the Android API.

**Not yet tested:** the real patch file (Hugging Face wasn't reachable from the build environment), the Compose UI compile, and running on a physical phone. The first GitHub Actions run is the first full build; if it fails, the log will say which step.

## Troubleshooting

- **Build fails at "applying the patch"** — the patch no longer matches; check the model repo for updates. **"could not download/verify"** — Hugging Face or GitHub was unreachable, or the patch changed (the hash check failed on purpose).
- **"The model didn't load"** — the error card shows the runtime's log. Try *Delete model* and download again.
- **App closes during transcription** — usually Android killing it for memory. Close other apps; on 4 GB phones prefer shorter clips.
- **Too slow** — try fewer or more threads (phones vary), enable fast mode for short clips, keep the phone cool and charging.

## Licences

whisper.cpp and the model weights are MIT-licensed (see the model repository's `LICENSE` and `runtime-source/LICENSE.whisper.cpp`). The few Material icons drawn in `AppIcons.kt` are Apache-2.0. The rest of this project is MIT.
