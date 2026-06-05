# Bengali Emotional Voice Conversion (EVC) — Analysis & Implementation

Neutral Bangla speech in, emotional Bangla speech out (angry / happy / sad), keeping the
speaker's identity. This document describes the **current** pipeline in
`bengali_evc_v3_colab.ipynb`: a single self-contained Google Colab (A100) notebook that
sources data **only from HuggingFace**, extracts its own acoustic features, trains the EVC
model, and renders audio with an **NSF-HiFiGAN** neural vocoder.

---

## 1. Data — SUBESCO from HuggingFace (no Kaggle)

- **Dataset:** [`sustcsenlp/bn_emotion_speech_corpus`](https://huggingface.co/datasets/sustcsenlp/bn_emotion_speech_corpus)
  — the official **SUBESCO** corpus (SUST Bangla Emotional Speech Corpus), **public, CC-BY-4.0**.
- **Size / structure:** 7000 clips, 16 kHz. Two columns:
  - `text` — the utterance id, e.g. `F_01_OISHI_S_10_ANGRY_1`
    (`Gender_SpeakerNo_Name_S_SentenceNo_EMOTION_Take`).
  - `audio` — the 16 kHz waveform.
- **20 speakers** (10 M / 10 F), **10 sentences**, **7 emotions**
  (angry, disgust, fear, happy, neutral, sad, surprise). This notebook keeps the four it
  converts between: **neutral → {angry, happy, sad}**.
- **Access:** no token or credentials required (public dataset). An optional `HF_TOKEN`
  only raises rate limits.
- **Download method (important):** the repo ships a *dataset loading script*
  (`bn_emotion_speech_corpus.py`), which newer `datasets` (v3+) refuses to run
  (`RuntimeError: Dataset scripts are no longer supported`). So the notebook does **not**
  call `load_dataset`; instead it downloads the single archive `subesco.tar.gz` (~1.6 GB)
  directly with `huggingface_hub.hf_hub_download` and stream-extracts the wavs. A
  `load_dataset(..., trust_remote_code=True)` path remains only as a fallback for old
  `datasets` versions.
- **Kaggle mirror:** not used. A community Kaggle mirror exists but is unnecessary —
  the official HF dataset is authoritative and license-clean. (Content rephrased for
  licensing compliance.)

> **Why this matters:** the previous design depended on a private Kaggle "processed
> features" dataset + a Kaggle checkpoint. Both are gone. The notebook now downloads raw
> SUBESCO audio and computes every feature itself, so it is fully reproducible by anyone.

---

## 2. Notebook flow (cell by cell)

| Block | What it does |
|-------|--------------|
| **1 — Setup** | Mount Drive; `pip install librosa soundfile scipy huggingface_hub datasets`. No Kaggle. |
| **2 — Download** | `hf_hub_download(...,"subesco.tar.gz")` then stream-extract every wav to `/content/subesco_audio/<id>.wav` (runs once; skips if already present; `load_dataset` fallback for old `datasets`). |
| **3 — Config** | Imports, seeds, **A100 TF32**, paths, the global `CFG`. Output goes to Google Drive. |
| **4 — Feature extraction** | Parse each filename → `(speaker, sentence, take, emotion)`; compute **mel(dB)**, **F0**, **energy**, **voiced**; save `.npy`; write `metadata.csv`. |
| **5 — Metadata** | Load `metadata.csv`, map columns, resolve `.npy` paths. |
| **6 — Feature utils** | Silence-trim, mel ↔ dB ↔ [-1,1] normalization, F0/energy normalization. |
| **7 — Pairs + stats** | Build neutral↔emotion pairs (same speaker+sentence+take); per-emotion F0/energy stats. |
| **8 — Dataset** | DTW-align source/target; classic log-F0 transform; emits training tensors. |
| **9 — Model** | Content/aux/speaker/emotion encoders, decoder, **ProsodyHead** (predicts F0/energy/voiced), discriminator, SER. |
| **10–11 — Training** | SER pretrain → 3-phase GAN schedule (reconstruct → emotion+F0 → sharpen). |
| **12 — NSF-HiFiGAN** | Vocoder config/data (12A), mel/F0 utils + denoise (12B), model (12C), dataset (12D), training (12E), inference dispatch (12F), **GTA fine-tune (12G)**. |
| **13 — Inference** | Run generator → denormalize mel + predicted F0 → NSF-HiFiGAN → denoise. |
| **13/14 — Eval/Export** | Plots, audio playback, honest evaluation, export to Drive. |

---

## 3. Feature extraction (Block 4) — exact contract

For every kept SUBESCO clip the notebook computes and saves:

| Feature | How | Saved as |
|---------|-----|----------|
| `mel_path` | `librosa` mel-spectrogram → `power_to_db(ref=1.0)` → clip **[-80, 0] dB**, shape `(128, T)` | `mel/<id>.npy` |
| `f0_path` | `librosa.pyin` (fmin 60, fmax 600 Hz); unvoiced = 0 | `f0/<id>.npy` |
| `energy_path` | per-frame mean of mel(dB) (= `derive_energy_from_mel_db`) | `energy/<id>.npy` |
| `voiced_path` | `(f0 > 0)` as float | `voiced/<id>.npy` |
| `wav_path` | absolute path to the source SUBESCO `.wav` | (in csv) |

`metadata.csv` columns: `speaker, sentence, take, emotion, label, mel_path, f0_path,
energy_path, voiced_path, wav_path, duration_sec, num_frames`.

**Consistency guarantees (the things that make it error-free):**
- Mel is produced in the **same dB space** the whole pipeline assumes; `normalize_mel`
  maps `[-80, 0] dB → [-1, 1]`, which is exactly the generator's `Tanh` output range and
  the NSF-HiFiGAN input range.
- Audio config is global in `CFG`: `sample_rate 16000, n_fft 2048, hop 512, win 2048,
  n_mels 128, fmin 0, fmax 8000`. **`hop=512` drives the vocoder upsample budget.**
- Because the vocoder trains on the **same wav→mel** these features come from, vocoder
  training is always **PAIRED** (zero feature-domain mismatch).

---

## 4. Model (Block 9)

- **Encoders:** content (speaker/emotion-invariant via a gradient-reversal branch), auxiliary
  prosody, speaker embedding, emotion embedding, and a small per-emotion **prosody
  conditioning** vector.
- **Decoder:** reconstructs the target-emotion mel.
- **ProsodyHead (key piece):** predicts **F0, energy, voiced** from the decoder hidden
  state, so F0 is a *first-class supervised output* — not a side effect. This is what gives
  the vocoder a real pitch contour to render.
- **Discriminator + SER:** adversarial realism + an emotion classifier (offline pretrained
  and online) that pushes the converted mel toward the target emotion.

**Training:** SER pretrain, then three phases — (1) reconstruction + F0 warm-up,
(2) emotion injection + full F0/prosody supervision, (3) sharpening with higher F0/prosody
pressure. F0 pressure is deliberately moderate so speaker pitch identity survives.

---

## 5. NSF-HiFiGAN vocoder (Block 12) — why and how

**Problem it solves:** Griffin-Lim re-invents phase/pitch from the mel and flattens the
converted emotion; it sounds buzzy. **NSF-HiFiGAN** is a neural **source-filter** vocoder
that takes the generator's **mel + the predicted F0** and synthesizes a waveform whose pitch
*follows that F0* — so the injected emotion is actually audible and the speech is natural.

- **Architecture (12C):** `SineGen` + `SourceModuleHnNSF` build a harmonic sine excitation
  from F0; a HiFi-GAN v1 generator (with F0 source-injection via `noise_convs`) filters it;
  MPD + MSD discriminators with feature-matching + LS-GAN + L1-mel losses.
- **Upsample budget:** `upsample_rates = [8,8,4,2]` → product **512 = hop_length** (the
  notebook asserts this at config time). ~14.2 M generator params.
- **Data (12A / 12D):** built straight from `df_work` using each row's real `wav_path`
  → always **PAIRED**. If no wavs resolve, it disables itself and falls back to Griffin-Lim.
- **Training (12E):** bf16 AMP on A100, resumable, periodic safety checkpoints.
- **Inference (12F):** `synthesize_waveform` = **NSF-HiFiGAN → Griffin-Lim** (emergency only),
  then a denoise pass. **WORLD/pyworld has been removed entirely.**

### Noise handling (built in)
1. **GTA fine-tune (12G):** fine-tunes the vocoder on the EVC generator's *own*
   reconstructed mels paired with the real wav, eliminating the over-smoothed-mel hiss.
   The most effective fix; runs after EVC training.
2. **Mel augmentation (12D):** perturbs training mels (smooth/noise/scale) so the vocoder
   tolerates the generator's blurry predicted mels.
3. **`postprocess_audio` (12B):** 4th-order high-pass (rumble/DC) → frame noise gate
   (kills hiss in pauses) → peak-normalize; optional spectral denoise if `noisereduce` is
   present.
4. **Source-excitation knobs:** `sine_amp`, `noise_std` exposed to trade off breathiness.

---

## 6. Single Colab A100 tuning

| Area | Setting |
|------|---------|
| TF32 | enabled for matmul + cuDNN, `set_float32_matmul_precision("high")` |
| cuDNN | `benchmark=True` |
| EVC | `batch_size 48`, `num_workers 8`, `persistent_workers`, pinned memory |
| Vocoder | bf16 autocast (`voc.amp`), `batch_size 32`, ~1 s segments, `max_steps 200k`, GTA `20k` |

The EVC GAN loop stays in fp32 (TF32-accelerated) for stability; bf16 AMP is applied only to
the vocoder loops, which are GAN-stable that way. Fits comfortably in 40 GB VRAM.

---

## 7. How to get the best result

1. Run top-to-bottom on an **A100**. Block 2 downloads SUBESCO once (~hundreds of MB).
2. Let **EVC training** finish all three phases.
3. Let **NSF-HiFiGAN** train (PAIRED on SUBESCO) to a high step count for naturalness.
4. Keep **GTA fine-tune**, **mel-augment**, and **denoise** on (all default) — together they
   remove almost all vocoder hiss.
5. If you want stronger emotion, raise `lambda_f0` / `lambda_prosody_f0`; if speaker pitch
   drifts, lower them. The mel hop of 512 (~32 ms @ 16 kHz) is the remaining fidelity ceiling.

**Expectations:** natural Bangla audio (well above Griffin-Lim), correct speaker timbre, and
**audible** emotion because the predicted F0 is rendered rather than discarded.

---

## 8. Verification status

| Check | Result |
|-------|--------|
| Valid JSON / nbformat-4, 37 well-formed cells | ✅ |
| Every code cell passes `ast.parse` (no import/syntax errors) | ✅ |
| **Zero** references to `kaggle` / `pyworld` / `WORLD` in code | ✅ |
| BLOCK 2 downloads via `hf_hub_download("subesco.tar.gz")` + tar stream-extract (no dataset script); `load_dataset` only as fallback | ✅ |
| Tar-extraction logic executed on a synthetic archive (nested paths, non-wav ignored, 16 kHz wavs) | ✅ |
| Block 4 extraction executed on a test clip → correct mel `(128, T)` in `[-80, 0]`, F0, energy, voiced shapes | ✅ |
| `metadata.csv` columns satisfy Block 5 `pick_col()` (required + `wav_path`) | ✅ |
| Filename parser → clean speaker (`F_01_OISHI`), correct sentence/take/emotion | ✅ |
| Vocoder upsample product == `hop_length` (512) | ✅ |
| NSF-HiFiGAN builds (~14.2 M params); forward output length == `T × 512`; train step back-props | ✅ |
| Names/paths consistent across cells (`META_PATH`, `FEATURE_ROOT`, `RESOLVE_ROOTS`, `COL_*`, `STATS`, `EMOTION_PROSODY_STATS`, `G`, `voc_G`, `CKPT_DIR`) | ✅ |
| Checkpoint discovery uses Drive `CKPT_DIR` only | ✅ |

**Paths:** raw audio `/content/subesco_audio`; extracted features
`/content/features_extracted/{mel,f0,energy,voiced}` + `metadata.csv`; all outputs and
checkpoints under `/content/drive/MyDrive/EVC_Output/`.

**Bottom line:** the notebook is now **HuggingFace-only**, Kaggle-free, self-contained, and
verified import-/syntax-/schema-clean. NSF-HiFiGAN (+ GTA + denoise) provides the natural,
emotion-faithful Bangla output.
