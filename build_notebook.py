#!/usr/bin/env python3
"""Build the corrected EVC v3 notebook for Google Colab A100 with Drive output."""
import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": source,
                  "outputs": [], "execution_count": None})


# ============ CELL 0: Title ============
md("""# Bengali EVC v3 — Emotion Voice Conversion with F0 Supervision
## Google Colab A100 Edition

**Fixes applied:**
- `pyworld` imports correctly (`import pyworld as pw`, NOT `import world`)
- All paths point to Google Drive for persistent output
- Dataset loaded from Kaggle API into Colab runtime
- WORLD vocoder uses correct `pw.synthesize()` API
- Output saved to `/content/drive/MyDrive/EVC_Output/`

**Run on:** Google Colab with A100 GPU runtime
""")

# ============ CELL 1: Colab Setup + Drive Mount ============
md("## 1 · Colab Setup — Mount Drive, Install Dependencies, Download Dataset")

code("""# ============================================================
# BLOCK 1 — Google Colab Setup: Drive + Dependencies + Dataset
# ============================================================

# --- Mount Google Drive for persistent output ---
from google.colab import drive
drive.mount('/content/drive')

# --- Install dependencies ---
import subprocess, sys
def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

_pip("librosa", "soundfile", "pyworld", "kaggle")

# --- Verify pyworld installation ---
import pyworld as pw
print(f"pyworld version: {pw.__version__ if hasattr(pw, '__version__') else 'installed OK'}")
print("pyworld functions available:", [f for f in dir(pw) if not f.startswith('_')])

# --- Setup Kaggle API for dataset download ---
# You need to upload your kaggle.json or set env vars
import os
from pathlib import Path

# Option A: Upload kaggle.json manually to /content/
# Option B: Set env vars (uncomment and fill):
# os.environ['KAGGLE_USERNAME'] = 'your_username'
# os.environ['KAGGLE_KEY'] = 'your_key'

kaggle_dir = Path.home() / '.kaggle'
kaggle_dir.mkdir(exist_ok=True)

# If kaggle.json exists in /content/, copy it
if Path('/content/kaggle.json').exists():
    import shutil
    shutil.copy('/content/kaggle.json', kaggle_dir / 'kaggle.json')
    os.chmod(str(kaggle_dir / 'kaggle.json'), 0o600)
    print("Kaggle credentials configured from /content/kaggle.json")
elif (kaggle_dir / 'kaggle.json').exists():
    print("Kaggle credentials already configured")
else:
    print("WARNING: No kaggle.json found!")
    print("  Upload kaggle.json to /content/ or set KAGGLE_USERNAME/KAGGLE_KEY env vars")
    print("  Get yours from: https://www.kaggle.com/settings -> API -> Create New Token")

print("\\nSetup complete. Drive mounted at /content/drive/")
""")


# ============ CELL 2: Download Datasets from Kaggle ============
md("## 2 · Download Datasets from Kaggle")

code("""# ============================================================
# BLOCK 2 — Download feature dataset + checkpoint from Kaggle
# ============================================================

import subprocess
from pathlib import Path

KAGGLE_DATASET_SLUG = "yousufasgormumin57/4-emo-dataset"
KAGGLE_CHECKPOINT_SLUG = "yousufasgormumin57/checkpoint-a-i-r"

DATA_DOWNLOAD_DIR = Path("/content/kaggle_data")
CKPT_DOWNLOAD_DIR = Path("/content/kaggle_checkpoint")

DATA_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Download feature dataset
if not list(DATA_DOWNLOAD_DIR.rglob("metadata.csv")):
    print(f"Downloading dataset: {KAGGLE_DATASET_SLUG}")
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET_SLUG,
                    "-p", str(DATA_DOWNLOAD_DIR), "--unzip"], check=True)
    print("Dataset downloaded!")
else:
    print("Dataset already present.")

# Download checkpoint
if not list(CKPT_DOWNLOAD_DIR.rglob("*.pt")):
    print(f"Downloading checkpoint: {KAGGLE_CHECKPOINT_SLUG}")
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_CHECKPOINT_SLUG,
                    "-p", str(CKPT_DOWNLOAD_DIR), "--unzip"], check=True)
    print("Checkpoint downloaded!")
else:
    print("Checkpoint already present.")

# Show what we got
print("\\nDataset contents:")
for f in sorted(DATA_DOWNLOAD_DIR.rglob("*"))[:20]:
    if f.is_file():
        print(f"  {f.relative_to(DATA_DOWNLOAD_DIR)}")

print("\\nCheckpoint contents:")
for f in sorted(CKPT_DOWNLOAD_DIR.rglob("*.pt")):
    print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
""")


# ============ CELL 3: Imports + Config ============
md("## 3 · Imports, Reproducibility, Configuration")

code("""# ============================================================
# BLOCK 3 — Imports + Config (Colab A100 + Drive output)
# ============================================================

import os, json, math, random, shutil, zipfile, copy, time, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import librosa.display
import librosa.sequence
import soundfile as sf
import pyworld as pw  # CORRECT import — NOT 'import world'

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from IPython.display import Audio, display

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    prop = torch.cuda.get_device_properties(0)
    print(f"  GPU: {prop.name} | {prop.total_mem / 1024**3:.1f} GB")

# ─── Paths (Google Colab + Drive) ────────────────────────────────────────────
KAGGLE_INPUT_ROOT = Path("/content/kaggle_data")
CHECKPOINT_INPUT_DIR = Path("/content/kaggle_checkpoint")

# Output goes to Drive (persists after runtime disconnects)
DRIVE_ROOT = Path("/content/drive/MyDrive/EVC_Output")
OUT_DIR    = DRIVE_ROOT / "evc_v3_output"
CKPT_DIR   = OUT_DIR / "checkpoints"
PLOT_DIR   = OUT_DIR / "plots"
AUDIO_DIR  = OUT_DIR / "audio"
CACHE_DIR  = Path("/content/dtw_cache")  # cache on local SSD (faster)
EXTRACT_DIR = Path("/content/features_extracted")

for p in [OUT_DIR, CKPT_DIR, PLOT_DIR, AUDIO_DIR, CACHE_DIR, EXTRACT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CFG = {
    "sample_rate": 16000,
    "n_fft": 2048,
    "hop_length": 512,
    "win_length": 2048,
    "n_mels": 128,
    "fmin": 0.0,
    "fmax": 8000.0,
    "top_db": 25.0,

    "trim_silence": True,
    "trim_use_voiced": True,
    "trim_top_db_margin": 25.0,
    "trim_pad_frames": 5,
    "min_frames_after_trim": 48,
    "edge_zero_window": 5,
    "edge_zero_apply": True,

    "source_emotion": "neutral",
    "target_emotions": ["angry", "happy", "sad"],
    "val_size": 0.10,
    "use_dtw_alignment": True,
    "max_dtw_frames": 420,
    "num_workers": 4,  # A100 has more CPU cores

    "content_dim": 256,
    "aux_dim": 64,
    "speaker_dim": 128,
    "emotion_dim": 64,
    "prosody_cond_dim": 32,
    "hidden_dim": 256,
    "dropout": 0.10,

    "total_epochs": 300,
    "phase1_epochs": 50,
    "phase2_epochs": 150,
    "phase3_epochs": 100,
    "batch_size": 32,  # A100 can handle larger batches
    "lr_G": 1e-4,
    "lr_D": 5e-5,
    "lr_SER": 1e-4,
    "weight_decay": 1e-5,
    "grad_clip": 5.0,

    "ser_pretrain_epochs": 15,

    "lambda_f0": 8.0,
    "lambda_energy_pred": 4.0,
    "lambda_voiced": 2.0,
    "lambda_prosody": 5.0,
    "lambda_prosody_f0": 6.0,

    "use_f0_transform": True,
    "f0_transform_weight": 0.5,

    "use_online_ser": True,
    "lr_ser_online": 2e-4,
    "online_ser_warmup": 3,
    "use_grl": True,
    "lambda_grl": 1.0,
    "grl_alpha": 1.0,

    "p2_lambda_l1": 3.0,
    "p2_lambda_content": 3.0,
    "p2_lambda_cycle": 1.0,
    "p2_lambda_ser": 4.0,
    "p2_lambda_adv": 0.3,
    "p3_lambda_l1": 2.0,
    "p3_lambda_content": 2.0,
    "p3_lambda_cycle": 0.5,
    "p3_lambda_ser": 6.0,
    "p3_lambda_adv": 0.6,

    "use_world_vocoder": True,

    "save_every": 25,
    "resume": True,
    "resume_path": None,
    "checkpoint_input_dir": str(CHECKPOINT_INPUT_DIR),
}

print(f"Output dir (Drive): {OUT_DIR}")
print(f"Batch size: {CFG['batch_size']} (optimized for A100)")
print(f"Workers: {CFG['num_workers']}")
""")


# ============ CELL 4: Dataset discovery ============
md("## 4 · Locate Feature Dataset")

code("""# ============================================================
# BLOCK 4 — Find processed features (from Kaggle download)
# ============================================================

def find_processed_dataset():
    # Search in the downloaded Kaggle data
    search_roots = [KAGGLE_INPUT_ROOT, EXTRACT_DIR]
    for root in search_roots:
        if not root.exists():
            continue
        candidates = list(root.rglob("metadata.csv"))
        for c in candidates:
            try:
                head = pd.read_csv(c, nrows=2)
                if "mel_path" in head.columns:
                    print(f"Found dataset: {c}")
                    return c, c.parent
            except Exception:
                pass

    # Try extracting a ZIP
    for root in search_roots:
        if not root.exists():
            continue
        for z in root.rglob("*.zip"):
            try:
                with zipfile.ZipFile(z) as zf:
                    names = zf.namelist()
                    if any(n.endswith("metadata.csv") for n in names):
                        print(f"Extracting: {z}")
                        zf.extractall(EXTRACT_DIR)
                        meta = list(EXTRACT_DIR.rglob("metadata.csv"))
                        if meta:
                            return meta[0], meta[0].parent
            except Exception:
                continue

    raise FileNotFoundError(
        "Could not find processed dataset (metadata.csv with mel_path column).\\n"
        "Make sure the Kaggle dataset was downloaded correctly.")

META_PATH, FEATURE_ROOT = find_processed_dataset()
RESOLVE_ROOTS = [FEATURE_ROOT, EXTRACT_DIR, KAGGLE_INPUT_ROOT]
print(f"Feature root: {FEATURE_ROOT}")
""")


# ============ CELL 5: Load metadata ============
md("## 5 · Load Metadata + Resolve Paths")

code("""# ============================================================
# BLOCK 5 — Load metadata
# ============================================================

df = pd.read_csv(META_PATH)
df.columns = [c.strip() for c in df.columns]
print(f"Rows: {len(df)}, Columns: {list(df.columns)}")

def pick_col(possible_names, required=True):
    lower_map = {c.lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    if required:
        raise KeyError(f"Missing column. Tried: {possible_names}")
    return None

COL_EMOTION = pick_col(["emotion"])
COL_LABEL   = pick_col(["label"], required=False)
COL_SPEAKER = pick_col(["speaker"])
COL_SENT    = pick_col(["sentence", "sent"])
COL_TAKE    = pick_col(["take"])
COL_MEL     = pick_col(["mel_path", "mel", "mel_file"])
COL_F0      = pick_col(["f0_path", "f0", "pitch_path"])
COL_ENERGY  = pick_col(["energy_path", "energy", "energy_file"])
COL_VOICED  = pick_col(["voiced_path", "voiced", "uv_path"])
COL_WAV     = pick_col(["wav_path", "audio_path"], required=False)

df[COL_EMOTION] = df[COL_EMOTION].astype(str).str.lower().str.strip()
df[COL_SPEAKER] = df[COL_SPEAKER].astype(str)

print("\\nEmotion counts:")
print(df[COL_EMOTION].value_counts().to_string())

def resolve_path(p):
    if pd.isna(p): return None
    p = str(p)
    cand = Path(p)
    if cand.is_absolute() and cand.exists(): return cand
    for root in RESOLVE_ROOTS:
        c = root / p
        if c.exists(): return c
    base = Path(p).name
    for root in RESOLVE_ROOTS:
        hits = list(root.rglob(base))
        if hits: return hits[0]
    return None

# Sanity check
print("\\nPath resolution check (first 10):")
for col in [COL_MEL, COL_F0, COL_ENERGY, COL_VOICED]:
    ok = sum(resolve_path(x) is not None for x in df[col].head(10))
    print(f"  {col}: {ok}/10")
""")


# ============ CELL 6: Feature loading ============
md("## 6 · Feature Loading, Trimming, Normalization")

code("""# ============================================================
# BLOCK 6 — Feature loading utilities
# ============================================================

def ensure_mel_shape(mel):
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim != 2: raise ValueError(f"Mel must be 2D, got {mel.shape}")
    if mel.shape[0] == CFG["n_mels"]: return mel
    if mel.shape[1] == CFG["n_mels"]: return mel.T
    raise ValueError(f"No {CFG['n_mels']} dimension in mel: {mel.shape}")

def load_mel_db(path_like):
    path = resolve_path(path_like)
    if path is None: raise FileNotFoundError(path_like)
    mel = np.load(path).astype(np.float32)
    mel = ensure_mel_shape(mel)
    mel = np.nan_to_num(mel, nan=-80.0, posinf=0.0, neginf=-80.0)
    return np.clip(mel, -80.0, 0.0)

def fit_length_1d(x, T):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == T: return x
    if len(x) > T: return x[:T]
    return np.concatenate([x, np.zeros(T - len(x), dtype=np.float32)])

def fit_length_2d(x, T):
    if x.shape[1] == T: return x
    if x.shape[1] > T: return x[:, :T]
    pad = np.full((x.shape[0], T - x.shape[1]), -80.0, dtype=np.float32)
    return np.concatenate([x, pad], axis=1)

def load_1d_feature(path_like, expected_len=None):
    path = resolve_path(path_like)
    if path is None: raise FileNotFoundError(path_like)
    arr = np.load(path).astype(np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if expected_len is not None: arr = fit_length_1d(arr, expected_len)
    return arr

def derive_energy_from_mel_db(mel_db):
    return mel_db.mean(axis=0).astype(np.float32)

def get_active_region(mel_db, voiced=None):
    T = mel_db.shape[1]
    if T <= 1: return 0, T
    frame_energy = derive_energy_from_mel_db(mel_db)
    threshold = float(frame_energy.max()) - CFG["trim_top_db_margin"]
    energy_active = frame_energy > threshold
    if voiced is not None and len(voiced) == T:
        combined = energy_active | (np.asarray(voiced).reshape(-1) > 0.5)
    else:
        combined = energy_active
    idx = np.where(combined)[0]
    if len(idx) < CFG["min_frames_after_trim"]: return 0, T
    pad = CFG["trim_pad_frames"]
    start = max(0, int(idx[0]) - pad)
    end = min(T, int(idx[-1]) + pad + 1)
    return (start, end) if end > start else (0, T)

def zero_silent_edges(mel_db, voiced, energy=None):
    window = CFG["edge_zero_window"]
    T = mel_db.shape[1]
    if T < window * 2 or voiced is None or len(voiced) != T: return mel_db
    if energy is None: energy = derive_energy_from_mel_db(mel_db)
    threshold = float(energy.max()) - CFG["trim_top_db_margin"]
    v = np.asarray(voiced).reshape(-1) > 0.5
    mel_out = mel_db.copy()
    for i in range(window):
        if not v[i] and energy[i] < threshold: mel_out[:, i] = -80.0
    for i in range(T - window, T):
        if not v[i] and energy[i] < threshold: mel_out[:, i] = -80.0
    return mel_out

def trim_feature_bundle(mel_db, f0_hz=None, energy=None, voiced=None):
    if not CFG["trim_silence"]:
        return mel_db, f0_hz, energy, voiced, 0, mel_db.shape[1]
    start, end = get_active_region(mel_db, voiced)
    mel_db = mel_db[:, start:end]
    if f0_hz is not None: f0_hz = f0_hz[start:end]
    if energy is not None: energy = energy[start:end]
    if voiced is not None: voiced = voiced[start:end]
    if CFG["edge_zero_apply"] and voiced is not None:
        mel_db = zero_silent_edges(mel_db, voiced, energy)
    return mel_db, f0_hz, energy, voiced, start, end

def normalize_mel(mel_db):
    return ((np.clip(mel_db, -80.0, 0.0) + 40.0) / 40.0).astype(np.float32)

def denormalize_mel(mel_norm):
    return np.clip(mel_norm * 40.0 - 40.0, -80.0, 0.0).astype(np.float32)

def load_full_features(row):
    mel_db = load_mel_db(row[COL_MEL])
    T = mel_db.shape[1]
    f0_hz = load_1d_feature(row[COL_F0], expected_len=T)
    energy = load_1d_feature(row[COL_ENERGY], expected_len=T)
    voiced = load_1d_feature(row[COL_VOICED], expected_len=T)
    voiced = (voiced > 0.5).astype(np.float32)
    mel_db, f0_hz, energy, voiced, s, e = trim_feature_bundle(
        mel_db, f0_hz, energy, voiced)
    return {"mel_db": mel_db, "f0_hz": f0_hz, "energy": energy, "voiced": voiced}

def compute_stats(df_subset):
    f0_logs, energies, kept = [], [], 0
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Stats"):
        try:
            feat = load_full_features(row)
        except Exception:
            continue
        f0, e = feat["f0_hz"], feat["energy"]
        voiced_mask = f0 > 0
        if voiced_mask.any():
            f0_logs.append(np.log(np.maximum(f0[voiced_mask], 1e-6)))
        energies.append(e)
        kept += 1
    f0_cat = np.concatenate(f0_logs) if f0_logs else np.array([np.log(150.0)])
    e_cat = np.concatenate(energies) if energies else np.array([-50.0])
    return {"f0_log_mean": float(f0_cat.mean()), "f0_log_std": float(max(f0_cat.std(), 0.01)),
            "energy_mean": float(e_cat.mean()), "energy_std": float(max(e_cat.std(), 0.01)),
            "rows_used": kept}

def normalize_f0(f0_hz, stats):
    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    voiced = f0_hz > 0
    out = np.zeros_like(f0_hz)
    if voiced.any():
        out[voiced] = (np.log(np.maximum(f0_hz[voiced], 1e-6)) - stats["f0_log_mean"]) / (stats["f0_log_std"] + 1e-8)
    return out.astype(np.float32), voiced.astype(np.float32)

def denormalize_f0(f0_norm, voiced, stats):
    f0_norm = np.asarray(f0_norm, dtype=np.float32)
    voiced = np.asarray(voiced) > 0.5
    out = np.zeros_like(f0_norm)
    out[voiced] = np.exp(f0_norm[voiced] * stats["f0_log_std"] + stats["f0_log_mean"])
    return out

def normalize_energy(energy, stats):
    return ((np.asarray(energy, dtype=np.float32) - stats["energy_mean"]) / (stats["energy_std"] + 1e-8)).astype(np.float32)

def denormalize_energy(energy_norm, stats):
    return energy_norm * stats["energy_std"] + stats["energy_mean"]

print("Feature utilities ready.")
""")


# ============ CELL 7: Build pairs + prosody stats ============
md("## 7 · Build Pairs + Per-Emotion Prosody Statistics")

code("""# ============================================================
# BLOCK 7 — Pairing + per-emotion F0/energy statistics
# ============================================================

source_emo = CFG["source_emotion"]
target_emos = set(CFG["target_emotions"])
df_work = df[df[COL_EMOTION].isin([source_emo] + list(target_emos))].copy().reset_index(drop=True)

emotion_names = sorted(df_work[COL_EMOTION].unique().tolist())
emotion_to_id = {e: i for i, e in enumerate(emotion_names)}
id_to_emotion = {i: e for e, i in emotion_to_id.items()}
speaker_names = sorted(df_work[COL_SPEAKER].unique().tolist())
speaker_to_id = {s: i for i, s in enumerate(speaker_names)}
id_to_speaker = {i: s for s, i in speaker_to_id.items()}

print("Emotions:", emotion_to_id)
print("Speakers:", len(speaker_to_id))

# Build pairs
def make_key(row, include_take=True):
    parts = [str(row[COL_SPEAKER])]
    if COL_SENT: parts.append(str(row[COL_SENT]))
    if include_take and COL_TAKE: parts.append(str(row[COL_TAKE]))
    return "||".join(parts)

neutral_df = df_work[df_work[COL_EMOTION] == source_emo]
target_df = df_work[df_work[COL_EMOTION].isin(target_emos)]

pairs = []
neutral_map = defaultdict(list)
for idx, row in neutral_df.iterrows():
    neutral_map[make_key(row)].append(idx)
for tidx, trow in target_df.iterrows():
    key = make_key(trow)
    if key in neutral_map:
        for nidx in neutral_map[key]:
            pairs.append((nidx, tidx, "strict"))

if len(pairs) < 100:
    neutral_map2 = defaultdict(list)
    for idx, row in neutral_df.iterrows():
        neutral_map2[make_key(row, include_take=False)].append(idx)
    used = set((n, t) for n, t, _ in pairs)
    for tidx, trow in target_df.iterrows():
        key = make_key(trow, include_take=False)
        if key in neutral_map2:
            for nidx in neutral_map2[key]:
                if (nidx, tidx) not in used:
                    pairs.append((nidx, tidx, "relaxed"))

pairs_df = pd.DataFrame(pairs, columns=["src_idx", "tgt_idx", "pair_type"])
pairs_df["target_emotion"] = [df_work.iloc[t][COL_EMOTION] for t in pairs_df["tgt_idx"]]
pairs_df["speaker"] = [df_work.iloc[s][COL_SPEAKER] for s in pairs_df["src_idx"]]
print(f"\\nTotal pairs: {len(pairs_df)}")
print(pairs_df["target_emotion"].value_counts().to_string())

train_pairs, val_pairs = train_test_split(
    pairs_df, test_size=CFG["val_size"], random_state=SEED,
    stratify=pairs_df["target_emotion"] if pairs_df["target_emotion"].nunique() > 1 else None)
train_pairs = train_pairs.reset_index(drop=True)
val_pairs = val_pairs.reset_index(drop=True)

train_indices = set(train_pairs["src_idx"].tolist() + train_pairs["tgt_idx"].tolist())
df_train_rows = df_work.iloc[sorted(train_indices)].reset_index(drop=True)

# Compute global stats
STATS = compute_stats(df_train_rows)
print(f"\\nGlobal stats: {STATS}")

# Per-emotion prosody statistics
EMOTION_PROSODY_STATS = {}
for emo in emotion_names:
    emo_df = df_work[df_work[COL_EMOTION] == emo].sample(min(200, len(df_work[df_work[COL_EMOTION] == emo])), random_state=SEED)
    f0_all, e_all = [], []
    for _, row in emo_df.iterrows():
        try:
            feat = load_full_features(row)
            v = feat["f0_hz"] > 0
            if v.any(): f0_all.append(np.log(np.maximum(feat["f0_hz"][v], 1e-6)))
            e_all.append(feat["energy"])
        except: continue
    if f0_all:
        f0c = np.concatenate(f0_all); ec = np.concatenate(e_all)
        EMOTION_PROSODY_STATS[emo] = {
            "f0_log_mean": float(f0c.mean()), "f0_log_std": float(max(f0c.std(), 0.01)),
            "energy_mean": float(ec.mean()), "energy_std": float(max(ec.std(), 0.01)),
            "f0_hz_mean": float(np.exp(f0c.mean()))}
    else:
        EMOTION_PROSODY_STATS[emo] = {"f0_log_mean": STATS["f0_log_mean"],
            "f0_log_std": STATS["f0_log_std"], "energy_mean": STATS["energy_mean"],
            "energy_std": STATS["energy_std"], "f0_hz_mean": 180.0}
    print(f"  {emo:8s}: F0={EMOTION_PROSODY_STATS[emo]['f0_hz_mean']:.0f} Hz")
""")


# ============ CELL 8: DTW + Dataset ============
md("## 8 · DTW Alignment + Dataset (with F0 Transform)")

code("""# ============================================================
# BLOCK 8 — DTW + EVCPairedDataset (v3: F0 transform + prosody targets)
# ============================================================

def align_1d_by_path(src_len, tgt_1d, wp):
    buckets = [[] for _ in range(src_len)]
    for si, ti in wp:
        if 0 <= si < src_len and 0 <= ti < len(tgt_1d):
            buckets[si].append(tgt_1d[ti])
    out = np.zeros(src_len, dtype=np.float32)
    for i, vals in enumerate(buckets):
        out[i] = float(np.mean(vals)) if vals else tgt_1d[min(max(i, 0), len(tgt_1d)-1)]
    return out

def align_mel_by_dtw(src_mel, tgt_mel, tgt_f0, tgt_e, tgt_v, cache_key=None):
    sT, tT = src_mel.shape[1], tgt_mel.shape[1]
    fallback = lambda: (fit_length_2d(tgt_mel, sT), fit_length_1d(tgt_f0, sT),
                        fit_length_1d(tgt_e, sT), fit_length_1d(tgt_v, sT))
    if sT <= 1 or tT <= 1 or not CFG["use_dtw_alignment"]: return fallback()
    if max(sT, tT) > CFG["max_dtw_frames"]: return fallback()

    if cache_key:
        cp = CACHE_DIR / f"v3_{cache_key}.npz"
        if cp.exists():
            z = np.load(cp)
            if z["mel"].shape[1] == sT: return z["mel"], z["f0"], z["energy"], z["voiced"]

    try:
        X, Y = normalize_mel(src_mel), normalize_mel(tgt_mel)
        _, wp = librosa.sequence.dtw(X=X, Y=Y, metric="cosine")
        wp = wp[::-1]
        aligned_mel = np.zeros((CFG["n_mels"], sT), dtype=np.float32)
        for si in range(sT):
            tis = [int(ti) for s, ti in wp if int(s) == si]
            aligned_mel[:, si] = tgt_mel[:, tis].mean(axis=1) if tis else tgt_mel[:, min(si, tT-1)]
        af0 = align_1d_by_path(sT, tgt_f0, wp)
        ae = align_1d_by_path(sT, tgt_e, wp)
        av = (align_1d_by_path(sT, tgt_v, wp) > 0.5).astype(np.float32)
    except Exception:
        return fallback()

    if cache_key:
        np.savez_compressed(CACHE_DIR / f"v3_{cache_key}.npz",
                            mel=aligned_mel, f0=af0, energy=ae, voiced=av)
    return aligned_mel, af0, ae, av

def log_f0_transform(src_f0_hz, src_emo, tgt_emo):
    weight = CFG.get("f0_transform_weight", 0.5)
    ss = EMOTION_PROSODY_STATS.get(src_emo)
    ts = EMOTION_PROSODY_STATS.get(tgt_emo)
    if ss is None or ts is None: return src_f0_hz.copy()
    f0 = src_f0_hz.copy()
    voiced = f0 > 0
    if not voiced.any(): return f0
    log_f0 = np.log(np.maximum(f0[voiced], 1e-6))
    transformed = (log_f0 - ss["f0_log_mean"]) / (ss["f0_log_std"] + 1e-8) * ts["f0_log_std"] + ts["f0_log_mean"]
    f0[voiced] = np.exp(log_f0 * (1 - weight) + transformed * weight)
    f0[voiced] = np.clip(f0[voiced], 50.0, 600.0)
    return f0

class EVCPairedDataset(Dataset):
    def __init__(self, pairs_df, train=True):
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.train = train
    def __len__(self): return len(self.pairs_df)
    def __getitem__(self, idx):
        item = self.pairs_df.iloc[idx]
        src_row = df_work.iloc[int(item["src_idx"])]
        tgt_row = df_work.iloc[int(item["tgt_idx"])]
        src = load_full_features(src_row); tgt = load_full_features(tgt_row)
        ck = f"pair_{int(item['src_idx'])}_{int(item['tgt_idx'])}"
        tgt_mel_a, tgt_f0_a, tgt_e_a, tgt_v_a = align_mel_by_dtw(
            src["mel_db"], tgt["mel_db"], tgt["f0_hz"], tgt["energy"], tgt["voiced"], cache_key=ck)
        T = src["mel_db"].shape[1]
        src_mel_n = normalize_mel(src["mel_db"]); tgt_mel_n = normalize_mel(tgt_mel_a)
        src_f0_n, src_v = normalize_f0(src["f0_hz"], STATS)
        tgt_f0_n, tgt_v = normalize_f0(tgt_f0_a, STATS)
        src_e_n = normalize_energy(src["energy"], STATS)
        tgt_e_n = normalize_energy(tgt_e_a, STATS)
        src_aux = np.stack([src_f0_n, src_e_n, src_v], axis=0)
        tgt_aux = np.stack([tgt_f0_n, tgt_e_n, tgt_v], axis=0)
        transformed_f0_hz = log_f0_transform(src["f0_hz"], src_row[COL_EMOTION], tgt_row[COL_EMOTION])
        transformed_f0_n, _ = normalize_f0(transformed_f0_hz, STATS)
        tp = EMOTION_PROSODY_STATS.get(tgt_row[COL_EMOTION], EMOTION_PROSODY_STATS["neutral"])
        prosody_cond = np.array([
            (tp["f0_log_mean"] - STATS["f0_log_mean"]) / (STATS["f0_log_std"] + 1e-8),
            tp["f0_log_std"] / (STATS["f0_log_std"] + 1e-8),
            (tp["energy_mean"] - STATS["energy_mean"]) / (STATS["energy_std"] + 1e-8),
            tp["energy_std"] / (STATS["energy_std"] + 1e-8)], dtype=np.float32)
        return {
            "src_mel": torch.from_numpy(src_mel_n), "tgt_mel": torch.from_numpy(tgt_mel_n),
            "src_aux": torch.from_numpy(src_aux), "tgt_aux": torch.from_numpy(tgt_aux),
            "tgt_f0_norm": torch.from_numpy(tgt_f0_n),
            "tgt_energy_norm": torch.from_numpy(tgt_e_n),
            "tgt_voiced": torch.from_numpy(tgt_v),
            "transformed_f0": torch.from_numpy(transformed_f0_n),
            "prosody_cond": torch.from_numpy(prosody_cond),
            "mask": torch.ones(T, dtype=torch.float32),
            "src_emo": torch.tensor(emotion_to_id[src_row[COL_EMOTION]], dtype=torch.long),
            "tgt_emo": torch.tensor(emotion_to_id[tgt_row[COL_EMOTION]], dtype=torch.long),
            "spk_id": torch.tensor(speaker_to_id[src_row[COL_SPEAKER]], dtype=torch.long),
            "pair_index": torch.tensor(idx, dtype=torch.long)}

def collate_fn(batch):
    max_T = max(b["src_mel"].shape[-1] for b in batch)
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if vals[0].dim() == 0: out[key] = torch.stack(vals)
        elif key == "prosody_cond": out[key] = torch.stack(vals)
        elif vals[0].dim() == 1:
            padded = torch.zeros(len(vals), max_T)
            for i, v in enumerate(vals): padded[i, :v.shape[0]] = v
            out[key] = padded
        elif vals[0].dim() == 2:
            C = vals[0].shape[0]
            fill = -1.0 if key in ("src_mel", "tgt_mel") else 0.0
            padded = torch.full((len(vals), C, max_T), fill)
            for i, v in enumerate(vals): padded[i, :, :v.shape[-1]] = v
            out[key] = padded
    return out

train_ds = EVCPairedDataset(train_pairs); val_ds = EVCPairedDataset(val_pairs, train=False)
train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
    num_workers=CFG["num_workers"], collate_fn=collate_fn, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
    num_workers=CFG["num_workers"], collate_fn=collate_fn, pin_memory=True)

# SER datasets
class SERDataset(Dataset):
    def __init__(self, df_sub):
        self.data = df_sub.reset_index(drop=True)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        feat = load_full_features(row)
        return torch.from_numpy(normalize_mel(feat["mel_db"])), torch.tensor(emotion_to_id[row[COL_EMOTION]], dtype=torch.long), torch.tensor(idx)

def ser_collate(batch):
    mels, emos, idxs = zip(*batch)
    max_T = max(m.shape[-1] for m in mels)
    padded = torch.full((len(mels), CFG["n_mels"], max_T), -1.0)
    for i, m in enumerate(mels): padded[i, :, :m.shape[-1]] = m
    return padded, torch.stack(emos), torch.stack(idxs)

ser_train_df = df_train_rows[df_train_rows[COL_EMOTION].isin(emotion_names)].reset_index(drop=True)
ser_val_df = df_work[~df_work.index.isin(train_indices)][lambda x: x[COL_EMOTION].isin(emotion_names)].reset_index(drop=True)
ser_train_loader = DataLoader(SERDataset(ser_train_df), batch_size=32, shuffle=True, num_workers=2, collate_fn=ser_collate, drop_last=True)
ser_val_loader = DataLoader(SERDataset(ser_val_df), batch_size=32, shuffle=False, num_workers=2, collate_fn=ser_collate)

print(f"Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs")
print(f"SER train: {len(ser_train_df)}, SER val: {len(ser_val_df)}")
""")


# ============ CELL 9: Model Definitions (same as before, abbreviated ref) ============
md("## 9 · Model Definitions (v3 with ProsodyHead)")

# The model code is long - I'll include it inline
code("""# ============================================================
# BLOCK 9 — Models: Generator w/ ProsodyHead, Discriminator, SER
# ============================================================

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity())
    def forward(self, x): return self.net(x)

class ContentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["dropout"]
        self.net = nn.Sequential(
            ConvBlock1D(CFG["n_mels"], 128, k=7, dropout=d),
            ConvBlock1D(128, 192, k=5, dropout=d),
            ConvBlock1D(192, CFG["content_dim"], k=5, dropout=d),
            ConvBlock1D(CFG["content_dim"], CFG["content_dim"], k=3, dropout=d))
    def forward(self, mel): return self.net(mel)

class AuxEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["dropout"]
        self.net = nn.Sequential(
            ConvBlock1D(4, 32, k=5, dropout=d),  # 4 channels: f0, energy, voiced, transformed_f0
            ConvBlock1D(32, CFG["aux_dim"], k=5, dropout=d))
    def forward(self, aux): return self.net(aux)

class ProsodyHead(nn.Module):
    def __init__(self):
        super().__init__()
        h = CFG["hidden_dim"]
        self.f0_head = nn.Sequential(ConvBlock1D(h, 128, k=5), ConvBlock1D(128, 64, k=3), nn.Conv1d(64, 1, 1))
        self.energy_head = nn.Sequential(ConvBlock1D(h, 64, k=5), nn.Conv1d(64, 1, 1))
        self.voiced_head = nn.Sequential(ConvBlock1D(h, 64, k=3), nn.Conv1d(64, 1, 1), nn.Sigmoid())
    def forward(self, hidden):
        return self.f0_head(hidden).squeeze(1), self.energy_head(hidden).squeeze(1), self.voiced_head(hidden).squeeze(1)

class Decoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        h = CFG["hidden_dim"]; d = CFG["dropout"]
        self.layers = nn.Sequential(
            ConvBlock1D(in_dim, h, k=5, dropout=d),
            ConvBlock1D(h, h, k=5, dropout=d),
            ConvBlock1D(h, h, k=3, dropout=d))
        self.mel_out = nn.Sequential(nn.Conv1d(h, CFG["n_mels"], 1), nn.Tanh())
    def forward(self, x):
        hidden = self.layers(x)
        return self.mel_out(hidden), hidden

class EVCGenerator(nn.Module):
    def __init__(self, n_speakers, n_emotions):
        super().__init__()
        self.content_encoder = ContentEncoder()
        self.aux_encoder = AuxEncoder()
        self.spk_emb = nn.Embedding(n_speakers, CFG["speaker_dim"])
        self.emo_emb = nn.Embedding(n_emotions, CFG["emotion_dim"])
        self.prosody_proj = nn.Linear(4, CFG["prosody_cond_dim"])
        in_dim = CFG["content_dim"] + CFG["aux_dim"] + CFG["speaker_dim"] + CFG["emotion_dim"] + CFG["prosody_cond_dim"]
        self.decoder = Decoder(in_dim)
        self.prosody_head = ProsodyHead()

    def forward(self, src_mel, src_aux_4ch, spk_id, tgt_emo, prosody_cond=None,
                return_content=False, return_prosody=True):
        B, _, T = src_mel.shape
        content = self.content_encoder(src_mel)
        aux = self.aux_encoder(src_aux_4ch)
        spk = self.spk_emb(spk_id).unsqueeze(-1).expand(-1, -1, T)
        emo = self.emo_emb(tgt_emo).unsqueeze(-1).expand(-1, -1, T)
        if prosody_cond is not None:
            pc = self.prosody_proj(prosody_cond).unsqueeze(-1).expand(-1, -1, T)
        else:
            pc = torch.zeros(B, CFG["prosody_cond_dim"], T, device=src_mel.device)
        x = torch.cat([content, aux, spk, emo, pc], dim=1)
        mel_out, hidden = self.decoder(x)
        if return_prosody:
            f0_p, e_p, v_p = self.prosody_head(hidden)
            if return_content: return mel_out, f0_p, e_p, v_p, content
            return mel_out, f0_p, e_p, v_p
        if return_content: return mel_out, content
        return mel_out

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return g.neg() * ctx.alpha, None

class EmotionFromContent(nn.Module):
    def __init__(self, n_emotions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CFG["content_dim"], 128), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(128, n_emotions))
    def forward(self, content_feat, alpha=1.0):
        pooled = content_feat.mean(dim=2)
        return self.net(_GradReverse.apply(pooled, alpha))

class MelDiscriminator(nn.Module):
    def __init__(self, n_emotions):
        super().__init__()
        self.emo_emb = nn.Embedding(n_emotions, 16)
        self.net = nn.Sequential(
            nn.Conv1d(CFG["n_mels"]+16, 128, 5, padding=2), nn.LeakyReLU(0.2, True),
            nn.Conv1d(128, 128, 5, padding=2), nn.LeakyReLU(0.2, True),
            nn.Conv1d(128, 64, 5, padding=2), nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 1, 1))
    def forward(self, mel, emo_id):
        B, _, T = mel.shape
        emo = self.emo_emb(emo_id).unsqueeze(-1).expand(-1, -1, T)
        return self.net(torch.cat([mel, emo], dim=1)).mean(dim=[1, 2])

class SERClassifier(nn.Module):
    def __init__(self, n_emotions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(CFG["n_mels"], 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(96, 160, 5, padding=2), nn.BatchNorm1d(160), nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(160, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(True), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(256, n_emotions)
    def forward(self, mel): return self.fc(self.net(mel).squeeze(-1))

# Instantiate
n_spk = len(speaker_to_id); n_emo = len(emotion_to_id)
G = EVCGenerator(n_spk, n_emo).to(DEVICE)
D = MelDiscriminator(n_emo).to(DEVICE)
SER = SERClassifier(n_emo).to(DEVICE)
EMO_FROM_CONTENT = EmotionFromContent(n_emo).to(DEVICE)
SER_ONLINE = SERClassifier(n_emo).to(DEVICE)

ct = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"G: {ct(G):,}  D: {ct(D):,}  SER: {ct(SER):,}")
""")


# ============ CELL 10: Checkpoint + SER + Phase + Training ============
md("## 10 · Checkpoint, SER, Phase Schedule, Training Loop")

code("""# ============================================================
# BLOCK 10 — Checkpoint utilities + SER + Phase + Training
# ============================================================
import re as _re

def _epoch_from_name(p):
    m = _re.search(r"epoch[_-]?(\\d+)", p.stem, _re.IGNORECASE)
    return int(m.group(1)) if m else -1

def find_latest_checkpoint():
    candidates = []
    for d in [CHECKPOINT_INPUT_DIR, CKPT_DIR]:
        if d.exists():
            for c in sorted(d.rglob("*.pt")):
                if c.name.startswith("ser_"): continue
                candidates.append(c)
    scored = []
    for c in candidates:
        try:
            head = torch.load(c, map_location="cpu", weights_only=False)
            if not isinstance(head, dict) or "G" not in head: continue
            scored.append((max(_epoch_from_name(c), head.get("epoch", -1)), c))
        except: continue
    return sorted(scored)[-1][1] if scored else None

def save_checkpoint(epoch, tag="latest"):
    ckpt = {"epoch": epoch, "G": G.state_dict(), "D": D.state_dict(), "SER": SER.state_dict(),
            "EMO_FROM_CONTENT": EMO_FROM_CONTENT.state_dict(), "SER_ONLINE": SER_ONLINE.state_dict(),
            "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
            "history": history, "emotion_to_id": emotion_to_id, "speaker_to_id": speaker_to_id,
            "stats": STATS, "emotion_prosody_stats": EMOTION_PROSODY_STATS, "cfg": CFG}
    torch.save(ckpt, CKPT_DIR / f"evc_v3_{tag}.pt")

def load_checkpoint(path):
    global history
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if "G" not in ckpt: return 0
    G.load_state_dict(ckpt["G"], strict=False)
    if "D" in ckpt: D.load_state_dict(ckpt["D"], strict=False)
    if "SER" in ckpt: SER.load_state_dict(ckpt["SER"], strict=False)
    if ckpt.get("EMO_FROM_CONTENT"):
        try: EMO_FROM_CONTENT.load_state_dict(ckpt["EMO_FROM_CONTENT"])
        except: pass
    if ckpt.get("SER_ONLINE"):
        try: SER_ONLINE.load_state_dict(ckpt["SER_ONLINE"])
        except: pass
    if isinstance(ckpt.get("history"), list): history = ckpt["history"]
    return ckpt.get("epoch", 0)

# --- SER pretrain/load ---
ser_loaded = False
latest_ckpt = find_latest_checkpoint()
if latest_ckpt:
    try:
        ck = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)
        if "SER" in ck: SER.load_state_dict(ck["SER"]); ser_loaded = True; print(f"SER loaded from {latest_ckpt.name}")
    except: pass

if not ser_loaded:
    print("Pretraining SER...")
    opt_ser = torch.optim.AdamW(SER.parameters(), lr=CFG["lr_SER"])
    best_acc = 0
    for ep in range(1, CFG["ser_pretrain_epochs"]+1):
        SER.train()
        for mel, y, _ in ser_train_loader:
            mel, y = mel.to(DEVICE), y.to(DEVICE)
            opt_ser.zero_grad(); loss = F.cross_entropy(SER(mel), y); loss.backward()
            nn.utils.clip_grad_norm_(SER.parameters(), 5.0); opt_ser.step()
        SER.eval(); correct, total = 0, 0
        with torch.no_grad():
            for mel, y, _ in ser_val_loader:
                mel, y = mel.to(DEVICE), y.to(DEVICE)
                correct += (SER(mel).argmax(1) == y).sum().item(); total += y.numel()
        acc = correct / max(1, total)
        if acc > best_acc: best_acc = acc; torch.save(SER.state_dict(), CKPT_DIR / "ser_best.pt")
        if ep % 5 == 0: print(f"  SER ep {ep}: acc={acc:.3f}")
    print(f"SER best: {best_acc:.3f}")

SER.eval()
for p in SER.parameters(): p.requires_grad_(False)
if CFG["use_online_ser"]:
    SER_ONLINE.load_state_dict(SER.state_dict())
    for p in SER_ONLINE.parameters(): p.requires_grad_(True)
print("SER frozen. Online SER ready.")

# --- Optimizers ---
opt_G = torch.optim.AdamW(G.parameters(), lr=CFG["lr_G"], betas=(0.5, 0.9), weight_decay=CFG["weight_decay"])
opt_D = torch.optim.AdamW(D.parameters(), lr=CFG["lr_D"], betas=(0.5, 0.9), weight_decay=CFG["weight_decay"])
opt_grl = torch.optim.AdamW(EMO_FROM_CONTENT.parameters(), lr=CFG["lr_G"], betas=(0.5, 0.9))
opt_ser_online = torch.optim.AdamW(SER_ONLINE.parameters(), lr=CFG["lr_ser_online"], betas=(0.5, 0.9))
history = []
content_teacher = None

# --- Resume ---
start_epoch = 1
if CFG["resume"] and latest_ckpt:
    ep = load_checkpoint(latest_ckpt)
    if ep > 0:
        start_epoch = ep + 1
        print(f"Resumed from epoch {ep}, next = {start_epoch}")
        print("  (ProsodyHead layers initialized fresh)")

if start_epoch > CFG["phase1_epochs"] + 1:
    content_teacher = copy.deepcopy(G.content_encoder).to(DEVICE)
    content_teacher.eval()
    for p in content_teacher.parameters(): p.requires_grad_(False)
    print("Content teacher restored.")

print(f"Starting from epoch {start_epoch}")
""")


# ============ CELL 11: Phase schedule + losses + main training loop ============
md("## 11 · Phase Schedule + Training Loop")

code("""# ============================================================
# BLOCK 11 — Phase schedule, losses, main training loop
# ============================================================

def lerp(a, b, t): return a + (b - a) * float(np.clip(t, 0.0, 1.0))

def get_phase(epoch):
    if epoch <= CFG["phase1_epochs"]:
        return {"name": "P1-recon+F0warmup", "mode": "reconstruct",
                "l_l1": 20.0, "l_content": 0, "l_cycle": 0, "l_energy": 2.0, "l_ser": 0, "l_adv": 0,
                "l_f0": 4.0, "l_epred": 2.0, "l_voiced": 1.0, "l_prosody": 0, "l_pf0": 0, "l_grl": 0,
                "lr_G_s": 1.0, "lr_D_s": 0, "noise": 0}
    if epoch <= CFG["phase1_epochs"] + CFG["phase2_epochs"]:
        t = (epoch - CFG["phase1_epochs"]) / CFG["phase2_epochs"]
        return {"name": "P2-emotion+F0", "mode": "convert",
                "l_l1": lerp(CFG["p2_lambda_l1"], CFG["p2_lambda_l1"]*0.8, t),
                "l_content": CFG["p2_lambda_content"], "l_cycle": CFG["p2_lambda_cycle"], "l_energy": 2.0,
                "l_ser": lerp(CFG["p2_lambda_ser"]*0.5, CFG["p2_lambda_ser"], t),
                "l_adv": lerp(CFG["p2_lambda_adv"]*0.5, CFG["p2_lambda_adv"], t),
                "l_f0": CFG["lambda_f0"], "l_epred": CFG["lambda_energy_pred"],
                "l_voiced": CFG["lambda_voiced"], "l_prosody": CFG["lambda_prosody"],
                "l_pf0": CFG["lambda_prosody_f0"],
                "l_grl": CFG["lambda_grl"] if CFG["use_grl"] else 0,
                "lr_G_s": 1.0, "lr_D_s": 1.0, "noise": lerp(0.03, 0.01, t)}
    t = (epoch - CFG["phase1_epochs"] - CFG["phase2_epochs"]) / CFG["phase3_epochs"]
    return {"name": "P3-sharpen", "mode": "convert",
            "l_l1": lerp(CFG["p3_lambda_l1"], CFG["p3_lambda_l1"]*0.7, t),
            "l_content": CFG["p3_lambda_content"], "l_cycle": CFG["p3_lambda_cycle"], "l_energy": 2.0,
            "l_ser": lerp(CFG["p3_lambda_ser"]*0.8, CFG["p3_lambda_ser"], t),
            "l_adv": lerp(CFG["p3_lambda_adv"]*0.7, CFG["p3_lambda_adv"], t),
            "l_f0": CFG["lambda_f0"]*1.2, "l_epred": CFG["lambda_energy_pred"],
            "l_voiced": CFG["lambda_voiced"], "l_prosody": CFG["lambda_prosody"]*1.2,
            "l_pf0": CFG["lambda_prosody_f0"]*1.2,
            "l_grl": CFG["lambda_grl"] if CFG["use_grl"] else 0,
            "lr_G_s": lerp(1.0, 0.5, t), "lr_D_s": 1.0, "noise": lerp(0.01, 0, t)}

def set_lr(opt, base, scale):
    for pg in opt.param_groups: pg["lr"] = base * scale

def gate(mel, mask): return mel * mask[:, None, :]

def masked_l1(pred, tgt, mask):
    m = mask[:, None, :]; loss = torch.abs(pred - tgt) * m
    return loss.sum() / (m.sum() * pred.shape[1] + 1e-8)

def masked_l1_1d(pred, tgt, mask):
    return (torch.abs(pred - tgt) * mask).sum() / (mask.sum() + 1e-8)

def f0_sup_loss(f0_pred, tgt_f0, tgt_voiced, mask):
    vm = (tgt_voiced > 0.5).float() * mask
    if vm.sum() < 1: return torch.tensor(0.0, device=f0_pred.device)
    return (torch.abs(f0_pred - tgt_f0) * vm).sum() / (vm.sum() + 1e-8)

def voiced_bce(pred, tgt, mask):
    return F.binary_cross_entropy(pred * mask, tgt * mask, reduction='sum') / (mask.sum() + 1e-8)

def _mm(x, m): return (x * m).sum(1) / (m.sum(1) + 1e-8)
def _ms(x, m):
    mu = _mm(x, m).unsqueeze(1)
    return torch.sqrt((((x - mu)**2) * m).sum(1) / (m.sum(1) + 1e-8) + 1e-8)

def prosody_energy_loss(fake, tgt_aux, mask):
    ge = fake.mean(dim=1); te = tgt_aux[:, 1, :]
    return F.l1_loss(_mm(ge, mask), _mm(te, mask)) + F.l1_loss(_ms(ge, mask), _ms(te, mask))

def f0_stats_loss(f0_pred, tgt_f0, tgt_voiced, mask):
    vm = (tgt_voiced > 0.5).float() * mask
    if vm.sum() < 10: return torch.tensor(0.0, device=f0_pred.device)
    return F.l1_loss(_mm(f0_pred, vm), _mm(tgt_f0, vm)) + F.l1_loss(_ms(f0_pred, vm), _ms(tgt_f0, vm))

# ─── MAIN TRAINING LOOP ─────────────────────────────────────────────────────
for epoch in range(start_epoch, CFG["total_epochs"] + 1):
    ph = get_phase(epoch)
    set_lr(opt_G, CFG["lr_G"], ph["lr_G_s"]); set_lr(opt_D, CFG["lr_D"], ph["lr_D_s"])
    if epoch == CFG["phase1_epochs"] + 1 and content_teacher is None:
        content_teacher = copy.deepcopy(G.content_encoder).to(DEVICE).eval()
        for p in content_teacher.parameters(): p.requires_grad_(False)

    G.train(); D.train(); totals = defaultdict(float); seen = 0; tic = time.time()
    for batch in tqdm(train_loader, desc=f"Ep{epoch:03d} {ph['name']}", leave=False):
        sm = batch["src_mel"].to(DEVICE); tm = batch["tgt_mel"].to(DEVICE)
        mask = batch["mask"].to(DEVICE); se = batch["src_emo"].to(DEVICE); te = batch["tgt_emo"].to(DEVICE)
        spk = batch["spk_id"].to(DEVICE); pc = batch["prosody_cond"].to(DEVICE)
        sa3 = batch["src_aux"].to(DEVICE); tf0 = batch["transformed_f0"].to(DEVICE)
        sa4 = torch.cat([sa3, tf0.unsqueeze(1)], dim=1)
        tgt_f0n = batch["tgt_f0_norm"].to(DEVICE); tgt_en = batch["tgt_energy_norm"].to(DEVICE)
        tgt_v = batch["tgt_voiced"].to(DEVICE); tgt_aux = batch["tgt_aux"].to(DEVICE)

        dm, de = (sm, se) if ph["mode"] == "reconstruct" else (tm, te)
        df0 = sa3[:, 0, :] if ph["mode"] == "reconstruct" else tgt_f0n
        den = sa3[:, 1, :] if ph["mode"] == "reconstruct" else tgt_en
        dv = sa3[:, 2, :] if ph["mode"] == "reconstruct" else tgt_v

        # Discriminator
        lD = torch.tensor(0.0, device=DEVICE)
        if ph["l_adv"] > 0:
            with torch.no_grad(): fd, _, _, _ = G(sm, sa4, spk, de, pc); fd = gate(fd, mask)
            noise = ph["noise"]
            pr = D(dm + torch.randn_like(dm)*noise, de); pf = D(fd + torch.randn_like(fd)*noise, de)
            lD = 0.5*(F.mse_loss(pr, torch.ones_like(pr)) + F.mse_loss(pf, torch.zeros_like(pf)))
            opt_D.zero_grad(set_to_none=True); lD.backward()
            nn.utils.clip_grad_norm_(D.parameters(), CFG["grad_clip"]); opt_D.step()

        # Online SER
        if CFG["use_online_ser"] and ph["mode"] == "convert":
            with torch.no_grad(): fs, _, _, _ = G(sm, sa4, spk, de, pc); fs = gate(fs, mask)
            SER_ONLINE.train()
            lo = F.cross_entropy(SER_ONLINE(dm), de) + F.cross_entropy(SER_ONLINE(fs), de)
            opt_ser_online.zero_grad(set_to_none=True); lo.backward()
            nn.utils.clip_grad_norm_(SER_ONLINE.parameters(), CFG["grad_clip"]); opt_ser_online.step()

        # Generator
        use_grl = CFG["use_grl"] and ph["l_grl"] > 0
        if use_grl:
            fake, f0p, ep, vp, cf = G(sm, sa4, spk, de, pc, return_content=True, return_prosody=True)
        else:
            fake, f0p, ep, vp = G(sm, sa4, spk, de, pc, return_prosody=True)
        fake = gate(fake, mask)

        ll1 = masked_l1(fake, dm, mask)
        le = masked_l1_1d(fake.mean(1), dm.mean(1), mask)
        lf0 = f0_sup_loss(f0p, df0, dv, mask)
        lep = masked_l1_1d(ep, den, mask)
        lv = voiced_bce(vp, dv, mask)

        lc = torch.tensor(0.0, device=DEVICE)
        if content_teacher and ph["l_content"] > 0:
            lc = masked_l1(content_teacher(fake), content_teacher(sm).detach(), mask)
        lcy = torch.tensor(0.0, device=DEVICE)
        if ph["l_cycle"] > 0:
            cyc, _, _, _ = G(fake, sa4, spk, se, pc, return_prosody=True); lcy = masked_l1(gate(cyc, mask), sm, mask)
        ls = torch.tensor(0.0, device=DEVICE)
        if ph["l_ser"] > 0:
            ls = F.cross_entropy(SER(fake), de)
            if CFG["use_online_ser"]: SER_ONLINE.eval(); ls = 0.5*ls + 0.5*F.cross_entropy(SER_ONLINE(fake), de)
        lpro = prosody_energy_loss(fake, tgt_aux, mask) if ph["l_prosody"] > 0 else torch.tensor(0.0, device=DEVICE)
        lpf0 = f0_stats_loss(f0p, df0, dv, mask) if ph["l_pf0"] > 0 else torch.tensor(0.0, device=DEVICE)
        lgrl = F.cross_entropy(EMO_FROM_CONTENT(cf, CFG["grl_alpha"]), se) if use_grl else torch.tensor(0.0, device=DEVICE)
        ladv = F.mse_loss(D(fake, de), torch.ones(fake.size(0), device=DEVICE)) if ph["l_adv"] > 0 else torch.tensor(0.0, device=DEVICE)

        lG = (ph["l_l1"]*ll1 + ph["l_energy"]*le + ph["l_f0"]*lf0 + ph["l_epred"]*lep + ph["l_voiced"]*lv
              + ph["l_content"]*lc + ph["l_cycle"]*lcy + ph["l_ser"]*ls + ph["l_adv"]*ladv
              + ph["l_prosody"]*lpro + ph["l_pf0"]*lpf0 + ph["l_grl"]*lgrl)

        opt_G.zero_grad(set_to_none=True)
        if use_grl: opt_grl.zero_grad(set_to_none=True)
        lG.backward(); nn.utils.clip_grad_norm_(G.parameters(), CFG["grad_clip"]); opt_G.step()
        if use_grl: nn.utils.clip_grad_norm_(EMO_FROM_CONTENT.parameters(), CFG["grad_clip"]); opt_grl.step()

        bs = sm.size(0); seen += bs
        for k, v in {"total": lG, "l1": ll1, "f0": lf0, "ser": ls, "adv": ladv}.items():
            totals[k] += v.item() * bs

    metrics = {k: v/max(1,seen) for k,v in totals.items()}
    elapsed = time.time() - tic
    history.append({"epoch": epoch, "phase": ph["name"], **{f"train_{k}": round(v,5) for k,v in metrics.items()}, "time": round(elapsed,1)})
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)

    if epoch % CFG["save_every"] == 0 or epoch == CFG["total_epochs"]:
        save_checkpoint(epoch, tag=f"epoch_{epoch:03d}")
    save_checkpoint(epoch, tag="latest")
    print(f"Ep{epoch:03d} | total={metrics.get('total',0):.3f} f0={metrics.get('f0',0):.4f} l1={metrics.get('l1',0):.3f} ser={metrics.get('ser',0):.3f} | {elapsed:.0f}s")

print("\\nTraining complete!")
""")


# ============ CELL 12: Inference with CORRECT pyworld API ============
md("""## 12 · Inference with WORLD Vocoder (pyworld)

**IMPORTANT:** pyworld is imported as `import pyworld as pw`, NOT `import world`.

The correct API:
- `pw.harvest(wav, sr)` → extract F0
- `pw.cheaptrick(wav, f0, t, sr)` → spectral envelope
- `pw.d4c(wav, f0, t, sr)` → aperiodicity
- `pw.synthesize(f0, sp, ap, sr)` → synthesize waveform
""")

code("""# ============================================================
# BLOCK 12 — Inference + WORLD vocoder (CORRECT pyworld API)
# ============================================================
import pyworld as pw  # CORRECT: 'pyworld' package imports as 'pw'

def mel_to_audio_griffinlim(mel_db):
    '''Fallback vocoder.'''
    mel_power = librosa.db_to_power(mel_db, ref=1.0)
    wav = librosa.feature.inverse.mel_to_audio(
        M=mel_power, sr=CFG["sample_rate"], n_fft=CFG["n_fft"],
        hop_length=CFG["hop_length"], win_length=CFG["win_length"],
        fmin=CFG["fmin"], fmax=CFG["fmax"], power=2.0, n_iter=64)
    wav = wav.astype(np.float32)
    if np.abs(wav).max() > 0: wav = wav / (np.abs(wav).max() + 1e-8) * 0.95
    return wav

def world_synthesize(mel_db, f0_hz, voiced_mask):
    '''
    Synthesize audio using pyworld (WORLD vocoder) with predicted F0.

    This is the correct way to use pyworld:
    - pw.synthesize(f0, spectral_envelope, aperiodicity, sample_rate, frame_period)
    - f0 must be float64, shape (n_frames,)
    - sp must be float64, shape (n_frames, fft_size//2 + 1)
    - ap must be float64, shape (n_frames, fft_size//2 + 1)
    '''
    try:
        sr = CFG["sample_rate"]
        hop = CFG["hop_length"]
        T = mel_db.shape[1]
        frame_period_ms = hop / sr * 1000.0  # frame period in milliseconds

        # Prepare F0 (float64, unvoiced frames = 0)
        f0_world = f0_hz.astype(np.float64).copy()
        f0_world[~(voiced_mask > 0.5)] = 0.0

        # Convert mel spectrogram back to approximate linear spectrogram
        mel_power = librosa.db_to_power(mel_db, ref=1.0)
        mel_basis = librosa.filters.mel(sr=sr, n_fft=CFG["n_fft"], n_mels=CFG["n_mels"],
                                        fmin=CFG["fmin"], fmax=CFG["fmax"])
        mel_basis_pinv = np.linalg.pinv(mel_basis)
        linear_spec = np.maximum(mel_basis_pinv @ mel_power, 1e-10)  # (fft/2+1, T)

        # Spectral envelope for WORLD: (T, fft_size//2 + 1), float64
        fft_size = CFG["n_fft"]
        n_sp = fft_size // 2 + 1
        sp = np.zeros((T, n_sp), dtype=np.float64)
        n_freq = min(linear_spec.shape[0], n_sp)
        sp[:, :n_freq] = linear_spec[:n_freq, :].T.astype(np.float64)
        # Fill remaining high frequencies with small values
        if n_freq < n_sp:
            sp[:, n_freq:] = 1e-10
        # Ensure sp is positive (required by WORLD)
        sp = np.maximum(sp, 1e-10)

        # Aperiodicity: (T, fft_size//2 + 1), float64
        # Simple model: voiced frames are periodic, unvoiced are aperiodic
        ap = np.ones((T, n_sp), dtype=np.float64) * 0.5
        for i in range(T):
            if f0_world[i] > 0:
                ap[i, :] = 0.003  # mostly periodic (small aperiodicity)
            else:
                ap[i, :] = 0.999  # mostly noise (high aperiodicity)
        # Clip to valid WORLD range
        ap = np.clip(ap, 0.001, 0.999)

        # Synthesize using pyworld
        wav = pw.synthesize(
            f0_world,           # (T,) float64 — F0 in Hz, 0 for unvoiced
            sp,                 # (T, fft_size//2+1) float64 — spectral envelope
            ap,                 # (T, fft_size//2+1) float64 — aperiodicity
            sr,                 # int — sample rate
            frame_period=frame_period_ms  # float — frame period in ms
        )

        wav = wav.astype(np.float32)
        if np.abs(wav).max() > 0:
            wav = wav / (np.abs(wav).max() + 1e-8) * 0.95
        return wav

    except Exception as e:
        print(f"  WORLD synthesis failed: {e}")
        print(f"  Falling back to Griffin-Lim")
        return mel_to_audio_griffinlim(mel_db)

def load_real_wav(row):
    if COL_WAV is None: return None
    p = resolve_path(row[COL_WAV])
    if p is None: return None
    try:
        wav, _ = librosa.load(str(p), sr=CFG["sample_rate"], mono=True)
        return wav.astype(np.float32)
    except: return None

def estimate_f0_from_wav(wav):
    try:
        f0, _, _ = librosa.pyin(wav, fmin=50, fmax=500, sr=CFG["sample_rate"],
                                frame_length=CFG["win_length"], hop_length=CFG["hop_length"])
        return np.nan_to_num(f0, nan=0.0).astype(np.float32)
    except: return np.zeros(1, dtype=np.float32)

@torch.no_grad()
def generate_from_pair(ds, idx):
    G.eval()
    b = ds[idx]
    sm = b["src_mel"].unsqueeze(0).to(DEVICE)
    sa3 = b["src_aux"].unsqueeze(0).to(DEVICE)
    tf0 = b["transformed_f0"].unsqueeze(0).unsqueeze(1).to(DEVICE)
    sa4 = torch.cat([sa3, tf0], dim=1)
    spk = b["spk_id"].view(1).to(DEVICE)
    te = b["tgt_emo"].view(1).to(DEVICE)
    pc = b["prosody_cond"].unsqueeze(0).to(DEVICE)

    fake_n, f0p, ep, vp = G(sm, sa4, spk, te, pc, return_prosody=True)
    fake_n = fake_n[0].cpu().numpy()
    f0p = f0p[0].cpu().numpy()
    vp = vp[0].cpu().numpy()

    T = b["src_mel"].shape[-1]
    src_db = denormalize_mel(b["src_mel"].numpy())[:, :T]
    tgt_db = denormalize_mel(b["tgt_mel"].numpy())[:, :T]
    gen_db = denormalize_mel(fake_n)[:, :T]

    gen_f0_hz = denormalize_f0(f0p[:T], vp[:T], STATS)
    src_aux_np = b["src_aux"].numpy()[:, :T]
    tgt_aux_np = b["tgt_aux"].numpy()[:, :T]
    src_f0 = denormalize_f0(src_aux_np[0], src_aux_np[2], STATS)
    tgt_f0 = denormalize_f0(tgt_aux_np[0], tgt_aux_np[2], STATS)

    pr = ds.pairs_df.iloc[int(b["pair_index"])]
    src_row = df_work.iloc[int(pr["src_idx"])]; tgt_row = df_work.iloc[int(pr["tgt_idx"])]
    src_wav = load_real_wav(src_row) or mel_to_audio_griffinlim(src_db)
    tgt_wav = load_real_wav(tgt_row) or mel_to_audio_griffinlim(tgt_db)

    # v3: Use WORLD vocoder with predicted F0
    if CFG["use_world_vocoder"]:
        gen_wav = world_synthesize(gen_db, gen_f0_hz, vp[:T])
    else:
        gen_wav = mel_to_audio_griffinlim(gen_db)

    gen_f0_wav = estimate_f0_from_wav(gen_wav)

    return {"src_db": src_db, "tgt_db": tgt_db, "gen_db": gen_db,
            "src_f0": src_f0, "tgt_f0": tgt_f0, "gen_f0_pred": gen_f0_hz,
            "gen_f0_wav": gen_f0_wav,
            "src_energy": derive_energy_from_mel_db(src_db),
            "tgt_energy": derive_energy_from_mel_db(tgt_db),
            "gen_energy": derive_energy_from_mel_db(gen_db),
            "src_wav": src_wav, "tgt_wav": tgt_wav, "gen_wav": gen_wav,
            "src_emotion": id_to_emotion[int(b["src_emo"])],
            "tgt_emotion": id_to_emotion[int(b["tgt_emo"])],
            "speaker": id_to_speaker[int(b["spk_id"])]}

# Quick test of WORLD vocoder
print("Testing WORLD vocoder...")
test_f0 = np.zeros(100, dtype=np.float64); test_f0[10:90] = 200.0
test_sp = np.ones((100, CFG["n_fft"]//2+1), dtype=np.float64) * 1e-5
test_ap = np.ones((100, CFG["n_fft"]//2+1), dtype=np.float64) * 0.5
test_wav = pw.synthesize(test_f0, test_sp, test_ap, CFG["sample_rate"], frame_period=CFG["hop_length"]/CFG["sample_rate"]*1000)
print(f"  WORLD synthesis OK! Output: {len(test_wav)} samples ({len(test_wav)/CFG['sample_rate']:.2f}s)")
print("Inference pipeline ready.")
""")


# ============ CELL 13: Visualization + Listen ============
md("## 13 · Visualization + Audio Playback + Evaluation")

code("""# ============================================================
# BLOCK 13 — Plots + Audio + Evaluation
# ============================================================

def compare(idx=0, save_prefix=None):
    out = generate_from_pair(val_ds, idx)
    t = f"{out['speaker']} | {out['src_emotion']} -> {out['tgt_emotion']}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, db, label in zip(axes, [out["src_db"], out["tgt_db"], out["gen_db"]],
                              ["Source (neutral)", f"Target ({out['tgt_emotion']})", f"Generated ({out['tgt_emotion']})"]):
        librosa.display.specshow(db, sr=CFG["sample_rate"], hop_length=CFG["hop_length"],
                                 x_axis="time", y_axis="mel", fmax=CFG["fmax"], ax=ax, cmap="magma")
        ax.set_title(label)
    fig.suptitle(t); plt.tight_layout()
    if save_prefix: fig.savefig(PLOT_DIR / f"{save_prefix}_mel.png", dpi=150, bbox_inches="tight")
    plt.show()

    # F0 comparison (THE diagnostic plot)
    plt.figure(figsize=(14, 4))
    ts = np.arange(len(out["src_f0"])) * CFG["hop_length"] / CFG["sample_rate"]
    tt = np.arange(len(out["tgt_f0"])) * CFG["hop_length"] / CFG["sample_rate"]
    tg = np.arange(len(out["gen_f0_pred"])) * CFG["hop_length"] / CFG["sample_rate"]
    plt.plot(ts, out["src_f0"], 'b-', alpha=0.5, label="Source (neutral)")
    plt.plot(tt, out["tgt_f0"], 'r-', alpha=0.7, label=f"Target ({out['tgt_emotion']})")
    plt.plot(tg, out["gen_f0_pred"], 'g-', lw=2, label="Generated (predicted F0)")
    plt.title("F0 — Green should track RED, not BLUE"); plt.xlabel("Time (s)"); plt.ylabel("Hz")
    plt.legend(); plt.grid(True, alpha=0.3)
    if save_prefix: plt.savefig(PLOT_DIR / f"{save_prefix}_f0.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Audio
    for label, wav in [("SOURCE", out["src_wav"]), ("TARGET", out["tgt_wav"]), ("GENERATED", out["gen_wav"])]:
        print(f"  {label}:")
        display(Audio(wav, rate=CFG["sample_rate"]))
    if save_prefix:
        sf.write(AUDIO_DIR / f"{save_prefix}_source.wav", out["src_wav"], CFG["sample_rate"])
        sf.write(AUDIO_DIR / f"{save_prefix}_target.wav", out["tgt_wav"], CFG["sample_rate"])
        sf.write(AUDIO_DIR / f"{save_prefix}_generated.wav", out["gen_wav"], CFG["sample_rate"])
    return out

# Run evaluation
for i in range(min(5, len(val_ds))):
    print(f"\\n{'='*50} Sample {i} {'='*50}")
    compare(i, save_prefix=f"val_{i:03d}")

# Batch evaluation
print("\\n\\n" + "="*60 + "\\n  BATCH EVALUATION\\n" + "="*60)
results = []
for i in tqdm(range(min(60, len(val_ds))), desc="Evaluating"):
    out = generate_from_pair(val_ds, i)
    sf0 = out["src_f0"][out["src_f0"]>0]; tf0 = out["tgt_f0"][out["tgt_f0"]>0]
    gf0 = out["gen_f0_pred"][out["gen_f0_pred"]>0]
    sm = float(sf0.mean()) if len(sf0) > 0 else 0
    tm = float(tf0.mean()) if len(tf0) > 0 else 0
    gm = float(gf0.mean()) if len(gf0) > 0 else 0
    moved_f0 = int(abs(gm - tm) < abs(sm - tm))
    results.append({"idx": i, "tgt_emotion": out["tgt_emotion"], "src_f0": round(sm,1),
                    "gen_f0": round(gm,1), "tgt_f0": round(tm,1), "moved_f0": moved_f0})

eval_df = pd.DataFrame(results)
eval_df.to_csv(OUT_DIR / "honest_evaluation_v3.csv", index=False)
print(f"\\nF0 moved toward target: {eval_df['moved_f0'].mean():.3f}")
print("\\nPer emotion:")
print(eval_df.groupby("tgt_emotion")[["moved_f0", "src_f0", "gen_f0", "tgt_f0"]].mean().round(1))
""")


# ============ CELL 14: Export ============
md("## 14 · Export Results to Drive")

code("""# ============================================================
# BLOCK 14 — Export
# ============================================================

print(f"All outputs saved to: {OUT_DIR}")
print(f"\\nContents:")
for f in sorted(OUT_DIR.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(OUT_DIR)} ({f.stat().st_size/1024:.1f} KB)")

# Also create a zip in Drive for easy download
zip_path = DRIVE_ROOT / "evc_v3_results.zip"
if zip_path.exists(): zip_path.unlink()
shutil.make_archive(str(zip_path.with_suffix("")), "zip", OUT_DIR)
print(f"\\nZIP: {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")
print("\\nDone! Check Google Drive -> MyDrive -> EVC_Output/")
""")

# ============ Build the notebook JSON ============
notebook = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0", "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3", "file_extension": ".py"},
        "accelerator": "GPU",
        "gpuClass": "premium",
        "colab": {"provenance": [], "gpuType": "A100"}
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells
}

output_path = "/projects/sandbox/Bengali_Speech_generation/bengali_evc_v3_colab.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook: {output_path}")
print(f"Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
