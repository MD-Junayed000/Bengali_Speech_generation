# EVC Emotion Injection — Diagnosis & Fix Plan

## What the Evaluation Images Actually Show

### Image 2 — F0 (most important)
The generated F0 (green) tracks the **source** F0 (blue), not the **target** F0 (orange). At 1.5–2.0s the target (angry) holds a high sustained pitch around 250 Hz; the generated line drops to follow the source contour and then goes silent.

> **Pitch contour is the single biggest carrier of emotion — and it is copying neutral. This is the core problem.**

### Images 3 & 4 — Energy
The generated energy partially tracks the target. Green follows orange more than blue in places. The energy fix is **partially working**.

### Image 5 — Prosody Drift
Most points sit slightly above the no-change line, and the angry points (blue) cluster higher than sad (green). Energy is moving — just weakly and not audibly.

---

## Root Cause Diagnosis

The v2 fixes are operating on the **wrong channel**.

The generator's emotion comes almost entirely through the decoder producing a mel spectrogram. But the F0 contour the listener hears is reconstructed by Griffin-Lim from that mel — and `prosody_loss` only constrains **energy** (mean over mel bands), not pitch.

**F0 lives in the harmonic spacing of the mel**, which none of the losses explicitly push toward the target. The model has zero incentive to change pitch, and pitch is what makes "angry" sound angry.

Worse: the generator never receives the target F0 as input or as a target. It gets `src_aux` (the source's F0/energy/voiced) and is asked to produce the target emotion's mel — but with no pitch supervision, **it just passes source pitch through**.

---

## The Fix Plan (priority order)

### Fix 1 — Make F0 a First-Class Predicted Output *(most important)*

Right now the generator only outputs a mel. It should also **predict an F0 contour** and energy contour, supervised directly against the DTW-aligned target's F0/energy (already in `tgt_aux`).

**What this means:**
- Add a small `ProsodyHead` to the generator that outputs predicted F0 and energy sequences
- Add a direct loss: `L1(predicted_F0, target_F0)` on voiced frames, and `L1(predicted_energy, target_energy)`
- At inference, use the predicted F0 to reconstruct audio instead of letting Griffin-Lim invent pitch from the mel

This is the **real fix**. Working emotion conversion systems (StarGAN-VC, CycleGAN-VC) all convert F0 explicitly — usually with log-F0 linear transformation matching target statistics. The current setup does none of that.

---

### Fix 2 — Feed the Generator the Target Prosody Intention

The generator should know what the target emotion's prosody looks like. Two options:

**Option A:** Condition the decoder on target F0/energy statistics (mean, std per emotion), computed from training data per emotion.

**Option B:** Do the classic log-F0 mean/variance transformation — shift and scale the source F0 toward the target emotion's F0 distribution as an explicit, non-learned baseline that the model then refines.

---

### Fix 3 — Extend Prosody Loss to Cover Pitch and Reweight It

`prosody_loss` currently only touches energy. Extend it to:

- F0 mean difference (angry/happy → raise it, sad → lower it)
- F0 variance/range (angry/happy → wide, sad → narrow)
- Weight it **much higher** — this is the signal that matters

---

## Fine-Tuning from the Latest Checkpoint

### The Honest Tradeoff

The current checkpoint has learned a generator whose content encoder and decoder are wired to reproduce source prosody. Fine-tuning can work, but the model has a strong basin-of-attraction toward "copy neutral." Adding F0 supervision and continuing training will pull it out, but slowly.

### Recommended Protocol

1. **Load the latest checkpoint** — the architecture change (adding `ProsodyHead`) means new layers are initialised fresh; the rest loads from the checkpoint via `strict=False`
2. **Freeze nothing**, but run a short **prosody-only warmup** — a few epochs where F0/energy prosody loss dominates and reconstruction L1 is very low, to force the model to discover pitch conversion
3. **Resume normal balanced training**

This is faster than training from scratch and reuses invested compute, but the new prosody head needs the warmup epochs to become useful.

---

## Checkpoint Loading Issue *(must fix first)*

Before any of the above — the last run crashed with `KeyError: 'G'`. This means the checkpoint in the input dataset is not a full EVC checkpoint (likely a raw state dict or the SER-only file).

**Action required:** Run the diagnostic and check what keys are in the checkpoint directory. "Fine-tune from latest checkpoint" depends entirely on loading it correctly.

---

## Execution Order

| Step | Action |
|------|--------|
| 1 | Confirm / fix checkpoint loading (check diagnostic output for keys) |
| 2 | Add F0 as a predicted, target-supervised output (`ProsodyHead` + `f0_supervision_loss`) |
| 3 | Condition on target prosody + log-F0 transformation baseline |
| 4 | Extend `prosody_loss` to pitch and weight it heavily |
| 5 | Fine-tune with prosody-warmup protocol |
