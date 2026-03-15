# INSTRUMENTAL

**Automatic synthesizer parameter recovery from audio.**

Given a sound from a song, INSTRUMENTAL recovers the synth settings that recreate it — not just the notes, but the actual instrument.

> Every DAW can tell you *what notes are playing*. None can tell you *what synth patch made that sound*. We solve the second problem.

---

## Demo

**Input:** 4.8s lead synth clip from a song (22 notes, 3 pitches)

**Output:** 28 synthesizer parameters that reproduce the sound

| | Loss | Centroid Δ | Notes |
|---|---|---|---|
| Silence | 1564.6 | — | Worst possible |
| Random synth | 52.8 | — | Noise |
| **Our best match** | **2.09** | **133 Hz** | **Top 0.13% of loss range** |

🔊 Audio comparison: [`ui/original.mp3`](ui/original.mp3) vs [`ui/matched.mp3`](ui/matched.mp3)

---

## How It Works

```
Audio file
  → Demucs (source separation)
  → TorchCrepe (pitch detection)
  → 22 notes extracted at A3, C#4, D4
  → CMA-ES optimization (28-param differentiable synth × 100K evals)
  → Best-match synth parameters
  → Rendered output audio
```

**The synth:** 28-parameter differentiable subtractive synthesizer in PyTorch:
- 4 oscillator types (saw, pulse, sine, noise) with 5-voice unison
- Sigmoid low-pass filter with learnable slope + 2-band parametric EQ
- Dual ADSR envelopes (amplitude + filter)
- Delay-line reverb

**The optimizer:** CMA-ES with spectral initialization — we analyze the target's spectrum to set smart starting parameters instead of random search.

**The loss:** Mel-scaled multi-res STFT + spectral centroid + MFCC (captures brightness, timbre, and spectral shape).

---

## Key Findings

### 1. CMA-ES beats gradient descent on this problem
Adam gets trapped in local minima (loss 4.60). CMA-ES escapes them (loss 2.09). A PPO RL agent matches CMA-ES but doesn't beat it after 1M training steps.

### 2. More parameters ≠ better
Expanding from 24 → 29 params (adding unconstrained distortion, delay, vibrato) made results **worse** — CMA-ES exploited extreme values (+24 semitone detune, 43% reverb) to "cheat" the loss. Constrained, meaningful parameters outperform unconstrained expansion.

### 3. Parametric EQ was the breakthrough
A low-pass filter can only *remove* frequencies. The target sound had mid-high presence (1–5 kHz) that required *boosting*. Adding a 2-band EQ — the only modification out of 8 tested that improved results — found a +1 dB peak at 5.8 kHz.

### 4. 90% of convergence happens in 10K evals
Out of 100K evaluations, the first 10K achieve 90% of the total improvement. The remaining 90K yield only 0.07 loss reduction — an architectural floor.

### 5. The harmonic gap reveals synthesis limits
The target has H3 > H2 (3rd harmonic stronger than 2nd) — impossible with sawtooth or square waves. This identifies the floor of subtractive synthesis and points to FM/wavetable as the next step.

---

## Hypothesis Testing

We systematically tested 8 modifications (10K evals each):

| Modification | Δ Loss | Verdict |
|---|---|---|
| Chebyshev waveshaping | +29% | ✗ |
| **Parametric EQ** | **−8%** | **✓** |
| TFS / envelope loss | +17% | ✗ |
| SPSA fine-tuning | +11% | ✗ |
| Multi-start CMA-ES (8×) | +29% | ✗ |
| FM synthesis | +17% | ✗ |
| CDPAM perceptual loss | skipped | — |
| CLAP embedding loss | converged | different quality |

Only one thing worked. Everything else made it worse.

---

## Performance

| Configuration | Evals/s | 100K evals |
|---|---|---|
| CPU sequential | 102/s | 16 min |
| CPU multiprocessing (10 cores) | 229/s | 7 min |
| Batched tensor ops (no loops) | 553/s | 3 min |
| Batched + GPU synth (MPS) | 1,170/s | 1.4 min |

**Hardware tested:** Apple M2 (8GB), Apple M4 Mini (16GB), Azure D48s_v5 (48 vCPUs). M4 Mini was fastest — Apple Silicon per-core performance dominates Azure EPYC even at 5× fewer cores.

---

## Project Structure

```
instrumental/
├── src/
│   ├── synth.py           # 28-param differentiable synthesizer
│   ├── synth_gpu.py        # Batched GPU variant
│   ├── losses.py           # 8 loss functions (STFT, mel, MFCC, centroid, etc.)
│   ├── batch_loss.py       # Batched loss for CMA-ES
│   ├── optimize.py         # CMA-ES + Adam optimization pipeline
│   ├── spectral_init.py    # Spectral analysis → smart init params
│   └── cmaes_search.py     # CMA-ES wrapper
├── paper/
│   └── main.tex            # Research paper (4 pages, LaTeX)
├── ui/
│   └── index.html          # Visualization dashboard
├── train_rl.py             # RL agent (PPO, gymnasium)
├── run_hypotheses.py       # 8-hypothesis test harness
├── run_cmaes_gpu.py        # Batched CMA-ES runner
├── notes/                  # 22 extracted target notes (.wav)
├── output/                 # Matched audio + paper PDF
├── DIRECTIONS.md           # All possible approaches explored
└── DELIVERABLES.md         # SMART criteria for Levels 0–8
```

---

## Quick Start

```bash
pip install torch torchaudio torchsynth auraloss soundfile librosa cma

# Play the comparison
python play_notes.py

# Run CMA-ES optimization on a target sound
python run_cmaes_v2.py

# Run all 8 hypothesis tests
python run_hypotheses.py

# View the UI
cd ui && python -m http.server 8080
```

---

## What Makes This Novel

We surveyed every major DAW and audio tool:

| Tool | Extracts Notes | Identifies Instrument | Recovers Synth Params |
|---|---|---|---|
| Logic Pro | ✓ | ✗ | ✗ |
| Ableton Live | ✓ | ✗ | ✗ |
| Melodyne | ✓ | ✗ | ✗ |
| Basic Pitch | ✓ | ✗ | ✗ |
| MT3 / YourMT3+ | ✓ | ✓ (class only) | ✗ |
| Syntheon | ✗ | ✗ | ✓ (Vital only) |
| **INSTRUMENTAL** | **✓** | **✓** | **✓** |

No existing tool does all three.

---

## Research Paper

4-page paper with equations, figures, and 10 references: [`output/paper.pdf`](output/paper.pdf)

Built at the [Mozart AI Hack](https://lu.ma/mozartai), London, March 14–15 2026.

---

*Built with Claude Code.*
