# Inverse Synthesis: All Possible Directions

> **Project**: .wav → MIDI instruments (sound matching / automatic synthesizer programming)
> **Hackathon**: Mozart AI Hack, 14-15 March 2026, Halkin offices, London
> **Goal**: Given a sound from a song, recover instrument/effect settings that recreate it

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Step 1: Source Separation](#3-step-1-source-separation)
4. [Step 2: Pitch Detection](#4-step-2-pitch-detection)
5. [Step 3: Synth / Sound Engine](#5-step-3-synth--sound-engine)
6. [Step 4: Loss Function](#6-step-4-loss-function)
7. [Step 5: Optimizer](#7-step-5-optimizer)
8. [Architecture Combinations](#8-architecture-combinations)
9. [Hypotheses to Test](#9-hypotheses-to-test)
10. [Simplification Ladder](#10-simplification-ladder)
11. [Known Pitfalls](#11-known-pitfalls)
12. [Fallback Plans](#12-fallback-plans)
13. [References & Tools Index](#13-references--tools-index)

---

## 1. Problem Definition

**Forward problem** (easy): synth settings → sound
**Inverse problem** (hard): sound → synth settings

This is ill-posed: many different settings can produce perceptually similar sounds. We're not looking for THE original settings — we're looking for ANY settings that sound close enough.

### What "close enough" could mean

| Level | Definition | Metric |
|-------|-----------|--------|
| **Waveform match** | Samples are numerically close | L1/L2 distance on raw waveform |
| **Spectral match** | Frequency content is similar | STFT magnitude distance |
| **Timbral match** | "Sounds like the same instrument" | MFCC distance, spectral centroid |
| **Perceptual match** | Humans can't tell them apart | CLAP embedding distance, listening test |

### Input → Output

```
Input:  A .wav file (a song, or an isolated clip)
Output: A set of synth parameters that, when rendered at the detected pitch,
        produce audio perceptually matching the input
```

---

## 2. Pipeline Overview

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│  .wav file   │────▶│ Source Separation │────▶│ Isolated Stem │
│  (full song) │     │ (Demucs/Spleeter)│     │ (one instr.) │
└─────────────┘     └─────────────────┘     └──────┬───────┘
                                                    │
                                            ┌───────▼───────┐
                                            │ Pitch Detection │
                                            │ (CREPE / pyin)  │
                                            └───────┬───────┘
                                                    │
                                                    ▼
                              ┌─────────────────────────────────────┐
                              │         Optimization Loop           │
                              │                                     │
                              │  ┌──────────┐    ┌──────────────┐  │
                              │  │  Synth    │───▶│  Generated   │  │
                              │  │  Engine   │    │  Audio       │  │
                              │  │ (params)  │    └──────┬───────┘  │
                              │  └─────▲────┘           │          │
                              │        │         ┌──────▼───────┐  │
                              │        │         │ Loss Function │  │
                              │   adjust         │ (compare to   │  │
                              │   params         │  target)      │  │
                              │        │         └──────┬───────┘  │
                              │        │                │          │
                              │   ┌────┴────────────────▼──┐      │
                              │   │      Optimizer          │      │
                              │   │ (gradient / CMA-ES /    │      │
                              │   │  Bayesian / random)     │      │
                              │   └─────────────────────────┘      │
                              └─────────────────────────────────────┘
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │ Best Parameters │
                                            │ (synth patch)   │
                                            └───────────────┘
```

Each box in the pipeline has **multiple implementation options**. The rest of this document enumerates all of them.

---

## 3. Step 1: Source Separation

**Purpose**: Isolate a single instrument from a mixed song so we have a clean target signal.

**Can be skipped if**: You already have an isolated instrument recording or a clean sample.

### Option A: Demucs (HTDemucs) — RECOMMENDED

- **What**: Meta's state-of-the-art source separation. Best quality available (2026).
- **Stems**: vocals, drums, bass, other (4 stems)
- **Install**: `pip install demucs`
- **Quality**: SDR ~7.5 on MUSDB18 (significantly better than Spleeter)
- **Speed**: ~2-3x slower than Spleeter, still fast enough for hackathon
- **Usage**:
  ```bash
  # CLI
  python -m demucs song.wav --out output/

  # Python API
  from demucs import pretrained
  from demucs.apply import apply_model
  import torchaudio

  model = pretrained.get_model('htdemucs')
  wav, sr = torchaudio.load('song.wav')
  sources = apply_model(model, wav[None])
  # sources shape: (1, 4, 2, samples) — 4 stems, stereo
  ```

### Option B: Spleeter

- **What**: Deezer's source separation (see our earlier research)
- **Install**: `pip install spleeter`
- **Quality**: SDR ~5.5 — noticeably worse than Demucs
- **Speed**: Fastest option (~100x realtime on GPU)
- **Limitation**: Effectively unmaintained, TF dependency issues
- **When to use**: If Demucs install fails or GPU memory is limited

### Option C: No separation — use isolated samples

- **What**: Skip separation entirely, use pre-recorded isolated instrument samples
- **When to use**: For initial testing / proof of concept
- **Source**: Freesound.org, MIDI-rendered samples, recording your own

### Option D: Manual selection

- **What**: User selects a segment of the song where one instrument is dominant
- **When to use**: When separation quality isn't good enough
- **Simplest approach**: Just let the user crop a section

---

## 4. Step 2: Pitch Detection

**Purpose**: Determine what note(s) the isolated instrument is playing, so we can render our synth at the same pitch.

### Option A: torchcrepe — RECOMMENDED (PyTorch, matches Demucs stack)

- **What**: PyTorch reimplementation of CREPE pitch tracker. Partially differentiable.
- **Install**: `pip install torchcrepe`
- **Usage**:
  ```python
  import torchcrepe
  import torchaudio

  audio, sr = torchaudio.load('isolated_instrument.wav')
  pitch = torchcrepe.predict(audio, sr, hop_length=256,
                              fmin=50, fmax=2006, model='tiny')
  # pitch in Hz, frame-by-frame
  ```
- **Strengths**: Same PyTorch stack as Demucs/torchsynth/auraloss, partially differentiable, 508 stars
- **Limitation**: Monophonic only (one note at a time)

### Option A2: CREPE (TensorFlow version)

- **What**: Original CREPE in TensorFlow. Very accurate for single notes.
- **Install**: `pip install crepe`
- **Strengths**: Best accuracy for monophonic signals, handles vibrato well
- **Limitation**: TensorFlow dependency — may conflict with PyTorch stack. Use torchcrepe instead.

### Option B: pyin (librosa)

- **What**: Probabilistic YIN algorithm, built into librosa
- **Install**: `pip install librosa` (already likely installed)
- **Usage**:
  ```python
  import librosa
  f0, voiced_flag, voiced_probs = librosa.pyin(
      audio, fmin=65, fmax=2093, sr=sr
  )
  ```
- **Strengths**: No extra dependency, fast, good enough for most cases
- **Limitation**: Less accurate than CREPE on noisy signals

### Option C: Basic FFT peak detection

- **What**: Find the dominant frequency via FFT
- **When to use**: Simplest possible approach, clean signals only
- **Implementation**: `np.fft.fft(audio)` → find peak → convert to Hz
- **Limitation**: Fails on harmonics-rich signals (might detect overtone, not fundamental)

### Option D: Manual / known pitch

- **What**: User specifies the note (e.g., "it's playing A4 = 440 Hz")
- **When to use**: For controlled testing with known signals

---

## 5. Step 3: Synth / Sound Engine

**Purpose**: A controllable sound source with parameters we can adjust to match the target.

This is the MOST CRITICAL choice. It determines what sounds you can recreate and whether you can use gradient-based optimization.

### Option A: DDSP (Google) — Differentiable

- **What**: Differentiable Digital Signal Processing. Neural network + signal processing. Includes additive synthesis, subtractive synthesis, reverb, all differentiable.
- **Install**: `pip install ddsp`
- **Differentiable**: YES — gradients flow through the entire synth
- **Sound palette**: Harmonic sounds (strings, brass, woodwinds, voice). Weak on percussion, electronic sounds.
- **Key components**: Harmonic oscillator bank + filtered noise + reverb
- **Parameters**: ~hundreds (oscillator amplitudes, filter coefficients, reverb params)
- **Usage**:
  ```python
  import ddsp
  import ddsp.training

  # Create a processor group (additive synth + noise + reverb)
  processor_group = ddsp.processors.ProcessorGroup(
      dag=[
          (ddsp.synths.Additive(), ['amplitudes', 'harmonic_distribution', 'f0_hz']),
          (ddsp.synths.FilteredNoise(), ['noise_magnitudes']),
          (ddsp.effects.Reverb(), ['reverb_ir']),
          (ddsp.processors.Add(), ['additive/signal', 'filtered_noise/signal']),
      ]
  )
  ```
- **Strengths**: Gradient descent works natively, Google-maintained, well-documented
- **Limitations**: TensorFlow-based (might conflict with Demucs which is PyTorch), limited to harmonic sounds
- **Maturity**: Production-ready, used in Magenta Studio

### Option B: SynthAX — Differentiable, PyTorch-friendly

- **What**: Differentiable synthesizer built in JAX. Designed for inverse synthesis.
- **Install**: `pip install synthax`
- **Differentiable**: YES
- **Sound palette**: Subtractive synthesis (oscillators, filters, envelopes, LFOs)
- **Parameters**: ~50-100 (osc type, cutoff, resonance, envelope ADSR, LFO rate, etc.)
- **Strengths**: Purpose-built for sound matching, JAX is fast
- **Limitations**: Relatively new, smaller community, JAX dependency
- **Maturity**: Research code, usable but less polished than DDSP

### Option C: TorchSynth — Differentiable, PyTorch

- **What**: Modular differentiable synthesizer in PyTorch
- **Install**: `pip install torchsynth`
- **Differentiable**: YES
- **Sound palette**: Modular synthesis (oscillators, filters, envelopes, VCAs)
- **Parameters**: Varies by module configuration
- **Strengths**: Native PyTorch (same framework as Demucs), modular design
- **Limitations**: Documentation may be sparse
- **Maturity**: Research code

### Option D: FluidSynth — NOT differentiable

- **What**: SoundFont-based synthesizer. Plays MIDI through sampled instrument patches (SF2 files).
- **Install**: `pip install pyfluidsynth`
- **Differentiable**: NO — it's a sample player, not a parametric synth
- **Sound palette**: Whatever SoundFonts you load (pianos, strings, orchestral, GM sounds)
- **Parameters**: Few controllable params (volume, pan, reverb send, chorus send)
- **When to use**: If you want to match against a fixed set of preset instruments rather than fine-tuning individual parameters
- **Approach**: Try every preset in a SoundFont bank, score each one, return the best match
- **Limitations**: Can't fine-tune the sound, only select from presets

### Option E: pyo — NOT differentiable, but flexible

- **What**: Python audio synthesis library. Full-featured: oscillators, filters, effects, granular, etc.
- **Install**: `pip install pyo`
- **Differentiable**: NO
- **Sound palette**: Very wide — anything you can build from oscillators, filters, effects
- **Parameters**: Whatever you expose
- **When to use**: If you want maximum flexibility with derivative-free optimization
- **Limitations**: Real-time oriented (not designed for batch rendering), not differentiable

### Option F: Custom minimal synth in scipy/numpy

- **What**: Build a simple synth from scratch using numpy
- **Differentiable**: Can be made differentiable with autograd (JAX/PyTorch)
- **Sound palette**: Whatever you implement (start with: sine + sawtooth + square + noise + filter + ADSR envelope)
- **Parameters**: You define them (typically 10-30 for a basic subtractive synth)
- **When to use**: Maximum control, minimum dependencies, educational
- **Strengths**: No dependency issues, full understanding of what's happening
- **Limitations**: Won't sound as rich as real synths, significant implementation effort
- **Implementation sketch**:
  ```python
  import numpy as np

  def simple_synth(f0, duration, sr, params):
      """Minimal subtractive synth."""
      t = np.linspace(0, duration, int(sr * duration))

      # Oscillator (mix of waveforms)
      saw = 2 * (t * f0 % 1) - 1
      sine = np.sin(2 * np.pi * f0 * t)
      signal = params['saw_mix'] * saw + (1 - params['saw_mix']) * sine

      # Simple low-pass filter (1-pole)
      cutoff = params['filter_cutoff']  # 0-1
      filtered = np.zeros_like(signal)
      filtered[0] = signal[0]
      for i in range(1, len(signal)):
          filtered[i] = cutoff * signal[i] + (1 - cutoff) * filtered[i-1]

      # ADSR envelope
      envelope = make_adsr(len(t), params['attack'], params['decay'],
                           params['sustain'], params['release'])

      return filtered * envelope
  ```

### Option G: DiffMoog — Differentiable Moog-style synth

- **What**: PyTorch-based differentiable modular synthesizer: FM/AM synthesis, LFOs, envelopes, filters. Full Moog-style signal chain. Published Jan 2024.
- **Install**: Clone from `github.com/aisynth/diffmoog` + `pip install -r requirements.txt` (NO pip package)
- **Differentiable**: YES
- **Sound palette**: FM synthesis, subtractive, modular — wider than DDSP
- **Limitations**: No pip install, requires manual setup, research code
- **When to use**: Only if torchsynth doesn't cover your needs and you have setup time
- **Maturity**: Research code only

### Option H: VST plugin via DawDreamer

- **What**: Load actual VST/AU synthesizer plugins and control them from Python
- **Install**: `pip install dawdreamer`
- **Differentiable**: NO (VST plugins are black boxes)
- **Sound palette**: ANY VST synth (Serum, Vital, Diva, etc.)
- **Parameters**: Whatever the VST exposes (can be hundreds)
- **When to use**: If you want to match sounds using real-world synths
- **Strengths**: Most realistic sound palette, industry-standard synths
- **Limitations**: Need VST plugin files, not differentiable, rendering is slow
- **Maturity**: Well-maintained, active development

---

## 6. Step 4: Loss Function

**Purpose**: Quantify how different the generated sound is from the target sound. This is what the optimizer minimizes.

### Option A: Multi-Resolution STFT Loss — RECOMMENDED

- **What**: Compare spectrograms at multiple window sizes. Captures both fine time detail (small window) and fine frequency detail (large window).
- **Install**: `pip install auraloss`
- **Components**: Spectral convergence + log magnitude distance, summed over 3-5 STFT resolutions
- **Usage**:
  ```python
  import auraloss
  loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
      fft_sizes=[1024, 2048, 512],
      hop_sizes=[256, 512, 128],
      win_lengths=[1024, 2048, 512]
  )
  loss = loss_fn(generated_audio, target_audio)
  ```
- **Differentiable**: YES
- **Strengths**: Good balance of perceptual relevance and computational cost
- **Limitations**: Doesn't capture phase well, can miss subtle timbral differences

### Option B: MFCC Distance

- **What**: Compare Mel-Frequency Cepstral Coefficients — standard features for timbre.
- **Install**: `pip install librosa` (built-in)
- **Usage**:
  ```python
  import librosa
  import numpy as np

  mfcc_target = librosa.feature.mfcc(y=target, sr=sr, n_mfcc=13)
  mfcc_generated = librosa.feature.mfcc(y=generated, sr=sr, n_mfcc=13)
  loss = np.mean((mfcc_target - mfcc_generated) ** 2)
  ```
- **Differentiable**: Can be made differentiable with torchaudio
- **Strengths**: Perceptually motivated, compact representation, very fast
- **Limitations**: Loses fine spectral detail, phase-blind

### Option C: CLAP Embedding Distance

- **What**: Use a neural audio embedding model (CLAP = Contrastive Language-Audio Pretraining) to compute perceptual similarity.
- **Install**: `pip install laion-clap` or `pip install transformers`
- **Usage**:
  ```python
  from transformers import ClapModel, ClapProcessor
  model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
  processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

  inputs_target = processor(audios=[target_audio], return_tensors="pt", sampling_rate=48000)
  inputs_gen = processor(audios=[generated_audio], return_tensors="pt", sampling_rate=48000)

  emb_target = model.get_audio_features(**inputs_target)
  emb_gen = model.get_audio_features(**inputs_gen)

  loss = 1 - torch.nn.functional.cosine_similarity(emb_target, emb_gen)
  ```
- **Differentiable**: Partially (embedding extraction is differentiable, but the CLAP model itself is frozen)
- **Strengths**: Most perceptually accurate, captures high-level "sounds like" similarity
- **Limitations**: Slow (runs a neural network), coarse-grained (might not distinguish subtle parameter changes)

### Option D: Raw Waveform L1/L2

- **What**: Direct sample-by-sample comparison
- **Usage**: `loss = torch.mean(torch.abs(target - generated))`
- **Differentiable**: YES
- **Strengths**: Simple, fast
- **Limitations**: Phase-sensitive (a phase-shifted identical sound has high loss), doesn't correlate well with human perception. **Expected to perform worst (H3).**

### Option E: Spectral Centroid + Spectral Flatness + Envelope

- **What**: Compare hand-picked audio features rather than full spectrogram
- **Features**: spectral centroid (brightness), spectral flatness (noisiness), RMS envelope (dynamics), spectral bandwidth, zero-crossing rate
- **Strengths**: Very fast, interpretable, each feature maps to a perceptual quality
- **Limitations**: Lossy — many different sounds can have same centroid/flatness

### Option F: Hybrid / Composite Loss

- **What**: Combine multiple losses with weights
- **Usage**: `loss = α * stft_loss + β * mfcc_loss + γ * envelope_loss`
- **Strengths**: Covers multiple perceptual aspects
- **Limitations**: Need to tune α, β, γ weights

---

## 7. Step 5: Optimizer

**Purpose**: Search the synth parameter space to minimize the loss function.

### Option A: Gradient Descent (Adam) — REQUIRES differentiable synth

- **What**: Standard backpropagation through the synth
- **Usage**:
  ```python
  params = torch.randn(n_params, requires_grad=True)
  optimizer = torch.optim.Adam([params], lr=0.01)

  for step in range(1000):
      audio = differentiable_synth(params)
      loss = loss_fn(audio, target)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
  ```
- **Strengths**: Fast convergence, precise, exploits gradient information
- **Limitations**: ONLY works with differentiable synths. Can get stuck in local minima.
- **Best with**: DDSP, SynthAX, TorchSynth, custom numpy-to-torch synth

### Option B: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

- **What**: State-of-the-art derivative-free optimizer. Maintains a population of candidate solutions and adapts the search distribution.
- **Install**: `pip install cmaes` or `pip install pycma`
- **Usage**:
  ```python
  import cma

  def objective(params):
      audio = synth.render(params)
      return loss_fn(audio, target)

  x0 = np.random.randn(n_params) * 0.5  # initial guess
  result = cma.fmin(objective, x0, sigma0=0.5, options={'maxiter': 500})
  best_params = result[0]
  ```
- **Strengths**: Works with ANY synth (no gradients needed), handles multimodal landscapes, robust
- **Limitations**: Slower than gradient descent (needs ~100-1000 evaluations), scales poorly to very high dimensions (>100 params)
- **Best with**: FluidSynth, pyo, DawDreamer/VST, any non-differentiable synth

### Option C: Bayesian Optimization

- **What**: Builds a probabilistic model of the objective function, strategically samples where improvement is likely.
- **Install**: `pip install optuna` or `pip install scikit-optimize`
- **Usage**:
  ```python
  import optuna

  def objective(trial):
      cutoff = trial.suggest_float('cutoff', 0.0, 1.0)
      resonance = trial.suggest_float('resonance', 0.0, 1.0)
      # ... more params
      audio = synth.render({'cutoff': cutoff, 'resonance': resonance})
      return loss_fn(audio, target)

  study = optuna.create_study()
  study.optimize(objective, n_trials=200)
  ```
- **Strengths**: Sample-efficient (needs fewer evaluations than CMA-ES), good for expensive objectives
- **Limitations**: Scales poorly beyond ~20 parameters
- **Best with**: Small parameter spaces, expensive synths (VST via DawDreamer)

### Option D: Random Search / Grid Search

- **What**: Just try random parameter combinations and keep the best
- **Strengths**: Dead simple, embarrassingly parallel, no failure modes
- **Limitations**: Inefficient — needs many evaluations to find good solutions
- **When to use**: As a baseline, or for initial exploration before switching to smarter search

### Option E: Nearest Preset Retrieval + Fine-tuning (Hybrid)

- **What**: Pre-compute loss for a bank of known presets, find the closest one, then fine-tune from there.
- **Strengths**: Starts from a good initial point, avoids cold-start problem
- **Approach**:
  ```
  1. Render all N presets at the detected pitch
  2. Compute loss against target for each
  3. Take top-K closest presets
  4. Fine-tune each with gradient descent or CMA-ES
  5. Return the best result
  ```
- **Best with**: VST synths that have large preset libraries

### Option F: Staged / Hierarchical Optimization

- **What**: Optimize parameter groups sequentially rather than all at once (your H2)
- **Approach**:
  ```
  Stage 1: Optimize oscillator type + pitch (coarse timbre)
  Stage 2: Optimize filter cutoff + resonance (spectral shape)
  Stage 3: Optimize envelope ADSR (dynamics)
  Stage 4: Optimize effects (reverb, delay, chorus)
  ```
- **Strengths**: Each stage has fewer parameters, avoids interference between groups
- **Limitations**: Later stages can't fix errors from earlier stages
- **Variation**: Run all-at-once first, then refine with staged optimization

---

## 8. Architecture Combinations

Every choice in steps 1-5 can be combined. Here are the most promising combos:

### Combo A: "Full Differentiable" (fastest convergence)

```
Separation: Demucs
Pitch: CREPE
Synth: DDSP or TorchSynth
Loss: Multi-Resolution STFT
Optimizer: Adam (gradient descent)
```

- **Pros**: End-to-end differentiable, fast convergence (~seconds)
- **Cons**: Limited to sounds the differentiable synth can make
- **Risk**: TensorFlow (DDSP) + PyTorch (Demucs) conflict

### Combo B: "Flexible Black-Box" (widest sound palette)

```
Separation: Demucs
Pitch: CREPE
Synth: DawDreamer + VST plugin (e.g., Vital, which is free)
Loss: Multi-Resolution STFT + MFCC
Optimizer: CMA-ES
```

- **Pros**: Can match against ANY synth, real-world sounds
- **Cons**: Slow (each evaluation renders audio), needs VST binaries
- **Risk**: CMA-ES may not converge in high-dimensional VST parameter space

### Combo C: "Minimal Hackathon" (simplest, most likely to work)

```
Separation: Skip (use isolated samples)
Pitch: Manual / known
Synth: Custom numpy synth (5-10 params)
Loss: MFCC distance
Optimizer: CMA-ES or Optuna
```

- **Pros**: Zero dependency issues, fully understood, works in an afternoon
- **Cons**: Simple sounds only, won't match complex instruments
- **Best for**: Proving the concept works before scaling up

### Combo D: "Preset Matcher" (different framing)

```
Separation: Demucs
Pitch: CREPE
Synth: FluidSynth with GM SoundFont (128 instruments)
Loss: CLAP embedding distance
Optimizer: Brute-force (try all 128 presets, rank by similarity)
```

- **Pros**: Fast, always returns an answer, no optimization needed
- **Cons**: Limited to preset palette, can't fine-tune, coarse matching
- **Good demo**: "This sounds most like preset #42: Acoustic Guitar"

### Combo E: "Hybrid Preset + Fine-tune"

```
Separation: Demucs
Pitch: CREPE
Synth: TorchSynth (differentiable, ~50 params)
Loss: Multi-Resolution STFT
Optimizer: Random search (100 trials) → pick top 5 → Adam fine-tune each
```

- **Pros**: Avoids local minima (random search for global exploration, gradient for local refinement)
- **Cons**: More complex pipeline
- **Good balance of**: robustness and precision

---

## 9. Hypotheses to Test

From the original plan:

### H1: Multi-pitch fitting gives more transferable patches than single-note fitting

- **Test**: Fit synth to a single note (e.g., C4). Then fit to 3 notes (C4, E4, G4) simultaneously. Compare how well each patch generalizes to unseen notes (A4, D5).
- **Metric**: Average loss across unseen notes
- **Why it matters**: A single note might overfit to resonances specific to that frequency. Multi-pitch fitting forces the optimizer to find the underlying instrument, not just the note.

### H2: Staged optimization is more stable than joint optimization

- **Test**: Compare all-at-once optimization vs. staged (osc → filter → envelope → effects).
- **Metric**: Final loss achieved, convergence speed, failure rate (how often it gets stuck)
- **Why it matters**: High-dimensional optimization landscapes have many local minima. Staging reduces dimensionality at each step.

### H3: Perceptual/spectral losses outperform raw waveform losses

- **Test**: Run the same optimization with L1 waveform loss vs. STFT loss vs. MFCC loss vs. CLAP distance.
- **Metric**: A/B listening test + quantitative loss on held-out comparison
- **Why it matters**: Waveform loss is phase-sensitive and doesn't match human perception. Spectral losses should be more robust.

---

## 10. Simplification Ladder

Start simple. Add complexity only when the simpler version works.

```
Level 0: Fixed synth, known pitch, known target params → verify pipeline works
Level 1: Fixed synth, known pitch, UNKNOWN target params → test optimization
Level 2: Fixed synth, detected pitch, unknown params → add pitch detection
Level 3: Fixed synth, detected pitch, unknown params, ISOLATED from song → add separation
Level 4: Fixed synth, multi-pitch fitting → test H1
Level 5: Effects chain (EQ + reverb) added to synth → harder matching
Level 6: Multiple candidate synths → "which synth is this?"
Level 7: Full pipeline: song → separate → detect → match → output patch
```

**For the hackathon**: Aim for Level 3 as the demo. Start implementation at Level 0 and climb.

---

## 11. Known Pitfalls

### Technical

| Pitfall | Symptom | Mitigation |
|---------|---------|------------|
| Phase alignment | Low spectral loss but high waveform loss | Use phase-invariant losses (STFT magnitude, MFCC) |
| Local minima | Optimizer converges to wrong sound | Multi-start optimization, random search first |
| Pitch octave errors | CREPE detects octave above/below | Sanity check: is detected f0 in expected instrument range? |
| TF/PyTorch conflict | DDSP (TF) + Demucs (PyTorch) in same process | Use subprocess for one, or pick all-PyTorch stack |
| Sample rate mismatch | Synth renders at 44.1kHz, target is 22kHz | Resample everything to common rate at start |
| Duration mismatch | Target is 0.8s, synth renders 1.0s | Pad/trim to same length before computing loss |

### Conceptual

| Pitfall | Issue | Mitigation |
|---------|-------|------------|
| Many-to-one mapping | Different settings produce same sound | Accept it — return top-K candidates, not one answer |
| Effects vs. source | Is the "sound" the raw instrument or the processed signal? | Start without effects (Level 1-4), add later |
| Polyphonic sources | Multiple notes playing simultaneously | Start monophonic (one note at a time) |
| Non-stationary sounds | Sound changes over time (e.g., wah pedal) | Start with sustained, stationary segments |

---

## 12. Fallback Plans

If the primary approach hits a wall:

| Dead end | Fallback |
|----------|----------|
| Differentiable synth can't make the target sound | Switch to CMA-ES + black-box synth |
| CMA-ES too slow | Reduce parameter space (fix some params) |
| Pitch detection fails | Ask user to specify the note |
| Source separation too noisy | Use clean isolated samples instead |
| Optimization converges to wrong sound | Try nearest-preset retrieval instead of optimization |
| Full pipeline too complex | Demo just the optimization loop with synthetic targets |
| Everything fails | Pivot to "instrument classifier" — just identify WHICH instrument, not recreate it |

---

## 13. References & Tools Index

### Python Packages

| Package | Purpose | Install | Differentiable |
|---------|---------|---------|---------------|
| `demucs` | Source separation | `pip install demucs` | N/A |
| `spleeter` | Source separation (legacy) | `pip install spleeter` | N/A |
| `crepe` | Pitch detection | `pip install crepe` | No |
| `librosa` | Audio analysis (pyin, MFCC, STFT) | `pip install librosa` | No |
| `ddsp` | Differentiable synth (TF) | `pip install ddsp` | YES |
| `torchsynth` | Differentiable synth (PyTorch) | `pip install torchsynth` | YES |
| `synthax` | Differentiable synth (JAX) | `pip install synthax` | YES |
| `dawdreamer` | VST host | `pip install dawdreamer` | No |
| `pyfluidsynth` | SoundFont player | `pip install pyfluidsynth` | No |
| `pyo` | Audio synthesis | `pip install pyo` | No |
| `auraloss` | Audio loss functions | `pip install auraloss` | YES |
| `laion-clap` | Audio embeddings | `pip install laion-clap` | Partially |
| `cmaes` / `pycma` | CMA-ES optimizer | `pip install cmaes` | N/A |
| `optuna` | Bayesian optimization | `pip install optuna` | N/A |
| `soundfile` | Audio I/O | `pip install soundfile` | N/A |
| `torchaudio` | Audio I/O + transforms (PyTorch) | `pip install torchaudio` | YES |

### Install Priority Order (run these first at the hackathon)

```bash
# Core stack (all PyTorch, no conflicts)
pip install torchsynth auraloss torchcrepe cmaes demucs torchaudio

# Audio I/O
pip install soundfile librosa

# CLAP embeddings (WARNING: pins numpy==1.23.5, may conflict — test separately)
pip install laion-clap

# FluidSynth (needs system dependency)
pip install pyfluidsynth
brew install fluid-synth  # macOS

# Optional: pyo for real-time DSP
pip install pyo

# DDSP — install LAST (TensorFlow may conflict with PyTorch stack)
pip install ddsp
```

### Known Dependency Conflicts

| Conflict | Issue | Mitigation |
|----------|-------|------------|
| DDSP (TF) + Demucs (PyTorch) | TF and PyTorch in same process can fight over GPU | Use subprocess for one, or pick all-PyTorch stack |
| laion-clap + modern numpy | CLAP pins `numpy==1.23.5` | Install in separate venv, or use transformers CLAP instead |
| crepe (TF) + torchsynth (PyTorch) | Same TF/PyTorch conflict | Use `torchcrepe` instead of `crepe` |

### Recommended Stacks

**Highest-confidence (no GPU needed, works on anything):**
```
cmaes + pyFluidSynth (or pyo) + auraloss → works in 60 seconds, any sound
```

**Highest-ceiling (GPU, differentiable, fast convergence):**
```
torchsynth + auraloss MRSTFT + torchcrepe + Adam → fully differentiable PyTorch
```

### Key Papers

1. Engel et al., "DDSP: Differentiable Digital Signal Processing", ICLR 2020
2. Masuda & Saito, "Synthesizer Sound Matching with Differentiable DSP", 2023
3. Yee-King et al., "Automatic Programming of VST Sound Synthesizers Using Deep Networks and Other Techniques", IEEE TASLP 2018
4. Barkan et al., "InverSynth: Deep Estimation of Synthesizer Parameter Configurations", IEEE/ACM TASLP 2019
5. Roth et al., "CMA-ES for Hyperparameter Optimization", 2020
6. Défossez et al., "Hybrid Transformers for Music Source Separation" (Demucs), 2021
7. Kim et al., "CREPE: A Convolutional Representation for Pitch Estimation", ICASSP 2018
