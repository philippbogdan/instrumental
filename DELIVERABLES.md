# Inverse Synthesis: Deliverables & Success Criteria

> **Stack**: Combo A — DDSP + all loss functions + gradient descent
> **Approach**: Climb the simplification ladder Level 0 → Level 7
> **Rule**: Each level MUST pass its Definition of Done before moving to the next

---

## Level 0: Closed-Loop Sanity Check

**What**: Generate a target sound from KNOWN synth parameters. Run the optimizer starting from DIFFERENT parameters. Verify it recovers the original parameters.

**Input**: Known DDSP parameters (ground truth)
**Output**: Recovered DDSP parameters (optimizer output)

### Definition of Done

- [ ] DDSP renders a 1-second tone at 440 Hz (A4) from a fixed set of parameters
- [ ] The same DDSP renders a 1-second tone from a RANDOM initial parameter set
- [ ] Multi-Resolution STFT loss between the two is computed and is > 0.5 (they sound different)
- [ ] Optimizer runs for ≤ 500 steps
- [ ] Final STFT loss < 0.01 (sounds identical to human ear)
- [ ] Parameter recovery error: L2 distance between recovered and ground truth params < 0.05 (normalized 0-1 range)
- [ ] Wall-clock time: < 60 seconds on M2 MacBook
- [ ] Reproducible: running the same script 5 times, all 5 converge to loss < 0.01

### What this proves
The optimization loop works. The synth is differentiable. The loss function is meaningful. Gradients flow correctly.

---

## Level 1: Unknown Parameters, Known Pitch, Multiple Losses

**What**: Same as Level 0, but now test ALL loss functions head-to-head. The ground truth parameters are hidden from the optimizer — it only sees the rendered audio.

**Input**: Target audio (rendered from hidden ground truth params at 440 Hz)
**Output**: Recovered parameters + loss comparison table

### Definition of Done

- [ ] All 5 loss functions implemented and runnable:
  - Multi-Resolution STFT (auraloss)
  - MFCC distance (librosa/torchaudio)
  - Raw waveform L1
  - Spectral centroid + flatness + envelope distance
  - Hybrid (STFT + MFCC weighted sum)
- [ ] Each loss function tested on the SAME target sound with the SAME initial random params
- [ ] Results table produced with columns: loss type, final loss value, parameter recovery L2 error, wall-clock time, number of steps to converge (loss < 0.01 equivalent)
- [ ] Run on 3 different target sounds (different timbres: bright, dark, noisy)
- [ ] H3 tested: STFT loss achieves parameter recovery L2 < 0.05 on ≥ 2 of 3 targets
- [ ] H3 tested: waveform L1 achieves parameter recovery L2 < 0.05 on ≤ 1 of 3 targets (confirming it's worse)
- [ ] Best-performing loss identified with justification

### What this proves
Which loss function to use for the rest of the project. H3 validated or refuted.

---

## Level 2: Pitch Detection Added

**What**: Target audio is rendered at an UNKNOWN pitch. The system must detect the pitch first, then optimize synth params.

**Input**: Target audio at unknown pitch (e.g., 329.6 Hz = E4)
**Output**: Detected pitch (Hz) + recovered parameters

### Definition of Done

- [ ] Pitch detection integrated (torchcrepe or librosa pyin)
- [ ] Tested on 5 target sounds at 5 different pitches: C3 (130.8 Hz), E4 (329.6 Hz), A4 (440 Hz), C5 (523.3 Hz), G5 (784 Hz)
- [ ] Pitch detection error < 2 Hz on all 5 targets (with clean synthetic audio)
- [ ] After pitch detection, optimizer converges to STFT loss < 0.01 on ≥ 4 of 5 targets
- [ ] Parameter recovery L2 < 0.05 on ≥ 4 of 5 targets
- [ ] End-to-end time (pitch detection + optimization): < 90 seconds per target on M2

### What this proves
Pitch detection doesn't break the pipeline. The system works across the musical range.

---

## Level 3: Source Separation Added

**What**: Target audio is an instrument ISOLATED FROM A REAL SONG using Demucs. The system must detect pitch and optimize params to match.

**Input**: A real song (.wav or .mp3)
**Output**: Separated stem + detected pitch + recovered synth parameters + rendered match audio

### Definition of Done

- [ ] Demucs integrated: accepts a song file, outputs 4 stems (vocals, drums, bass, other)
- [ ] User can select which stem to match
- [ ] Pipeline runs end-to-end: song → Demucs → select stem → crop 1-second segment → detect pitch → optimize → output matched audio
- [ ] Tested on 3 real songs with clearly identifiable instruments
- [ ] STFT loss between matched audio and target stem < 0.1 on ≥ 2 of 3 songs (relaxed from 0.01 because real audio has noise/artifacts the synth can't reproduce)
- [ ] A/B listening test: matched audio is recognizably "the same kind of sound" as the target (subjective, but honest self-assessment)
- [ ] Output: both target stem audio and matched synth audio saved as .wav files for comparison
- [ ] End-to-end time: < 5 minutes per song (including Demucs separation)

### What this proves
The system works on real-world audio, not just synthetic targets. The synthetic-to-real gap is manageable.

---

## Level 4: Multi-Pitch Fitting (H1)

**What**: Optimize synth parameters to match the target across MULTIPLE pitches simultaneously, not just one note. Test whether this produces a more "transferable" patch.

**Input**: Target instrument playing ≥ 3 different notes (from separated stem or synthetic)
**Output**: Single set of synth parameters that works across all pitches

### Definition of Done

- [ ] System can extract multiple 1-second segments from a stem at different detected pitches
- [ ] Optimizer minimizes the SUM of losses across all segments simultaneously
- [ ] Two conditions compared:
  - **Single-pitch fit**: optimize on 1 note (e.g., C4), test on 2 unseen notes (E4, G4)
  - **Multi-pitch fit**: optimize on 3 notes (C4, E4, G4), test on 2 unseen notes (A4, D5)
- [ ] Metric: average STFT loss on unseen notes
- [ ] H1 tested: multi-pitch fit achieves ≥ 20% lower loss on unseen notes than single-pitch fit
- [ ] Results table: single-pitch vs multi-pitch, loss on training notes, loss on unseen notes
- [ ] Tested with ≥ 2 different target timbres

### What this proves
H1 validated or refuted. Multi-pitch fitting is worth the extra compute (or not).

---

## Level 5: Effects Chain Added

**What**: Add post-processing effects (reverb, EQ) to the synth and optimize those parameters too.

**Input**: Target audio with audible reverb/room sound
**Output**: Synth parameters + effect parameters

### Definition of Done

- [ ] Reverb module added to DDSP chain (differentiable)
- [ ] EQ/filter module added to DDSP chain (differentiable)
- [ ] Total parameter count documented (synth params + effect params)
- [ ] Optimizer converges on synthetic targets WITH effects: STFT loss < 0.02
- [ ] Tested on real separated stems: STFT loss < 0.15 (relaxed — effects add ambiguity)
- [ ] Compared: with-effects matching vs without-effects matching on same reverb-heavy target
- [ ] With-effects achieves ≥ 15% lower STFT loss than without-effects on reverb-heavy targets
- [ ] H2 tested (optional): staged optimization (synth first, then effects) vs joint optimization
- [ ] If H2 tested: report which approach achieves lower final loss and faster convergence

### What this proves
The system can handle real-world audio that has room characteristics baked in.

---

## Level 6: Multi-Synth Comparison

**What**: Run the optimizer against MULTIPLE synth configurations (different oscillator types, different architectures). Return a ranked list of "which synth type best matches this sound?"

**Input**: Target audio
**Output**: Ranked list of synth configurations with their best-fit loss

### Definition of Done

- [ ] ≥ 3 distinct synth configurations defined (e.g., additive-only, additive+noise, additive+noise+reverb, or different harmonic distributions)
- [ ] Optimizer runs independently on each configuration
- [ ] Results ranked by final STFT loss
- [ ] Tested on ≥ 3 target sounds with known "correct" synth types
- [ ] Correct synth type ranked #1 on ≥ 2 of 3 targets
- [ ] Output: ranked table with synth type, final loss, rendered audio file
- [ ] Total time for full comparison: < 10 minutes for 3 synths × 1 target

### What this proves
The system can identify WHICH type of synthesis best fits a sound, not just optimize within one type.

---

## Level 7: Full Pipeline Demo

**What**: End-to-end system. User provides a song. System separates, detects, matches, and outputs playable synth patches for each stem.

**Input**: Any song (.wav or .mp3)
**Output**: For each stem — separated audio, detected pitch(es), best synth configuration, rendered match audio, parameter file

### Definition of Done

- [ ] Single entry point: `python match.py song.mp3`
- [ ] Outputs directory structure:
  ```
  output/
    song/
      vocals/
        separated.wav
        matched.wav
        params.json
        pitch.txt
        loss.txt
      drums/
        ...
      bass/
        ...
      other/
        ...
  ```
- [ ] params.json contains all synth + effect parameters in human-readable format
- [ ] loss.txt contains final STFT loss value
- [ ] pitch.txt contains detected fundamental frequency
- [ ] Tested on ≥ 2 complete songs
- [ ] ≥ 2 of 4 stems per song achieve STFT loss < 0.15
- [ ] Total end-to-end time: < 15 minutes per song on M2 MacBook
- [ ] No manual intervention required after initial command
- [ ] Output .wav files are playable and not silent/corrupted

### What this proves
The full vision works end-to-end. This is the hackathon demo.

---

## Summary Table

| Level | Name | Key Metric | Target | Depends On |
|-------|------|-----------|--------|------------|
| 0 | Closed-loop sanity | Parameter recovery L2 | < 0.05 | Nothing |
| 1 | Loss comparison | Best loss identified | STFT < 0.01 on ≥ 2/3 | Level 0 |
| 2 | Pitch detection | Pitch error | < 2 Hz on 5/5 | Level 1 |
| 3 | Source separation | Real-audio STFT loss | < 0.1 on ≥ 2/3 | Level 2 |
| 4 | Multi-pitch (H1) | Unseen note loss improvement | ≥ 20% better | Level 2 |
| 5 | Effects chain | Reverb-target STFT loss | < 0.15 | Level 3 |
| 6 | Multi-synth | Correct synth ranked #1 | ≥ 2/3 targets | Level 3 |
| 7 | Full pipeline | End-to-end demo | < 15 min, ≥ 2/4 stems < 0.15 | Level 3, 4, 5, 6 |

### Dependency Graph

```
Level 0 → Level 1 → Level 2 → Level 3 → Level 5 → Level 7
                         │                           ↑
                         ├──→ Level 4 ────────────────┤
                         │                            │
                         └──→ Level 3 → Level 6 ──────┘
```

Levels 4, 5, 6 can run in parallel after Level 3 is done. Level 7 integrates everything.
Level 8 branches into two sub-tracks (8A and 8B) that can run in parallel, both requiring GPU training.

---

## Level 8: Learned Sound Matching (GPU training required)

**What**: Replace the per-sound optimization loop with a trained model that predicts synth parameters in one forward pass. Two sub-tracks exploring different paradigms.

### Level 8A: Learned Encoder (supervised regression)

**What**: Train a neural network that maps audio → synth parameters directly. One forward pass, instant prediction, no optimization loop.

**Approach**:
1. Generate synthetic dataset: 100K-500K random synth patches → render audio → extract features
2. Train a model (random forest baseline, then CNN on spectrograms) to predict params from audio
3. At inference: target audio → encoder → predicted params → optional Adam fine-tune (50 steps)

**Input**: Target audio (spectrogram or extracted features)
**Output**: Predicted synth parameters in one forward pass (~5ms)

#### Definition of Done

- [ ] Dataset generated: ≥ 100K (params, audio) pairs from our 15-param synth
- [ ] Random forest baseline trained and evaluated: feature vector → params regression
  - Baseline metric: median parameter L2 error < 0.2 on held-out test set
- [ ] CNN encoder trained on mel spectrograms → 15 params
  - CNN metric: median parameter L2 error < 0.1 on held-out test set
- [ ] Encoder predictions used as initialization for Adam fine-tuning (50 steps)
  - End-to-end STFT loss < 0.5 on ≥ 80% of test targets
- [ ] Tested on real audio (your lead synth): STFT loss < best CMA-ES result
- [ ] Single-sound inference time: < 100ms (encoder) + < 5s (optional fine-tune)
- [ ] Comparison table: random init vs spectral init vs encoder init → final loss after fine-tune

#### What this proves
A trained model can replace expensive optimization with instant prediction, and the fine-tuning step closes the gap.

---

### Level 8B: Reinforcement Learning Agent (SynthRL-style)

**What**: Train an RL agent that learns a general STRATEGY for matching sounds — not memorizing a mapping, but learning how to turn knobs. The agent observes the current sound vs target and decides which parameters to adjust.

**Approach**:
1. Build a Gymnasium environment wrapping our synth
2. State = [current 15 params] + [target audio features] + [current audio features] = ~50-dim vector
3. Action = continuous adjustment to each of 15 params (Box space, [-0.1, +0.1])
4. Reward = loss_before - loss_after (did the sound get closer?)
5. Train PPO agent via stable-baselines3 for 500K-2M timesteps
6. At inference: agent plays 50-100 steps of knob-turning → converges to match

**Input**: Target audio + initial random synth state
**Output**: Sequence of parameter adjustments → final matched params

#### Definition of Done

- [ ] Gymnasium environment `SynthMatchEnv` implemented and passes `check_env()`
- [ ] State space: current params (15) + target features (17) + current features (17) = 49-dim
- [ ] Action space: 15-dim continuous Box([-0.1, +0.1])
- [ ] Reward: negative STFT loss delta (improvement = positive reward)
- [ ] PPO agent trained for ≥ 500K timesteps
- [ ] Training reward curve shows clear upward trend (agent is learning)
- [ ] Evaluation: agent matches ≥ 70% of held-out synthetic targets to STFT loss < 1.0 within 100 steps
- [ ] Evaluation on real audio: agent achieves STFT loss < best CMA-ES result on your lead synth
- [ ] Comparison: RL agent vs CMA-ES vs gradient descent — steps to convergence, final loss, wall-clock time
- [ ] Agent can match a sound it has NEVER seen during training (generalization test)

#### What this proves
An RL agent can learn a transferable skill for sound matching that generalizes beyond training data, including to real-world recordings.

---

### Level 8 Summary

| Sub-track | Paradigm | Training time | Inference time | Handles new sounds |
|-----------|----------|---------------|----------------|-------------------|
| 8A Encoder | Supervised regression | 1-4 hours | ~5ms + optional 5s fine-tune | Only similar to training data |
| 8B RL Agent | Reinforcement learning | 4-12 hours | ~1-5s (50-100 steps) | Yes — learns a general strategy |

**Hardware**: Mac Mini (M4, 16GB) for training. GPU (Azure/AWS) if CPU too slow.
**Dependencies**: `pip install stable-baselines3 gymnasium` (for 8B), scikit-learn (for 8A baseline)

Levels 4, 5, 6 can run in parallel after Level 3 is done. Level 7 integrates everything.
Level 8 branches into two sub-tracks (8A and 8B) that can run in parallel, both requiring GPU training.

---

## Hackathon Time Budget (rough)

| Level | Estimated Time | Cumulative |
|-------|---------------|-----------|
| 0 | 1-2 hours | 2 hours |
| 1 | 1-2 hours | 4 hours |
| 2 | 1 hour | 5 hours |
| 3 | 2-3 hours | 8 hours |
| 4 | 2 hours | 10 hours (parallel with 5, 6) |
| 5 | 2 hours | 10 hours (parallel with 4, 6) |
| 6 | 2 hours | 10 hours (parallel with 4, 5) |
| 7 | 3-4 hours | 14 hours |

**Total**: ~14 hours of work. Fits in a hackathon if Levels 4/5/6 run in parallel (via swarm).
