# INSTRUMENTAL

**Try it: [philippbogdan.com/instrumental](https://philippbogdan.com/instrumental)**
&nbsp;·&nbsp; Paper: [arXiv:2603.15905](https://arxiv.org/abs/2603.15905)

Search for a song or drop in a synth recording, and watch the patch being
searched for. The matching runs on one Mac mini behind a Cloudflare tunnel, so
when that machine is off the page says so and the pre-matched demos still play.
Deployment notes, and what it costs to run, are in [deploy/](deploy/).

---
**Automatic synthesizer parameter recovery from audio.**

Given a sound from a song, INSTRUMENTAL recovers the synth settings that recreate it. Not just the notes, but the actual instrument.

> Every DAW can tell you *what notes are playing*. We tackle the harder problem: *what synth patch made that sound*.

## Listen

**Original** (lead synth from a real song): [ui/original.mp3](ui/original.mp3)

**Our match** (28 recovered parameters): [ui/matched.mp3](ui/matched.mp3)

## Demo

**Input:** 4.8s lead synth clip from a song (22 notes, 3 pitches)

**Output:** 28 synthesizer parameters that reproduce the sound

| | Loss | Centroid delta | Notes |
|---|---|---|---|
| Silence | 1564.6 | | Worst possible |
| Random synth | 52.8 | | Noise |
| **Our best match** | **2.09** | **133 Hz** | **Top 0.13% of loss range** |

## How It Works

```
Audio file
  Demucs (source separation)
  TorchCrepe (pitch detection)
  22 notes extracted at A3, C#4, D4
  CMA-ES optimization (28-param differentiable synth x 100K evals)
  Best match synth parameters
  Rendered output audio
```

**The synth:** 28 parameter differentiable subtractive synthesizer in PyTorch:
- 4 oscillator types (saw, pulse, sine, noise) with 5 voice unison
- Sigmoid low pass filter with learnable slope + 2 band parametric EQ
- Dual ADSR envelopes (amplitude + filter)
- Delay line reverb

**The optimizer:** CMA-ES with spectral initialization. We analyze the target spectrum to set smart starting parameters instead of random search.

**The loss:** Mel scaled multi res STFT + spectral centroid + MFCC (captures brightness, timbre, and spectral shape).

## Key Findings

### 1. CMA-ES beats gradient descent on this problem
Adam gets trapped in local minima (loss 4.60). CMA-ES escapes them (loss 2.09). A PPO RL agent matches CMA-ES but does not beat it after 1M training steps.

### 2. More parameters does not mean better
Expanding from 24 to 29 params (adding unconstrained distortion, delay, vibrato) made results **worse**. CMA-ES exploited extreme values (+24 semitone detune, 43% reverb) to "cheat" the loss. Constrained, meaningful parameters outperform unconstrained expansion.

### 3. Parametric EQ was the breakthrough
A low pass filter can only *remove* frequencies. The target sound had mid high presence (1 to 5 kHz) that required *boosting*. Adding a 2 band EQ, the only modification out of 8 tested that improved results, found a +1 dB peak at 5.8 kHz.

### 4. 90% of convergence happens in 10K evals
Out of 100K evaluations, the first 10K achieve 90% of the total improvement. The remaining 90K yield only 0.07 loss reduction, hitting an architectural floor.

### 5. The harmonic gap reveals synthesis limits
The target has H3 > H2 (3rd harmonic stronger than 2nd), which is impossible with sawtooth or square waves. This identifies the floor of subtractive synthesis and points to FM/wavetable as the next step.

## Hypothesis Testing

We systematically tested 8 modifications (10K evals each):

| Modification | Loss change | Verdict |
|---|---|---|
| Chebyshev waveshaping | +29% | no |
| **Parametric EQ** | **-8%** | **yes** |
| TFS / envelope loss | +17% | no |
| SPSA fine tuning | +11% | no |
| Multi start CMA-ES (8x) | +29% | no |
| FM synthesis | +17% | no |
| CDPAM perceptual loss | skipped | |
| CLAP embedding loss | converged | different quality |

Only one thing worked. Everything else made it worse.

## Performance

| Configuration | Evals/s | 100K evals |
|---|---|---|
| CPU sequential | 102/s | 16 min |
| CPU multiprocessing (10 cores) | 229/s | 7 min |
| Batched tensor ops (no loops) | 553/s | 3 min |
| Batched + GPU synth (MPS) | 1,170/s | 1.4 min |

**Hardware tested:** Apple M2 (8GB), Apple M4 Mini (16GB), Azure D48s v5 (48 vCPUs). M4 Mini was fastest. Apple Silicon per core performance dominates Azure EPYC even at 5x fewer cores.

## Project Structure

```
instrumental/
├── src/                # the synth, the losses, the CMA-ES pipeline
│   ├── synth.py            # 28 param differentiable synthesizer
│   ├── synth_gpu.py        # batched GPU variant
│   ├── losses.py           # 8 loss functions (STFT, mel, MFCC, centroid, ...)
│   ├── spectral_init.py    # spectral analysis, smart init params
│   └── cmaes_search.py     # CMA-ES wrapper
├── server/             # backend for the live demo at /instrumental
├── deploy/             # Mac mini + Cloudflare tunnel deployment notes
├── experiments/        # every runner from the build, kept as run
│                       # (run_hypotheses.py is the 8-test harness)
├── paper/              # LaTeX source; PDF in output/
├── ui/                 # dashboard + the original/matched demo audio
├── research/           # background notes: synthesis, the Vital format
├── output/             # matched audio, paper PDF, RL agent, best preset
├── DIRECTIONS.md       # the plan as written on the hackathon night
└── DELIVERABLES.md     # the Level 0 to 8 success criteria
```

## Quick Start

```bash
pip install torch torchaudio torchsynth auraloss soundfile librosa cma

# Play the comparison
python experiments/play_notes.py

# Run CMA-ES optimization on a target sound
python experiments/run_cmaes_v2.py

# Run all 8 hypothesis tests
python experiments/run_hypotheses.py

# View the UI
cd ui && python -m http.server 8080
```

## Research Paper

4 page paper with equations, figures, and 10 references: [output/paper.pdf](output/paper.pdf)

Built at the [Mozart AI Hack](https://lu.ma/mozartai), London, March 14 to 15 2026.

*Built with Claude Code.*
