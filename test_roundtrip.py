"""Round-trip test: generate synth sound with KNOWN params, optimize to recover them."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import numpy as np
import soundfile as sf
from synth import SynthPatch, PARAM_DEFS, _denorm
from losses import MatchingLoss, MelSTFTLoss, CentroidLoss, MFCCLoss
from spectral_init import spectral_init
import cma
import time

# Known ground truth params (normalized 0-1)
# A bright saw+square synth, open filter, some reverb
GROUND_TRUTH = [
    0.80,  # saw_mix
    0.60,  # square_mix
    0.05,  # sine_mix
    0.00,  # noise_mix
    0.50,  # detune (0 semitones)
    0.70,  # filter_cutoff → 200 + 0.7*15800 = 11260Hz
    0.10,  # filter_resonance
    0.01,  # attack (fast)
    0.30,  # decay
    0.70,  # sustain
    0.40,  # release
    0.80,  # gain
    0.55,  # filter_env (slight positive)
    0.30,  # reverb_size
    0.10,  # reverb_mix
    0.30,  # unison_voices (~2.8)
    0.10,  # unison_spread
    0.00,  # noise_floor
    0.10,  # filter_attack
    0.30,  # filter_decay
    0.60,  # filter_sustain
    0.30,  # filter_release
    0.50,  # pulse_width
    0.25,  # filter_slope
    0.50,  # eq1_freq (mid)
    0.50,  # eq1_gain (0dB)
    0.50,  # eq2_freq (mid)
    0.50,  # eq2_gain (0dB)
]

synth = SynthPatch()
gt_params = torch.tensor(GROUND_TRUTH, dtype=torch.float32)

# Print ground truth denormalized
print("=== Ground Truth Parameters ===")
gt_denorm = _denorm(gt_params)
for name, lo, hi in PARAM_DEFS:
    print(f"  {name:>20}: {gt_denorm[name].item():>10.3f}")

# Render target notes at 3 different pitches (like v24)
freqs = [220.0, 277.2, 293.7]  # A3, C#4, D4
names = ["A3", "C#4", "D4"]
target_notes = []

print("\n=== Generating target notes ===")
for freq, name in zip(freqs, names):
    audio = synth.render(gt_params, f0_hz=freq, duration=0.5, note_duration=0.45)
    audio_np = audio.detach().numpy().squeeze()
    target_notes.append((audio_np, freq, 0.5))
    sf.write(f"/tmp/roundtrip_target_{name}.wav", audio_np, 44100)
    print(f"  {name} @ {freq}Hz: {len(audio_np)} samples, {len(audio_np)/44100:.3f}s")

# Spectral init from first note
print("\n=== Spectral Init ===")
init_15 = spectral_init(target_notes[0][0], 44100)
x0 = torch.full((len(PARAM_DEFS),), 0.5, dtype=torch.float32)
x0[0] = init_15[0]
x0[1] = init_15[1]
x0[2] = init_15[2]
x0[3] = init_15[3]
x0[4] = init_15[4]
x0[5] = init_15[5]
x0[6] = init_15[6]
x0[7] = init_15[7]
x0[8] = init_15[8]
x0[9] = init_15[9]
x0[10] = init_15[10]
x0[11] = init_15[11]
x0[12] = init_15[12]
x0[13] = init_15[13]
x0[14] = init_15[14]
x0 = x0.numpy()

# Print spectral init vs ground truth
print(f"  {'Param':>20}  {'GT':>8}  {'Init':>8}  {'Delta':>8}")
for i, (name, lo, hi) in enumerate(PARAM_DEFS):
    if i < 15:
        print(f"  {name:>20}  {GROUND_TRUTH[i]:>8.3f}  {x0[i]:>8.3f}  {abs(GROUND_TRUTH[i]-x0[i]):>8.3f}")

# Run CMA-ES
print(f"\n=== CMA-ES Optimization (10k evals, sigma=0.15) ===")
loss_fn = MatchingLoss()

def evaluate(params_np):
    params = torch.tensor(params_np, dtype=torch.float32)
    total = 0.0
    for audio_np, freq, dur in target_notes:
        target = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        gen = synth.render(params, f0_hz=freq, duration=dur, note_duration=dur * 0.9)
        ml = min(target.shape[1], gen.shape[1])
        loss = loss_fn(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
        total += loss.item()
    return total / len(target_notes)

t0 = time.time()
es = cma.CMAEvolutionStrategy(x0, 0.15, {
    "bounds": [[0] * len(PARAM_DEFS), [1] * len(PARAM_DEFS)],
    "maxfevals": 10000,
    "popsize": 20,
    "verbose": -9,
})

best_loss = float("inf")
while not es.stop():
    sols = es.ask()
    fits = [evaluate(s) for s in sols]
    es.tell(sols, fits)
    if min(fits) < best_loss:
        best_loss = min(fits)
    if es.result.evaluations % 2000 < 25:
        print(f"  Evals: {es.result.evaluations}, Loss: {best_loss:.4f}")

result_params = es.result.xbest
opt_time = time.time() - t0
print(f"  Final loss: {best_loss:.6f}")
print(f"  Time: {opt_time:.1f}s")

# Compare recovered vs ground truth
print(f"\n=== Parameter Recovery ===")
print(f"  {'Param':>20}  {'GT':>8}  {'Recovered':>8}  {'Delta':>8}  {'Status'}")
total_delta = 0
close_count = 0
for i, (name, lo, hi) in enumerate(PARAM_DEFS):
    gt_val = GROUND_TRUTH[i]
    rec_val = float(result_params[i])
    delta = abs(gt_val - rec_val)
    total_delta += delta
    status = "OK" if delta < 0.1 else "CLOSE" if delta < 0.2 else "MISS"
    if delta < 0.1: close_count += 1
    print(f"  {name:>20}  {gt_val:>8.3f}  {rec_val:>8.3f}  {delta:>8.3f}  {status}")

print(f"\n  Params within 0.1: {close_count}/{len(PARAM_DEFS)}")
print(f"  Mean absolute error: {total_delta/len(PARAM_DEFS):.4f}")

# Audio comparison
result_t = torch.tensor(result_params, dtype=torch.float32)
gen_audio = synth.render(result_t, f0_hz=220.0, duration=0.5, note_duration=0.45)
gen_np = gen_audio.detach().numpy().squeeze()
target_np = target_notes[0][0]

from scipy import signal as sig
def spectral_centroid_hz(audio, sr):
    freqs, psd = sig.welch(audio.astype(np.float32), sr, nperseg=2048)
    return float(np.sum(freqs * psd) / (np.sum(psd) + 1e-8))

ml = min(len(target_np), len(gen_np))
corr = float(np.corrcoef(target_np[:ml], gen_np[:ml])[0, 1])
tc = spectral_centroid_hz(target_np, 44100)
gc = spectral_centroid_hz(gen_np, 44100)

print(f"\n=== Audio Quality ===")
print(f"  Waveform correlation: {corr:.4f}")
print(f"  Spectral centroid — Target: {tc:.0f}Hz, Recovered: {gc:.0f}Hz (delta: {abs(tc-gc):.0f}Hz)")

sf.write("/tmp/roundtrip_recovered_A3.wav", gen_np, 44100)
print(f"\n  Saved: /tmp/roundtrip_target_A3.wav, /tmp/roundtrip_recovered_A3.wav")
