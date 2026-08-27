"""Benchmark multiprocessing efficiency on Mini."""
import torch, time, numpy as np
from multiprocessing import Pool
from src.synth import SynthPatch, PARAM_DEFS
from src.losses import get_loss
import soundfile as sf

# Load target notes
audio, sr = sf.read('target.wav')
audio = audio[:5*44100].astype(np.float32)

from app.server import _extract_notes
notes = _extract_notes(audio, sr)
print(f'Notes: {len(notes)} at {[f"{n[1]:.0f}Hz" for n in notes]}')

TARGET_NOTES = [(n[0], n[1], n[2]) for n in notes]

def eval_single(params_np):
    synth = SynthPatch()
    loss_fn = get_loss("matching")
    params = torch.tensor(params_np, dtype=torch.float32)
    total = 0.0
    for audio_np, freq, dur in TARGET_NOTES:
        target = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        gen = synth.render(params, f0_hz=freq, duration=dur, note_duration=dur * 0.9)
        ml = min(target.shape[1], gen.shape[1])
        loss = loss_fn(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
        total += loss.item()
    return total / len(TARGET_NOTES)

# Single worker
t0 = time.time()
for _ in range(20):
    eval_single(np.random.rand(28))
single = time.time() - t0
rate1 = 20 / single
print(f'Single worker: {rate1:.0f} evals/s')

# 9-worker pool
solutions = [np.random.rand(28) for _ in range(200)]
t0 = time.time()
with Pool(9) as pool:
    results = pool.map(eval_single, solutions)
multi = time.time() - t0
rate9 = 200 / multi
print(f'Pool(9): {rate9:.0f} evals/s')
print(f'Theoretical: {rate1 * 9:.0f} evals/s')
print(f'Efficiency: {rate9 / (rate1 * 9) * 100:.0f}%')
print(f'20k evals at Pool(9): {20000/rate9:.0f}s')
