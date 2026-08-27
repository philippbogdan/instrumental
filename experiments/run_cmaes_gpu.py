"""CMA-ES with batched GPU synth — 100K evals, CPU batched."""

import torch
import numpy as np
import soundfile as sf
import subprocess
import time
import librosa
import cma
import torchaudio
from src.synth_gpu import SynthPatchGPU, PARAM_DEFS, N_PARAMS

# Load targets
TARGET_NOTES = []
for f, freq in [('notes/note_03_A3_221Hz.wav', 221.0), ('notes/note_02_Cs4_278Hz.wav', 278.0), ('notes/note_01_D4_295Hz.wav', 295.0)]:
    audio, sr = sf.read(f)
    TARGET_NOTES.append((torch.tensor(audio, dtype=torch.float32), freq, len(audio) / 44100))

synth = SynthPatchGPU()

# Batched matching loss
_mfcc_transform = torchaudio.transforms.MFCC(sample_rate=44100, n_mfcc=13,
    melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 40})

def batched_loss(gen_batch, target):
    B, N = gen_batch.shape
    tgt_expanded = target.unsqueeze(0).expand(B, -1)
    total = torch.zeros(B)
    for n_fft, hop in [(1024, 256), (2048, 512)]:
        if N < n_fft: continue
        window = torch.hann_window(n_fft)
        sg = torch.stft(gen_batch, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        st = torch.stft(tgt_expanded, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        mg, mt = sg.abs(), st.abs()
        mel_fb = torchaudio.functional.melscale_fbanks(n_freqs=n_fft//2+1, f_min=0, f_max=22050, n_mels=128, sample_rate=44100)
        mg_mel = torch.einsum('bft,fm->bmt', mg, mel_fb)
        mt_mel = torch.einsum('bft,fm->bmt', mt, mel_fb)
        sc = torch.norm(mt_mel - mg_mel, dim=(1,2)) / (torch.norm(mt_mel, dim=(1,2)) + 1e-8)
        lm = torch.mean(torch.abs(torch.log(mg_mel.clamp(min=1e-8)) - torch.log(mt_mel.clamp(min=1e-8))), dim=(1,2))
        total += sc + lm
    # Centroid
    window = torch.hann_window(2048)
    sg = torch.stft(gen_batch, n_fft=2048, hop_length=512, window=window, return_complex=True).abs()
    st = torch.stft(tgt_expanded, n_fft=2048, hop_length=512, window=window, return_complex=True).abs()
    freqs = torch.fft.rfftfreq(2048, d=1/44100).view(1,-1,1)
    cent_g = (freqs * sg).sum(dim=1) / (sg.sum(dim=1) + 1e-8)
    cent_t = (freqs * st).sum(dim=1) / (st.sum(dim=1) + 1e-8)
    total += 0.1 * torch.mean(torch.abs(cent_g - cent_t), dim=1) / 22050
    # MFCC
    mfcc_g = _mfcc_transform(gen_batch)
    mfcc_t = _mfcc_transform(tgt_expanded)
    total += 0.05 * torch.mean((mfcc_g - mfcc_t)**2, dim=(1,2))
    return total

# v11 init mapped to 24 params
v11 = [0.7329, 0.9156, 0.0029, 0.0127, 0.5, 0.9937, 0.2039, 0.0002, 0.1137, 0.8624,
       0.9048, 0.1422, 0.2972, 0.5472, 0.0247, 0.7223, 0.0012, 0.4182,
       0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # filter ADSR + pulse_width + filter_slope at center

BUDGET = 100000
POPSIZE = 80

if __name__ == "__main__":
    print(f"Params: {N_PARAMS}, Budget: {BUDGET}, Pop: {POPSIZE}", flush=True)
    print(f"Starting from v11 init, sigma=0.15", flush=True)

    es = cma.CMAEvolutionStrategy(v11, 0.15, {
        'bounds': [[0]*N_PARAMS, [1]*N_PARAMS],
        'maxfevals': BUDGET, 'popsize': POPSIZE, 'verbose': -9,
        'tolx': 1e-8, 'tolfun': 1e-10, 'tolstagnation': 20000,
    })

    best_loss = float('inf')
    evals = 0
    t0 = time.time()

    while not es.stop():
        solutions = es.ask()
        params_batch = torch.tensor(np.array(solutions), dtype=torch.float32)

        # Fully batched render + loss
        total_losses = torch.zeros(POPSIZE)
        for target_audio, freq, dur in TARGET_NOTES:
            gen_batch = synth.render(params_batch, f0_hz=freq, duration=dur, note_duration=dur * 0.9)
            ml = min(target_audio.shape[0], gen_batch.shape[1])
            losses = batched_loss(gen_batch[:, :ml], target_audio[:ml])
            total_losses += losses.detach()

        fitnesses = (total_losses / len(TARGET_NOTES)).tolist()
        es.tell(solutions, fitnesses)
        evals += POPSIZE

        if min(fitnesses) < best_loss:
            best_loss = min(fitnesses)
        if evals % 5000 < POPSIZE:
            elapsed = time.time() - t0
            rate = evals / elapsed
            print(f"  evals {evals:>7}: best={best_loss:.4f} ({rate:.0f}/s, ETA {(BUDGET-evals)/rate:.0f}s)", flush=True)

    elapsed = time.time() - t0
    best_params = torch.tensor(es.result.xbest, dtype=torch.float32)
    print(f"\nDone: loss={best_loss:.4f}, {evals} evals in {elapsed:.1f}s ({evals/elapsed:.0f}/s)", flush=True)

    print("\nParams:")
    for name, val in zip(synth.get_param_names(), best_params):
        idx = synth.get_param_names().index(name)
        lo, hi = PARAM_DEFS[idx][1], PARAM_DEFS[idx][2]
        real_val = val.item() * (hi - lo) + lo
        print(f"  {name:>18}: {val.item():.4f} (={real_val:.3f})")

    np.save('output/best_params_gpu.npy', best_params.numpy())

    # Render full sequence using CPU synth (known good)
    from src.synth import SynthPatch
    cpu_synth = SynthPatch()
    note_data = [
        (0.023,0.148,294.5),(0.171,0.154,278.0),(0.325,0.145,220.6),
        (0.470,0.151,294.5),(0.621,0.145,278.0),(0.778,0.136,220.6),
        (0.914,0.154,292.8),(1.068,0.148,278.0),(1.216,0.148,292.8),
        (1.364,0.145,278.0),(1.509,0.145,292.8),(1.654,0.148,278.0),
        (1.802,0.157,220.6),(1.959,0.145,294.5),(2.104,0.148,278.0),
        (2.252,0.151,220.6),(2.403,0.145,292.8),(2.548,0.151,278.0),
        (2.699,0.145,220.6),(2.844,0.151,294.5),(2.995,0.470,292.8),
    ]
    orig_full, _ = librosa.load('target.wav', sr=44100, mono=True)
    s, e = int(1.312*44100), int(4.5*44100)
    rms_orig = np.sqrt(np.mean(orig_full[s:e]**2))
    full_audio = np.zeros(len(orig_full))
    for onset, dur, freq in note_data:
        audio = cpu_synth.render(best_params, f0_hz=freq, duration=dur+0.3, note_duration=dur)
        audio_np = audio.detach().numpy().squeeze()
        pos = int((1.312+onset)*44100)
        end = pos+len(audio_np)
        if end <= len(full_audio): full_audio[pos:end] += audio_np
        else:
            trim = len(full_audio)-pos
            if trim > 0: full_audio[pos:pos+trim] += audio_np[:trim]
    rms_m = np.sqrt(np.mean(full_audio[s:e]**2))
    full_audio *= rms_orig/(rms_m+1e-8)
    if np.max(np.abs(full_audio)) > 0.99: full_audio = full_audio / np.max(np.abs(full_audio)) * 0.99
    sf.write('output/matched_v23_batched.wav', full_audio.astype(np.float32), 44100)

    print("\n--- ORIGINAL ---", flush=True)
    subprocess.run(["afplay", "target.wav"])
    print("--- V23 (batched CPU, 24 params) ---", flush=True)
    subprocess.run(["afplay", "output/matched_v23_batched.wav"])
