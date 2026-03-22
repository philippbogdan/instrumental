"""Diagnose WHY CMA-ES converges to wrong spectral balance."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import requests, json, time, io
import numpy as np
import soundfile as sf
import torch
from synth import SynthPatch, PARAM_DEFS, _denorm
from losses import MelSTFTLoss, CentroidLoss, MFCCLoss, MatchingLoss

BASE = "http://localhost:8000"

# Run pipeline up to optimization
print("=== Running pipeline ===")
r = requests.get(f"{BASE}/api/search", params={"q": "kraftwerk the model"})
preview_url = r.json()["data"][0]["preview"]
mp3 = requests.get(f"{BASE}/api/preview", params={"url": preview_url}).content

r = requests.post(f"{BASE}/api/separate",
    files={"file": ("preview.mp3", mp3, "audio/mpeg")},
    data={"stem_count": "4"})
sep_id = r.json()["job_id"]

r = requests.post(f"{BASE}/api/extract-notes", data={"stem_job_id": sep_id, "stem_name": "other"})
nd = r.json()
note_job_id = nd["note_job_id"]
print(f"Notes: {len(nd['notes'])} representative, {len(nd['all_notes'])} sequence")
for n in nd["notes"]:
    print(f"  {n['name']} @ {n['freq']}Hz, dur={n['duration']:.3f}s")

# Run optimization
print("\n=== Optimizing (10k evals) ===")
r = requests.post(f"{BASE}/api/match-single", data={
    "note_job_id": note_job_id, "n_evals": "10000", "selected_note": "0"
})
job_id = r.json()["job_id"]
match_data = None
while True:
    time.sleep(3)
    r = requests.get(f"{BASE}/api/job-result/{job_id}")
    d = r.json()
    if d.get("type") == "complete" or d.get("status") == "done":
        match_data = d
        break
    evals = d.get("evals", 0)
    if evals and evals % 3000 < 200:
        print(f"  {evals} evals, loss={d.get('loss', '?')}")

params_normalized = match_data["params"]
final_loss = match_data.get("loss", "?")
print(f"Final loss: {final_loss}")

# Denormalize and print actual synth parameters
print("\n=== Denormalized synth parameters ===")
params_t = torch.tensor(params_normalized, dtype=torch.float32)
denormed = _denorm(params_t)
for name, lo, hi in PARAM_DEFS:
    val = denormed[name].item()
    norm = params_normalized[PARAM_DEFS.index((name, lo, hi))]
    flag = ""
    if name == "filter_cutoff" and val < 500: flag = " *** VERY LOW"
    elif name == "filter_cutoff" and val < 1000: flag = " ** LOW"
    elif name == "filter_cutoff": flag = f" (range: {lo}-{hi})"
    print(f"  {name:>20}: {val:>10.3f}  (norm={norm:.3f}){flag}")

# Load the target note
print("\n=== Per-component loss breakdown ===")
r = requests.get(f"{BASE}/api/note/{note_job_id}/0.wav")
target_audio, sr = sf.read(io.BytesIO(r.content))
if target_audio.ndim > 1:
    target_audio = target_audio.mean(axis=1)
target_t = torch.tensor(target_audio, dtype=torch.float32).unsqueeze(0)

# Render synth with matched params at same freq/dur
synth = SynthPatch()
note_info = nd["notes"][0]
gen = synth.render(params_t, f0_hz=note_info["freq"], duration=note_info["duration"],
                   note_duration=note_info["duration"] * 0.9)
ml = min(target_t.shape[1], gen.shape[1])
target_3d = target_t[:, :ml].unsqueeze(0)
gen_3d = gen[:, :ml].unsqueeze(0)

# Individual loss components
mel_stft = MelSTFTLoss()
centroid = CentroidLoss()
mfcc = MFCCLoss()
matching = MatchingLoss()

mel_val = mel_stft(gen_3d, target_3d).item()
cent_val = centroid(gen_3d, target_3d).item()
mfcc_val = mfcc(gen_3d, target_3d).item()
total_val = matching(gen_3d, target_3d).item()

print(f"  MelSTFT:     {mel_val:.4f}  (weight 1.0 → contributes {1.0*mel_val:.4f})")
print(f"  Centroid:    {cent_val:.4f}  (weight 0.1 → contributes {0.1*cent_val:.4f})")
print(f"  MFCC:        {mfcc_val:.4f}  (weight 0.05 → contributes {0.05*mfcc_val:.4f})")
print(f"  Total:       {total_val:.4f}")

# Spectral comparison of target vs generated
from scipy import signal as sig

target_np = target_audio[:ml]
gen_np = gen.squeeze().detach().numpy()[:ml]
target_np = target_np / (np.max(np.abs(target_np)) + 1e-8)
gen_np = gen_np / (np.max(np.abs(gen_np)) + 1e-8)

def spectral_centroid_hz(audio, sr):
    freqs, psd = sig.welch(audio, sr, nperseg=2048)
    return float(np.sum(freqs * psd) / (np.sum(psd) + 1e-8))

tc = spectral_centroid_hz(target_np.astype(np.float32), sr)
gc = spectral_centroid_hz(gen_np.astype(np.float32), sr)
print(f"\n  Spectral centroid — Target: {tc:.0f}Hz, Generated: {gc:.0f}Hz")

# What if we just set filter_cutoff high?
print("\n=== Experiment: same params but filter_cutoff=8000Hz ===")
params_bright = params_normalized.copy()
# filter_cutoff is param index 5, range 200-16000
# To set to 8000: norm = (8000 - 200) / (16000 - 200) = 0.494
params_bright[5] = 0.494
params_bright_t = torch.tensor(params_bright, dtype=torch.float32)
gen_bright = synth.render(params_bright_t, f0_hz=note_info["freq"], duration=note_info["duration"],
                          note_duration=note_info["duration"] * 0.9)
ml2 = min(target_t.shape[1], gen_bright.shape[1])
gen_bright_3d = gen_bright[:, :ml2].unsqueeze(0)
target_3d2 = target_t[:, :ml2].unsqueeze(0)

bright_loss = matching(gen_bright_3d, target_3d2).item()
bright_mel = mel_stft(gen_bright_3d, target_3d2).item()
bright_cent = centroid(gen_bright_3d, target_3d2).item()
gc_bright = spectral_centroid_hz(gen_bright.squeeze().detach().numpy()[:ml2].astype(np.float32), sr)
print(f"  Loss: {bright_loss:.4f} (was {total_val:.4f})")
print(f"  MelSTFT: {bright_mel:.4f} (was {mel_val:.4f})")
print(f"  Centroid: {bright_cent:.4f} (was {cent_val:.4f})")
print(f"  Spectral centroid: {gc_bright:.0f}Hz (was {gc:.0f}Hz, target {tc:.0f}Hz)")

# Save both for listening
sf.write("/tmp/diag_target.wav", target_np, sr)
sf.write("/tmp/diag_matched.wav", gen_np, sr)
sf.write("/tmp/diag_bright.wav", gen_bright.squeeze().detach().numpy()[:ml2], sr)
print("\n  Saved: /tmp/diag_target.wav, /tmp/diag_matched.wav, /tmp/diag_bright.wav")
