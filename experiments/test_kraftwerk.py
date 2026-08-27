"""Run full INSTRUMENTAL pipeline on Kraftwerk - The Model and analyze output."""
import requests
import time
import json
import numpy as np
import soundfile as sf
import io

BASE = "http://localhost:8000"

# Step 1: Search
print("=== Step 1: Getting preview URL ===")
r = requests.get(f"{BASE}/api/search", params={"q": "kraftwerk the model"})
tracks = r.json().get("data", [])
track = tracks[0]
preview_url = track["preview"]
print(f"  Track: {track['artist']['name']} - {track['title']}")

# Step 2: Download preview
print("\n=== Step 2: Downloading preview MP3 ===")
r = requests.get(f"{BASE}/api/preview", params={"url": preview_url})
mp3_bytes = r.content
print(f"  Downloaded {len(mp3_bytes)} bytes")

# Step 3: Separate
print("\n=== Step 3: Separating stems (htdemucs_ft) ===")
t0 = time.time()
r = requests.post(f"{BASE}/api/separate",
    files={"file": ("preview.mp3", mp3_bytes, "audio/mpeg")},
    data={"stem_count": "4"}
)
sep_data = r.json()
sep_id = sep_data.get("job_id")
print(f"  Stems: {list(sep_data.get('stems', {}).keys())}")
print(f"  Time: {time.time()-t0:.1f}s")

# Step 4: Extract notes
print("\n=== Step 4: Extracting notes from 'other' stem ===")
r = requests.post(f"{BASE}/api/extract-notes", data={
    "stem_job_id": sep_id, "stem_name": "other"
})
note_data = r.json()
note_job_id = note_data.get("note_job_id")
notes = note_data.get("notes", [])
all_notes = note_data.get("all_notes", [])
analyzed_dur = note_data.get("analyzed_duration", 10)
print(f"  Representative notes: {len(notes)}")
print(f"  Sequence notes: {len(all_notes)}")
print(f"  Analyzed duration: {analyzed_dur}s")
for i, n in enumerate(notes):
    print(f"    Rep {i}: {n['name']} @ {n['freq']}Hz, dur={n['duration']:.3f}s, rms={n.get('rms', '?')}")
print()
for i, n in enumerate(all_notes):
    ntype = "SUSTAIN" if n['duration'] > 0.4 else ""
    print(f"    Seq {i:2d}: onset={n['onset']:.3f}s  {n.get('name','?'):>4}  {n['freq']:>7.1f}Hz  dur={n['duration']:.3f}s  {ntype}")

# Step 5: Optimize
print(f"\n=== Step 5: CMA-ES optimization (10k evals, note 0) ===")
t0 = time.time()
r = requests.post(f"{BASE}/api/match-single", data={
    "note_job_id": note_job_id, "n_evals": "10000", "selected_note": "0"
})
start_data = r.json()
match_job_id = start_data.get("job_id")
print(f"  Job started: {match_job_id}")

# Poll for result
match_data = None
while True:
    time.sleep(3)
    r = requests.get(f"{BASE}/api/job-result/{match_job_id}")
    if r.status_code == 200:
        d = r.json()
        if d.get("type") == "complete" or d.get("status") == "done":
            match_data = d
            break
        elif d.get("evals"):
            elapsed = time.time() - t0
            loss_v = d.get('loss', '?')
            loss_str = f"{float(loss_v):.4f}" if loss_v != '?' else '?'
            print(f"    Evals: {d['evals']}, Loss: {loss_str}, Time: {elapsed:.0f}s")
    elif r.status_code == 202:
        elapsed = time.time() - t0
        rd = r.json()
        if rd.get("evals"):
            loss_v = rd.get('loss', '?')
            loss_str = f"{float(loss_v):.4f}" if loss_v != '?' else '?'
            print(f"    Evals: {rd['evals']}, Loss: {loss_str}, Time: {elapsed:.0f}s")
        else:
            print(f"    Running... ({elapsed:.0f}s)")

opt_time = time.time() - t0
print(f"  Optimization time: {opt_time:.1f}s")
if not match_data:
    print("  ERROR: No match data received!")
    exit(1)

params = match_data.get("params", [])
final_loss = match_data.get("loss", "?")
print(f"  Final loss: {final_loss}")
print(f"  Params count: {len(params)}")

# Step 6: Render sequence
print(f"\n=== Step 6: Rendering matched sequence ===")
r = requests.post(f"{BASE}/api/render-sequence",
    json={"params": params, "note_job_id": note_job_id, "total_duration": analyzed_dur},
)
matched_wav = r.content
print(f"  Matched WAV: {len(matched_wav)} bytes")

# Step 7: Deep comparison
print(f"\n=== Step 7: Deep audio comparison ===")

matched_audio, matched_sr = sf.read(io.BytesIO(matched_wav))
if matched_audio.ndim > 1:
    matched_audio = matched_audio.mean(axis=1)

r = requests.get(f"{BASE}/api/stem/{sep_id}/other.wav")
orig_audio, orig_sr = sf.read(io.BytesIO(r.content))
if orig_audio.ndim > 1:
    orig_audio = orig_audio.mean(axis=1)

orig_trimmed = orig_audio[:int(analyzed_dur * orig_sr)]
min_len = min(len(orig_trimmed), len(matched_audio))
o = orig_trimmed[:min_len].astype(np.float32)
m = matched_audio[:min_len].astype(np.float32)
o = o / (np.max(np.abs(o)) + 1e-8)
m = m / (np.max(np.abs(m)) + 1e-8)

print(f"  Original:  {len(orig_audio)/orig_sr:.2f}s total, {len(o)/orig_sr:.2f}s analyzed")
print(f"  Matched:   {len(matched_audio)/matched_sr:.2f}s")

# Waveform correlation
corr = float(np.corrcoef(o, m)[0, 1])
print(f"\n  Waveform correlation: {corr:.4f}")

# RMS
o_rms = float(np.sqrt(np.mean(o**2)))
m_rms = float(np.sqrt(np.mean(m**2)))
print(f"  RMS — Original: {o_rms:.4f}, Matched: {m_rms:.4f}")

# Spectral analysis
from scipy import signal as sig

def spectral_centroid(audio, sr):
    freqs, psd = sig.welch(audio, sr, nperseg=4096)
    return float(np.sum(freqs * psd) / (np.sum(psd) + 1e-8))

def spectral_rolloff(audio, sr, pct=0.85):
    freqs, psd = sig.welch(audio, sr, nperseg=4096)
    cumsum = np.cumsum(psd)
    idx = np.searchsorted(cumsum, pct * cumsum[-1])
    return float(freqs[min(idx, len(freqs)-1)])

o_c = spectral_centroid(o, orig_sr)
m_c = spectral_centroid(m, matched_sr)
o_r = spectral_rolloff(o, orig_sr)
m_r = spectral_rolloff(m, matched_sr)
print(f"\n  Spectral centroid — Original: {o_c:.0f}Hz, Matched: {m_c:.0f}Hz (delta: {abs(o_c-m_c):.0f}Hz)")
print(f"  Spectral rolloff  — Original: {o_r:.0f}Hz, Matched: {m_r:.0f}Hz (delta: {abs(o_r-m_r):.0f}Hz)")

# Band energy
def band_energy(audio, sr, low, high):
    freqs, psd = sig.welch(audio, sr, nperseg=4096)
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(psd[mask]))

bands = [(50, 200, "Sub/Bass"), (200, 800, "Low-Mid"), (800, 3000, "Mid"), (3000, 8000, "High-Mid"), (8000, 20000, "High")]
print(f"\n  Band energy comparison:")
print(f"  {'Band':<12} {'Original':>10} {'Matched':>10} {'Ratio':>8}")
for lo, hi, name in bands:
    o_e = band_energy(o, orig_sr, lo, hi)
    m_e = band_energy(m, matched_sr, lo, hi)
    ratio = m_e / (o_e + 1e-12)
    print(f"  {name:<12} {o_e:>10.6f} {m_e:>10.6f} {ratio:>7.2f}x")

# Per-note timing
print(f"\n  Note sequence timeline:")
prev_end = 0
for i, n in enumerate(all_notes):
    gap = n['onset'] - prev_end if prev_end > 0 else 0
    flags = []
    if n['duration'] > 0.4: flags.append("SUSTAIN")
    if gap > 0.3: flags.append(f"gap={gap:.2f}s")
    flag_str = "  ".join(flags)
    print(f"    {i:2d}  {n['onset']:5.2f}s  {n.get('name','?'):>4}  dur={n['duration']:.3f}s  {flag_str}")
    prev_end = n['onset'] + n['duration']

# Save
sf.write("/tmp/kraftwerk_original.wav", o, orig_sr)
sf.write("/tmp/kraftwerk_matched.wav", m, matched_sr)
print(f"\n  Saved: /tmp/kraftwerk_original.wav, /tmp/kraftwerk_matched.wav")
print(f"\n=== SUMMARY ===")
print(f"  Song: {track['artist']['name']} - {track['title']}")
print(f"  Loss: {final_loss}")
print(f"  Correlation: {corr:.4f}")
print(f"  Notes: {len(all_notes)} ({sum(1 for n in all_notes if n['duration'] > 0.4)} sustained)")
print(f"  Spectral centroid delta: {abs(o_c-m_c):.0f}Hz")
