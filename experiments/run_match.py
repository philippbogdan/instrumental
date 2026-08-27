"""
Integration test: run full_pipeline on 3 representative notes, evaluate quality,
render a 21-note sequence, save output/matched_v6.wav, and play results.
"""

import subprocess
import sys
import time

import librosa
import numpy as np
import soundfile as sf
import torch

from src.synth import SynthPatch
from src.optimize import full_pipeline

SR = 44100
NOTES_DIR = "/Users/MOPOLLIKA/coding/hackathons/instrumental/notes"
OUTPUT_PATH = "/Users/MOPOLLIKA/coding/hackathons/instrumental/output/matched_v6.wav"
TARGET_WAV = "/Users/MOPOLLIKA/coding/hackathons/instrumental/output/target.wav"

# The 3 representative notes to match against
NOTE_FILES = [
    (f"{NOTES_DIR}/note_03_A3_221Hz.wav", 221.0),
    (f"{NOTES_DIR}/note_02_Cs4_278Hz.wav", 278.0),
    (f"{NOTES_DIR}/note_01_D4_295Hz.wav", 295.0),
]

# Full 21-note sequence
NOTE_DATA = [
    (0.023, 0.148, 294.5), (0.171, 0.154, 278.0), (0.325, 0.145, 220.6),
    (0.470, 0.151, 294.5), (0.621, 0.145, 278.0), (0.778, 0.136, 220.6),
    (0.914, 0.154, 292.8), (1.068, 0.148, 278.0), (1.216, 0.148, 292.8),
    (1.364, 0.145, 278.0), (1.509, 0.145, 292.8), (1.654, 0.148, 278.0),
    (1.802, 0.157, 220.6), (1.959, 0.145, 294.5), (2.104, 0.148, 278.0),
    (2.252, 0.151, 220.6), (2.403, 0.145, 292.8), (2.548, 0.151, 278.0),
    (2.699, 0.145, 220.6), (2.844, 0.151, 294.5), (2.995, 0.470, 292.8),
]
START_TIME = 1.312  # silence offset in original recording


def load_note(path: str, freq: float) -> dict:
    """Load a WAV note and return {'audio': tensor (1D), 'freq': float}."""
    y, file_sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != SR:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=SR)
    audio = torch.tensor(y, dtype=torch.float32)
    return {"audio": audio, "freq": freq}


def compute_spectral_centroid(y: np.ndarray) -> float:
    """Compute median spectral centroid in Hz."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR)
    return float(np.median(centroid))


def compute_spectral_rolloff(y: np.ndarray) -> float:
    """Compute median spectral rolloff in Hz."""
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR)
    return float(np.median(rolloff))


def compute_rms(y: np.ndarray) -> float:
    """Compute mean RMS."""
    rms = librosa.feature.rms(y=y)
    return float(np.mean(rms))


def render_sequence(params: torch.Tensor, synth: SynthPatch) -> np.ndarray:
    """Render all 21 notes into a single audio buffer."""
    # Determine total length
    last_onset, last_dur, _ = NOTE_DATA[-1]
    total_time = START_TIME + last_onset + last_dur + 0.5
    total_samples = int(total_time * SR)
    output = np.zeros(total_samples)

    with torch.no_grad():
        for onset, duration, freq in NOTE_DATA:
            n_samples = int((duration + 0.05) * SR)  # a bit of tail
            rendered = synth.render(params, f0_hz=freq, duration=duration + 0.05,
                                    note_duration=duration)
            audio_np = rendered.squeeze().numpy()
            start_sample = int((START_TIME + onset) * SR)
            end_sample = start_sample + len(audio_np)
            if end_sample > total_samples:
                audio_np = audio_np[:total_samples - start_sample]
                end_sample = total_samples
            output[start_sample:end_sample] += audio_np

    return output


def normalize_to_match(output: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Normalize output RMS to match reference RMS."""
    ref_rms = np.sqrt(np.mean(reference ** 2) + 1e-8)
    out_rms = np.sqrt(np.mean(output ** 2) + 1e-8)
    scale = ref_rms / out_rms
    normalized = output * scale
    # Hard-clip to [-1, 1] to avoid clipping artifacts
    return np.clip(normalized, -1.0, 1.0)


def main():
    print("=" * 60)
    print("Instrumental v6: CMA-ES + Mel Loss + Spectral Init")
    print("=" * 60)

    # Step 1: Load notes
    print("\n[1] Loading notes...")
    target_notes = []
    for path, freq in NOTE_FILES:
        note = load_note(path, freq)
        print(f"  Loaded {path.split('/')[-1]}: {len(note['audio'])} samples ({freq} Hz)")
        target_notes.append(note)

    # Step 2: Run full pipeline
    print("\n[2] Running full_pipeline (spectral_init -> CMA-ES -> Adam)...")
    synth = SynthPatch(sr=SR)
    t0 = time.time()
    result = full_pipeline(target_notes, synth, n_cmaes=3000, n_adam=500)
    total_time = time.time() - t0

    best_params = result["params"]
    final_loss = result["loss"]
    print(f"\n  Final loss: {final_loss:.6f}")
    print(f"  Total pipeline time: {total_time:.1f}s")

    # Step 3: Self-evaluate on first note (A3 221Hz)
    print("\n[3] Self-evaluation on note_03_A3_221Hz.wav...")
    eval_note = target_notes[0]
    eval_audio_np = eval_note["audio"].numpy()
    eval_freq = eval_note["freq"]
    eval_duration = len(eval_audio_np) / SR

    with torch.no_grad():
        gen_tensor = synth.render(best_params, f0_hz=eval_freq, duration=eval_duration)
    gen_np = gen_tensor.squeeze().numpy()

    orig_centroid = compute_spectral_centroid(eval_audio_np)
    gen_centroid = compute_spectral_centroid(gen_np)
    centroid_diff = abs(orig_centroid - gen_centroid)

    orig_rolloff = compute_spectral_rolloff(eval_audio_np)
    gen_rolloff = compute_spectral_rolloff(gen_np)
    rolloff_diff = abs(orig_rolloff - gen_rolloff)

    orig_rms = compute_rms(eval_audio_np)
    gen_rms = compute_rms(gen_np)
    rms_diff = abs(orig_rms - gen_rms)

    print(f"  Spectral centroid: orig={orig_centroid:.1f} Hz, gen={gen_centroid:.1f} Hz, diff={centroid_diff:.1f} Hz")
    print(f"  Spectral rolloff:  orig={orig_rolloff:.1f} Hz, gen={gen_rolloff:.1f} Hz, diff={rolloff_diff:.1f} Hz")
    print(f"  RMS:               orig={orig_rms:.4f}, gen={gen_rms:.4f}, diff={rms_diff:.4f}")

    # Step 4: Render full 21-note sequence
    print("\n[4] Rendering 21-note sequence...")
    output_audio = render_sequence(best_params, synth)

    # Load target for amplitude normalization
    target_audio_np, _ = sf.read(TARGET_WAV)
    if target_audio_np.ndim > 1:
        target_audio_np = target_audio_np.mean(axis=1)

    output_normalized = normalize_to_match(output_audio, target_audio_np)

    # Save output
    sf.write(OUTPUT_PATH, output_normalized, SR)
    print(f"  Saved: {OUTPUT_PATH}")

    # Step 5: Results table
    print("\n" + "=" * 60)
    print("RESULTS TABLE")
    print("=" * 60)
    centroid_pass = centroid_diff < 500
    loss_pass = final_loss < 2.0
    print(f"  {'Metric':<30} {'Value':>12} {'Status':>8}")
    print(f"  {'-'*50}")
    print(f"  {'Centroid diff (Hz)':<30} {centroid_diff:>12.1f} {'PASS' if centroid_pass else 'FAIL':>8}")
    print(f"  {'Final loss':<30} {final_loss:>12.6f} {'PASS' if loss_pass else 'FAIL':>8}")
    print(f"  {'Rolloff diff (Hz)':<30} {rolloff_diff:>12.1f}")
    print(f"  {'RMS diff':<30} {rms_diff:>12.4f}")
    print(f"  {'Pipeline time (s)':<30} {total_time:>12.1f}")
    print("=" * 60)

    overall = "PASS" if (centroid_pass and loss_pass) else "FAIL"
    print(f"  OVERALL: {overall}")
    print("=" * 60)

    # Step 6: Play audio
    print("\n[5] Playing target.wav...")
    subprocess.run(["afplay", TARGET_WAV])
    print("[5] Playing matched_v6.wav...")
    subprocess.run(["afplay", OUTPUT_PATH])

    print("\nDone.")
    return {
        "final_loss": final_loss,
        "centroid_diff": centroid_diff,
        "rolloff_diff": rolloff_diff,
        "rms_diff": rms_diff,
        "centroid_pass": centroid_pass,
        "loss_pass": loss_pass,
        "overall": overall,
        "total_time": total_time,
    }


if __name__ == "__main__":
    main()
