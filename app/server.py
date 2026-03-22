"""FastAPI backend for INSTRUMENTAL — inverse synthesis web app."""

import sys
import os
import io
import uuid
import time
import asyncio
import json
import multiprocessing
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import librosa
import soundfile as sf
import cma
import httpx
import torchaudio

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Path setup so we can import from src/ ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.synth import SynthPatch, PARAM_DEFS
from src.losses import get_loss
from src.spectral_init import spectral_init

N_PARAMS = len(PARAM_DEFS)  # 28

app = FastAPI(title="INSTRUMENTAL")

# Fork required for multiprocessing.Pool on macOS (spawn fails with module imports)
multiprocessing.set_start_method("fork", force=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store: job_id -> {queue, status, result, ...}
jobs: Dict[str, Dict[str, Any]] = {}

# Concurrency locks: prevent OOM from concurrent heavy compute
# Created lazily on first use (Python 3.9 asyncio.Lock binds to event loop at creation)
_demucs_lock = None
_cmaes_lock = None
_job_queue: list = []           # list of job_ids waiting for CMA-ES


def _get_demucs_lock():
    global _demucs_lock
    if _demucs_lock is None:
        _demucs_lock = asyncio.Lock()
    return _demucs_lock


def _get_cmaes_lock():
    global _cmaes_lock
    if _cmaes_lock is None:
        _cmaes_lock = asyncio.Lock()
    return _cmaes_lock

# Stem separation store: job_id -> {"path": tempdir, "created": time}
stem_dirs: Dict[str, Dict] = {}

APP_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_spectral_init(init_15: torch.Tensor) -> torch.Tensor:
    """Pad 15-param spectral_init output to full 28 params with sensible defaults."""
    x = torch.full((N_PARAMS,), 0.5, dtype=torch.float32)
    # Map spectral_init's 15 outputs to PARAM_DEFS order
    x[0] = init_15[0]    # saw_mix
    x[1] = init_15[1]    # square_mix
    x[2] = init_15[2]    # sine_mix
    x[3] = init_15[3]    # noise_mix
    x[4] = init_15[4]    # detune
    x[5] = init_15[5]    # filter_cutoff
    x[6] = init_15[6]    # filter_resonance
    x[7] = init_15[7]    # attack
    x[8] = init_15[8]    # decay
    x[9] = init_15[9]    # sustain
    x[10] = init_15[10]  # release
    x[11] = init_15[11]  # gain
    x[12] = init_15[12]  # filter_env
    x[13] = init_15[13]  # reverb_size
    x[14] = init_15[14]  # reverb_mix
    # Remaining params get defaults:
    x[15] = 0.5   # unison_voices
    x[16] = 0.15  # unison_spread
    x[17] = 0.5   # noise_floor  (midpoint)
    x[18] = 0.05  # filter_attack
    x[19] = 0.3   # filter_decay
    x[20] = 0.5   # filter_sustain
    x[21] = 0.3   # filter_release
    x[22] = 0.5   # pulse_width
    x[23] = 0.5   # filter_slope
    x[24] = 0.5   # eq1_freq
    x[25] = 0.5   # eq1_gain
    x[26] = 0.5   # eq2_freq
    x[27] = 0.5   # eq2_gain
    return x


async def _load_audio_from_upload(file: UploadFile):
    """Load audio from an uploaded file (handles MP3, WAV, etc.)."""
    audio_bytes = await file.read()
    ext = os.path.splitext(file.filename or ".wav")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(audio_bytes)
    tmp.close()
    audio_np, sr = librosa.load(tmp.name, sr=44100, mono=True)
    os.unlink(tmp.name)
    return audio_np, sr


MIN_NOTE_SAMPLES = 4500  # ~0.1s at 44100Hz, enough for loss function STFTs

def _extract_notes(audio_np: np.ndarray, sr: int = 44100, max_notes: int = 3, max_note_dur: float = 0.5):
    """
    Extract individual notes from audio via onset detection + pitch tracking.
    Returns list of (audio_segment, freq_hz, duration) tuples for CMA-ES multi-pitch fitting.
    Each segment is guaranteed >= MIN_NOTE_SAMPLES to avoid FFT crashes.
    """
    audio_np = audio_np.astype(np.float32)

    # Onset detection with adaptive sensitivity
    # Start sensitive (delta=0.03), increase if too many false positives
    for delta in [0.03, 0.05, 0.07]:
        onsets_frames = librosa.onset.onset_detect(
            y=audio_np, sr=sr, hop_length=512, delta=delta
        )
        if len(onsets_frames) <= 40:
            break
    onset_times = librosa.frames_to_time(onsets_frames, sr=sr, hop_length=512)

    if len(onset_times) == 0:
        # F3 fallback: no onsets detected (sustained pad/drone)
        print("[_extract_notes] WARNING: No onsets detected, using 1s chunk at median pitch")
        chunk = audio_np[:min(len(audio_np), sr)]  # first 1 second
        f0 = _detect_pitch(chunk, sr)
        return [(chunk, f0, len(chunk) / sr)], [{"onset": 0, "freq": f0, "duration": len(chunk) / sr}]

    # Build note segments with pitch per segment
    notes = []
    for i, onset in enumerate(onset_times):
        if i + 1 < len(onset_times):
            duration = onset_times[i + 1] - onset
        else:
            duration = len(audio_np) / sr - onset
        # F2: Skip very short segments (transients, bleed artifacts)
        if duration < 0.05:
            continue
        # Cap note duration for fast rendering
        duration = min(duration, max_note_dur)

        start_sample = int(onset * sr)
        end_sample = start_sample + int(duration * sr)
        end_sample = min(end_sample, len(audio_np))
        seg_audio = audio_np[start_sample:end_sample]

        # Skip notes too short for the loss function's FFT (8192 samples = ~0.19s)
        if len(seg_audio) < MIN_NOTE_SAMPLES:
            continue

        # Detect pitch with fmax=500 only (prevents harmonic locking)
        # No fallback to higher range - if there's no fundamental below 500Hz, skip
        f0_arr, voiced, _ = librosa.pyin(
            seg_audio, fmin=librosa.note_to_hz("C2"), fmax=500.0, sr=sr
        )
        voiced_f0 = f0_arr[voiced] if voiced is not None else np.array([])
        if len(voiced_f0) == 0:
            continue
        # Reject segments where pitch detection is unreliable (<50% voiced)
        voiced_pct = len(voiced_f0) / len(f0_arr) if len(f0_arr) > 0 else 0
        if voiced_pct < 0.5:
            continue
        freq = float(np.median(voiced_f0))
        if freq < 50 or freq > 2000 or np.isnan(freq):
            continue

        notes.append({"audio": seg_audio, "freq": freq, "duration": duration, "onset": onset})

    if len(notes) == 0:
        # F3 fallback
        print("[_extract_notes] WARNING: Pitch detection failed on all segments, using 1s chunk")
        chunk = audio_np[:min(len(audio_np), sr)]
        f0 = _detect_pitch(chunk, sr)
        return [(chunk, f0, len(chunk) / sr)], [{"onset": 0, "freq": f0, "duration": len(chunk) / sr}]

    # Filter out likely sub-harmonics: if most notes are above 150Hz, remove notes below 100Hz
    freqs = [n["freq"] for n in notes]
    median_freq = float(np.median(freqs))
    if median_freq > 150:
        notes = [n for n in notes if n["freq"] > 100]
    if len(notes) == 0:
        chunk = audio_np[:min(len(audio_np), sr)]
        f0 = _detect_pitch(chunk, sr)
        return [(chunk, f0, len(chunk) / sr)], [{"onset": 0, "freq": f0, "duration": len(chunk) / sr}]

    # Merge consecutive notes at the same pitch into sustained notes
    # Prevents a held synth pad from becoming 10 rapid re-triggers
    notes.sort(key=lambda n: n["onset"])
    merged = []
    for nd in notes:
        semitone = round(12 * np.log2(nd["freq"] / 440.0))
        if merged:
            prev = merged[-1]
            prev_semi = round(12 * np.log2(prev["freq"] / 440.0))
            prev_end = prev["onset"] + prev["duration"]
            gap = nd["onset"] - prev_end
            if prev_semi == semitone and gap < 0.15:
                # Merge: extend duration, concatenate audio
                new_dur = nd["onset"] + nd["duration"] - prev["onset"]
                # Build merged audio from original signal
                start_s = int(prev["onset"] * sr)
                end_s = min(int((prev["onset"] + new_dur) * sr), len(audio_np))
                prev["audio"] = audio_np[start_s:end_s]
                prev["duration"] = new_dur
                continue
        merged.append(nd)
    notes = merged

    # Select up to max_notes representative notes (one per unique semitone, prefer loudest)
    # Loudest notes are most likely real signal, not Demucs bleed artifacts
    for nd in notes:
        nd["rms"] = float(np.sqrt(np.mean(nd["audio"] ** 2)))
    seen_semitones = {}
    for nd in sorted(notes, key=lambda n: n["rms"], reverse=True):  # loudest first
        semitone = round(12 * np.log2(nd["freq"] / 440.0))
        if semitone not in seen_semitones and len(seen_semitones) < max_notes:
            seen_semitones[semitone] = nd

    representative = [
        (nd["audio"], nd["freq"], nd["duration"])
        for nd in seen_semitones.values()
    ]

    # Also build full note list with onsets for sequence playback
    all_notes_with_onsets = [
        {"onset": nd["onset"], "freq": nd["freq"], "duration": nd["duration"]}
        for nd in notes
    ]

    print(f"[_extract_notes] Extracted {len(notes)} notes, selected {len(representative)} representatives "
          f"at pitches: {[f'{r[1]:.0f}Hz' for r in representative]}")
    return representative, all_notes_with_onsets


def _detect_pitch(audio_np: np.ndarray, sr: int = 44100) -> float:
    """Detect fundamental frequency using pyin with fmax=500Hz."""
    f0, voiced_flag, _ = librosa.pyin(
        audio_np, fmin=librosa.note_to_hz("C2"), fmax=500.0, sr=sr,
    )
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) > 0:
        return float(np.median(voiced_f0))
    return 440.0


# Module-level globals for multiprocessing workers
_WORKER_TARGET_NOTES = []

# Indices of the 13 params CMA-ES optimizes (the rest are frozen)
# Chosen by round-trip recovery test: only params consistently recovered
OPTIMIZED_INDICES = [
    0,   # saw_mix
    2,   # sine_mix
    3,   # noise_mix
    4,   # detune
    5,   # filter_cutoff
    6,   # filter_resonance
    7,   # attack
    8,   # decay
    9,   # sustain
    10,  # release
    11,  # gain
    12,  # filter_env
    22,  # pulse_width
]
N_OPTIMIZED = len(OPTIMIZED_INDICES)

# Frozen defaults for the other 15 params (sensible neutral values)
FROZEN_DEFAULTS = np.full(N_PARAMS, 0.5, dtype=np.float32)
FROZEN_DEFAULTS[1] = 0.0    # square_mix — degenerate with saw
FROZEN_DEFAULTS[13] = 0.3   # reverb_size
FROZEN_DEFAULTS[14] = 0.05  # reverb_mix — nearly dry
FROZEN_DEFAULTS[15] = 0.0   # unison_voices — 1 voice
FROZEN_DEFAULTS[16] = 0.0   # unison_spread — none
FROZEN_DEFAULTS[17] = 0.0   # noise_floor — none
FROZEN_DEFAULTS[18] = 0.05  # filter_attack
FROZEN_DEFAULTS[19] = 0.3   # filter_decay
FROZEN_DEFAULTS[20] = 0.5   # filter_sustain
FROZEN_DEFAULTS[21] = 0.3   # filter_release
FROZEN_DEFAULTS[23] = 0.25  # filter_slope — gentle
FROZEN_DEFAULTS[24] = 0.5   # eq1_freq — mid
FROZEN_DEFAULTS[25] = 0.5   # eq1_gain — 0dB
FROZEN_DEFAULTS[26] = 0.5   # eq2_freq — mid
FROZEN_DEFAULTS[27] = 0.5   # eq2_gain — 0dB


def _expand_params(reduced_np):
    """Expand 13 optimized params into full 28 by injecting frozen defaults."""
    full = FROZEN_DEFAULTS.copy()
    for i, idx in enumerate(OPTIMIZED_INDICES):
        full[idx] = reduced_np[i]
    return full


def _reduce_params(full_np):
    """Extract the 13 optimized params from a full 28-param vector."""
    return np.array([full_np[idx] for idx in OPTIMIZED_INDICES], dtype=np.float32)


def _evaluate_single(reduced_np):
    """Evaluate a single candidate in a worker process."""
    synth = SynthPatch()
    loss_fn = get_loss("matching")
    full_np = _expand_params(reduced_np)
    params = torch.tensor(full_np, dtype=torch.float32)
    total = 0.0
    for audio_np, freq, dur in _WORKER_TARGET_NOTES:
        target = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        gen = synth.render(params, f0_hz=freq, duration=dur, note_duration=dur * 0.9)
        ml = min(target.shape[1], gen.shape[1])
        loss = loss_fn(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
        total += loss.item()
    return total / len(_WORKER_TARGET_NOTES)


def _init_worker(target_notes):
    """Initializer for pool workers."""
    global _WORKER_TARGET_NOTES
    _WORKER_TARGET_NOTES = target_notes


def _run_cmaes(
    target_notes: list,
    x0: np.ndarray,
    n_evals: int,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    note_data_for_result: Optional[list] = None,
):
    """Run CMA-ES with CPU multiprocessing in reduced 13D param space."""
    sigma0 = 0.15
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    popsize = max(16, n_cores * 2)

    # Reduce x0 from 28D to 13D
    x0_reduced = _reduce_params(x0)

    es = cma.CMAEvolutionStrategy(x0_reduced, sigma0, {
        "bounds": [[0] * N_OPTIMIZED, [1] * N_OPTIMIZED],
        "maxfevals": n_evals,
        "popsize": popsize,
        "verbose": -9,
        "tolx": 1e-8,
        "tolfun": 1e-10,
    })

    best_loss = float("inf")
    evals = 0
    t0 = time.time()

    with multiprocessing.Pool(n_cores, initializer=_init_worker, initargs=(target_notes,)) as pool:
        while not es.stop():
            solutions = es.ask()
            fitnesses = pool.map(_evaluate_single, solutions)
            es.tell(solutions, fitnesses)
            evals += len(solutions)
            gen_best = min(fitnesses)
            if gen_best < best_loss:
                best_loss = gen_best

            elapsed = time.time() - t0
            msg = {
                "type": "progress",
                "evals": evals,
                "total_evals": n_evals,
                "best_loss": round(best_loss, 4),
                "elapsed_seconds": round(elapsed, 1),
            }
            loop.call_soon_threadsafe(queue.put_nowait, msg)

    # Expand best result from 13D back to 28D
    param_defs = [{"name": name, "lo": lo, "hi": hi} for name, lo, hi in PARAM_DEFS]
    best_params = list(_expand_params(es.result.xbest))

    complete_msg = {
        "type": "complete",
        "params": best_params,
        "param_defs": param_defs,
        "loss": round(best_loss, 4),
        "notes": note_data_for_result,
    }
    loop.call_soon_threadsafe(queue.put_nowait, complete_msg)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/job-result/{job_id}")
async def job_result(job_id: str):
    """Poll endpoint for job result/progress (fallback when WebSocket drops)."""
    if job_id not in jobs:
        return JSONResponse({"error": "Unknown job"}, status_code=404)
    job = jobs[job_id]
    if job.get("status") == "complete" and job.get("result"):
        return job["result"]
    # Drain any pending progress messages from the queue
    queue = job.get("queue")
    latest = None
    if queue:
        while not queue.empty():
            try:
                msg = queue.get_nowait()
                if msg.get("type") == "complete":
                    job["result"] = msg
                    job["status"] = "complete"
                    return msg
                latest = msg
            except asyncio.QueueEmpty:
                break
    if latest:
        return latest
    return JSONResponse({"status": "running"}, status_code=202)


@app.get("/api/preset/best")
async def preset_best():
    """Return the best-ever params (v24, loss 2.09) padded to 28 with 0.5 defaults."""
    params_path = os.path.join(os.path.dirname(APP_DIR), "output", "best_params_v24.npy")
    p24 = np.load(params_path)
    p28 = np.full(N_PARAMS, 0.5, dtype=np.float32)
    p28[:len(p24)] = p24
    param_defs = [{"name": name, "lo": lo, "hi": hi} for name, lo, hi in PARAM_DEFS]
    return {"params": p28.tolist(), "param_defs": param_defs, "loss": 2.09}


@app.post("/api/match-single")
async def match_single(
    file: UploadFile = File(None),
    n_evals: int = Form(10000),
    stem_job_id: Optional[str] = Form(None),
    stem_name: Optional[str] = Form(None),
    note_job_id: Optional[str] = Form(None),
    selected_note: Optional[int] = Form(None),
):
    # If pre-extracted notes exist, use them directly (skip re-extraction)
    if note_job_id and note_job_id in note_jobs:
        target_notes = note_jobs[note_job_id]["target_notes"]
        # Always optimize on ALL representative notes (like v24 did)
        # Multi-note gives the loss function more pitch references for robust matching
        print(f"[match-single] Using all {len(target_notes)} notes from {note_job_id}")
    else:
        # Load audio from stem or upload
        if stem_job_id and stem_name and stem_job_id in stem_dirs:
            sep_base = os.path.join(stem_dirs[stem_job_id]["path"], "separated")
            model_dirs = [d for d in os.listdir(sep_base) if os.path.isdir(os.path.join(sep_base, d))] if os.path.exists(sep_base) else []
            sep_dir = os.path.join(sep_base, model_dirs[0]) if model_dirs else sep_base
            subdirs = [d for d in os.listdir(sep_dir) if os.path.isdir(os.path.join(sep_dir, d))] if os.path.exists(sep_dir) else []
            stem_path = os.path.join(sep_dir, subdirs[0], f"{stem_name}.wav") if subdirs else None
            if stem_path and os.path.exists(stem_path):
                audio_np, sr = sf.read(stem_path)
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                if sr != 44100:
                    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=44100)
                    sr = 44100
            else:
                audio_np, sr = await _load_audio_from_upload(file)
        else:
            audio_np, sr = await _load_audio_from_upload(file)

        if len(audio_np) > 10 * 44100:
            audio_np = audio_np[:10 * 44100]

        target_notes, _ = await asyncio.to_thread(
            _extract_notes, audio_np, 44100, 5, 0.5
        )

    # Spectral init from first representative note
    init_15 = await asyncio.to_thread(spectral_init, target_notes[0][0], 44100)
    x0 = _pad_spectral_init(init_15).numpy()

    job_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    jobs[job_id] = {"queue": queue, "status": "queued"}

    loop = asyncio.get_event_loop()
    asyncio.ensure_future(
        _queued_cmaes(job_id, target_notes, x0, n_evals, queue, loop, None)
    )

    return {"job_id": job_id, "status": "started"}


async def _queued_cmaes(job_id, target_notes, x0, n_evals, queue, loop, note_data):
    """Run CMA-ES with queue management. Only one job runs at a time."""
    _job_queue.append(job_id)

    lock = _get_cmaes_lock()
    # If lock is held, send queue position updates while waiting
    if lock.locked():
        jobs[job_id]["status"] = "queued"
        while lock.locked():
            pos = _job_queue.index(job_id) if job_id in _job_queue else 0
            queue.put_nowait({
                "type": "queued",
                "position": pos,
                "message": f"Position {pos} in queue" if pos > 0 else "Starting soon..."
            })
            await asyncio.sleep(3)

    # Acquire lock (blocks if another job just started between our check and acquire)
    async with lock:
        jobs[job_id]["status"] = "running"
        try:
            _job_queue.remove(job_id)
        except ValueError:
            pass
        try:
            await asyncio.to_thread(_run_cmaes, target_notes, x0, n_evals, queue, loop, note_data)
        except Exception as e:
            # If CMA-ES crashes, send error and release lock for next job
            queue.put_nowait({"type": "error", "message": str(e)})
            print(f"[_queued_cmaes] Job {job_id} failed: {e}")


@app.post("/api/match-sequence")
async def match_sequence(
    file: UploadFile = File(...),
    n_evals: int = Form(10000),
    stem_job_id: Optional[str] = Form(None),
    stem_name: Optional[str] = Form(None),
):
    # If stem params provided, load stem WAV instead of uploaded file
    if stem_job_id and stem_name and stem_job_id in stem_dirs:
        sep_base = os.path.join(stem_dirs[stem_job_id]["path"], "separated")
        model_dirs = [d for d in os.listdir(sep_base) if os.path.isdir(os.path.join(sep_base, d))] if os.path.exists(sep_base) else []
        sep_dir = os.path.join(sep_base, model_dirs[0]) if model_dirs else sep_base
        subdirs = [d for d in os.listdir(sep_dir) if os.path.isdir(os.path.join(sep_dir, d))] if os.path.exists(sep_dir) else []
        stem_path = os.path.join(sep_dir, subdirs[0], f"{stem_name}.wav") if subdirs else None
        if stem_path and os.path.exists(stem_path):
            audio_np, sr = sf.read(stem_path)
        else:
            audio_np, sr = await _load_audio_from_upload(file)
    else:
        audio_np, sr = await _load_audio_from_upload(file)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr != 44100:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=44100)
        sr = 44100

    audio_np = audio_np.astype(np.float32)

    # Pitch detection with torchcrepe
    import torchcrepe
    audio_t = torch.tensor(audio_np).unsqueeze(0)
    pitch = torchcrepe.predict(
        audio_t, sr, hop_length=256, fmin=50, fmax=2000,
        model="tiny", batch_size=1, device="cpu",
    )
    pitch_np = pitch.squeeze().numpy()

    # Onset detection
    onsets_frames = librosa.onset.onset_detect(y=audio_np, sr=sr, hop_length=256)
    onset_times = librosa.frames_to_time(onsets_frames, sr=sr, hop_length=256)

    # Build note segments
    note_data = []
    for i, onset in enumerate(onset_times):
        if i + 1 < len(onset_times):
            duration = onset_times[i + 1] - onset
        else:
            duration = len(audio_np) / sr - onset
        if duration < 0.02:
            continue
        # Get pitch for this segment
        onset_frame = int(onset * sr / 256)
        end_frame = int((onset + duration) * sr / 256)
        end_frame = min(end_frame, len(pitch_np))
        if onset_frame >= end_frame:
            continue
        seg_pitch = pitch_np[onset_frame:end_frame]
        # Filter out zeros/NaN
        valid = seg_pitch[(seg_pitch > 50) & (seg_pitch < 2000)]
        if len(valid) == 0:
            continue
        freq = float(np.median(valid))
        note_data.append({"onset": round(onset, 4), "duration": round(duration, 4), "freq": round(freq, 1)})

    # Select up to 3 representative notes (one per unique pitch rounded to nearest semitone)
    seen_pitches = {}
    representative = []
    for nd in note_data:
        semitone = round(12 * np.log2(nd["freq"] / 440.0))
        if semitone not in seen_pitches and len(representative) < 3:
            seen_pitches[semitone] = True
            # Extract audio segment
            start_sample = int(nd["onset"] * sr)
            end_sample = start_sample + int(nd["duration"] * sr)
            seg_audio = audio_np[start_sample:end_sample]
            if len(seg_audio) > 0:
                representative.append((seg_audio, nd["freq"], nd["duration"]))

    if not representative:
        # Fallback: use whole audio
        f0 = _detect_pitch(audio_np, sr)
        representative = [(audio_np, f0, len(audio_np) / sr)]

    # Spectral init from first representative note
    init_15 = spectral_init(representative[0][0], sr)
    x0 = _pad_spectral_init(init_15).numpy()

    job_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    jobs[job_id] = {"queue": queue, "status": "queued"}

    loop = asyncio.get_event_loop()
    asyncio.ensure_future(
        _queued_cmaes(job_id, representative, x0, n_evals, queue, loop, note_data)
    )

    return {"job_id": job_id, "status": "started"}


@app.websocket("/ws/progress/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in jobs:
        await websocket.send_json({"type": "error", "message": "Unknown job_id"})
        await websocket.close()
        return

    queue = jobs[job_id]["queue"]
    try:
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)
            if msg.get("type") == "complete":
                jobs[job_id]["result"] = msg
                jobs[job_id]["status"] = "complete"
                break
    except WebSocketDisconnect:
        # WS dropped but optimization continues. Drain queue in background
        # so result is stored when it arrives.
        async def drain():
            try:
                while True:
                    msg = await asyncio.wait_for(queue.get(), timeout=600)
                    if msg.get("type") == "complete":
                        jobs[job_id]["result"] = msg
                        jobs[job_id]["status"] = "complete"
                        break
            except Exception:
                pass
        asyncio.ensure_future(drain())


@app.post("/api/render-sequence")
async def render_sequence(body: dict):
    """Render matched synth at all detected note onsets into one WAV."""
    params = body.get("params", [0.5] * N_PARAMS)
    note_job_id = body.get("note_job_id")

    if not note_job_id or note_job_id not in note_jobs:
        return JSONResponse({"error": "Unknown note job"}, status_code=404)

    all_notes = note_jobs[note_job_id].get("all_notes", [])
    if not all_notes:
        return JSONResponse({"error": "No notes available"}, status_code=400)

    def do_render():
        synth = SynthPatch()
        params_t = torch.tensor(params, dtype=torch.float32)

        # Match the original stem duration (or use note extent + padding)
        requested_dur = body.get("total_duration", 0)
        max_end = max(n["onset"] + n["duration"] + 0.3 for n in all_notes)
        total_dur = max(max_end, float(requested_dur))
        total_samples = int(total_dur * 44100)
        output = np.zeros(total_samples, dtype=np.float32)

        for note in all_notes:
            dur = note["duration"] + 0.2  # add release tail
            audio = synth.render(params_t, f0_hz=note["freq"], duration=dur, note_duration=note["duration"])
            audio_np = audio.detach().numpy().squeeze()
            pos = int(note["onset"] * 44100)
            end = pos + len(audio_np)
            if end <= total_samples:
                output[pos:end] += audio_np
            else:
                trim = total_samples - pos
                if trim > 0:
                    output[pos:pos + trim] += audio_np[:trim]

        # Normalize
        peak = np.max(np.abs(output))
        if peak > 0.01:
            output = output / peak * 0.9

        buf = io.BytesIO()
        sf.write(buf, output, 44100, format="WAV")
        buf.seek(0)
        return buf.read()

    wav_bytes = await asyncio.to_thread(do_render)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/api/export/vital")
async def export_vital(body: dict):
    """Export matched params as a Vital .vital preset file."""
    from src.vital_export import write_vital_preset
    params = body.get("params", [0.5] * N_PARAMS)
    param_defs = [{"name": name, "lo": lo, "hi": hi} for name, lo, hi in PARAM_DEFS]
    tmp_path = os.path.join(tempfile.gettempdir(), "instrumental_export.vital")
    write_vital_preset(params, param_defs, tmp_path)
    return FileResponse(tmp_path, media_type="application/json", filename="INSTRUMENTAL_Match.vital")


@app.post("/api/render")
async def render_audio(body: dict):
    params = body.get("params", [0.5] * N_PARAMS)
    freq = body.get("freq", 440.0)
    duration = body.get("duration", 1.0)

    params_t = torch.tensor(params, dtype=torch.float32)
    synth = SynthPatch()
    audio = synth.render(params_t, f0_hz=freq, duration=duration, note_duration=duration * 0.9)
    audio_np = audio.detach().numpy().squeeze()

    buf = io.BytesIO()
    sf.write(buf, audio_np, 44100, format="WAV")
    buf.seek(0)

    return Response(content=buf.read(), media_type="audio/wav")


# ---------------------------------------------------------------------------
# Deezer proxy endpoints (avoids CORS issues in browser)
# ---------------------------------------------------------------------------

@app.get("/api/search")
async def search_deezer(q: str = ""):
    """Proxy Deezer search API."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"https://api.deezer.com/search?q={q}")
            return resp.json()
    except Exception:
        return {"data": []}


@app.get("/api/preview")
async def preview_proxy(url: str = ""):
    """Proxy Deezer MP3 preview to avoid CORS."""
    if not (url.startswith("https://cdns-preview-") or url.startswith("https://cdnt-preview.dzcdn.net/")):
        return JSONResponse({"error": "Invalid URL"}, status_code=400)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            return Response(content=resp.content, media_type="audio/mpeg")
    except Exception:
        return JSONResponse({"error": "Preview fetch failed"}, status_code=502)


# ---------------------------------------------------------------------------
# Note extraction preview
# ---------------------------------------------------------------------------

# Store extracted notes: job_id -> {"path": dir, "notes": [{freq, dur, file}], "target_notes": [...]}
note_jobs: Dict[str, Dict] = {}


@app.post("/api/extract-notes")
async def extract_notes_endpoint(
    stem_job_id: str = Form(...),
    stem_name: str = Form(...),
):
    """Extract notes from a separated stem. Returns playable WAV URLs + pitch info."""
    if stem_job_id not in stem_dirs:
        return JSONResponse({"error": "Unknown stem job"}, status_code=404)

    # Load the stem audio
    sep_base = os.path.join(stem_dirs[stem_job_id]["path"], "separated")
    model_dirs = [d for d in os.listdir(sep_base) if os.path.isdir(os.path.join(sep_base, d))] if os.path.exists(sep_base) else []
    if not model_dirs:
        return JSONResponse({"error": "Stems not found"}, status_code=404)
    sep_dir = os.path.join(sep_base, model_dirs[0])
    subdirs = [d for d in os.listdir(sep_dir) if os.path.isdir(os.path.join(sep_dir, d))]
    if not subdirs:
        return JSONResponse({"error": "Stems not found"}, status_code=404)
    stem_path = os.path.join(sep_dir, subdirs[0], f"{stem_name}.wav")
    if not os.path.exists(stem_path):
        return JSONResponse({"error": f"Stem {stem_name} not found"}, status_code=404)

    audio_np, sr = sf.read(stem_path)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if sr != 44100:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=44100)
        sr = 44100

    # Cap to 10s
    analyzed_duration = min(len(audio_np) / sr, 10.0)
    if len(audio_np) > 10 * sr:
        audio_np = audio_np[:10 * sr]

    # Extract notes in thread
    target_notes, all_notes_with_onsets = await asyncio.to_thread(
        _extract_notes, audio_np.astype(np.float32), sr, 5, 0.5
    )

    # Save each representative note as a WAV file
    note_job_id = str(uuid.uuid4())
    notes_dir = tempfile.mkdtemp(prefix="instrumental_notes_")
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_info = []

    for i, (seg, freq, dur) in enumerate(target_notes):
        wav_path = os.path.join(notes_dir, f"note_{i}.wav")
        sf.write(wav_path, seg.astype(np.float32), sr)

        midi = round(12 * np.log2(freq / 440.0) + 69)
        name = note_names[midi % 12] + str(midi // 12 - 1)
        rms = float(np.sqrt(np.mean(seg ** 2)))

        note_info.append({
            "index": i,
            "freq": round(freq, 1),
            "name": name,
            "duration": round(dur, 3),
            "rms": round(rms, 4),
            "url": f"/api/note/{note_job_id}/{i}.wav",
        })

    # Add note names to all_notes for sequence playback
    for nd in all_notes_with_onsets:
        midi = round(12 * np.log2(nd["freq"] / 440.0) + 69)
        nd["name"] = note_names[midi % 12] + str(midi // 12 - 1)
        nd["midi"] = midi

    note_jobs[note_job_id] = {
        "path": notes_dir,
        "notes": note_info,
        "target_notes": target_notes,
        "all_notes": all_notes_with_onsets,  # full sequence with onsets
        "stem_job_id": stem_job_id,
        "stem_name": stem_name,
        "created": time.time(),
    }

    is_fallback = len(target_notes) == 1 and target_notes[0][2] > 0.5
    return {
        "note_job_id": note_job_id,
        "notes": note_info,
        "all_notes": all_notes_with_onsets,  # for sequence playback
        "analyzed_duration": round(analyzed_duration, 2),
        "fallback": is_fallback,
    }


@app.get("/api/note/{note_job_id}/{note_index}.wav")
async def get_note_wav(note_job_id: str, note_index: int):
    """Serve an extracted note WAV file."""
    if note_job_id not in note_jobs:
        return JSONResponse({"error": "Unknown note job"}, status_code=404)
    wav_path = os.path.join(note_jobs[note_job_id]["path"], f"note_{note_index}.wav")
    if not os.path.exists(wav_path):
        return JSONResponse({"error": "Note not found"}, status_code=404)
    return FileResponse(wav_path, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Demucs source separation
# ---------------------------------------------------------------------------

@app.post("/api/separate")
async def separate_stems(file: UploadFile = File(...), stem_count: int = Form(4)):
    """Run Demucs source separation. stem_count=4 (htdemucs_ft) or 6 (htdemucs_6s)."""
    job_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp(prefix="instrumental_")
    # Preserve original extension so Demucs/ffmpeg handles format correctly
    ext = os.path.splitext(file.filename or "input.wav")[1] or ".wav"
    input_path = os.path.join(tmp_dir, "input" + ext)

    audio_bytes = await file.read()
    with open(input_path, "wb") as f:
        f.write(audio_bytes)

    model_name = "htdemucs_6s" if stem_count == 6 else "htdemucs_ft"
    stem_names = ("vocals", "drums", "bass", "other", "guitar", "piano") if stem_count == 6 else ("vocals", "drums", "bass", "other")

    def run_demucs():
        out_dir = os.path.join(tmp_dir, "separated")
        os.makedirs(out_dir, exist_ok=True)
        result = subprocess.run(
            [
                sys.executable, "-m", "demucs",
                "-n", model_name,
                "-d", "mps",
                "--float32",
                "-o", out_dir,
                input_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[Demucs] STDERR: {result.stderr[:500]}")
            raise RuntimeError(f"Demucs failed: {result.stderr[:200]}")
        # Find output directory dynamically
        model_dirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
        if not model_dirs:
            raise RuntimeError("Demucs produced no output directory")
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        stems_dir = os.path.join(out_dir, model_dirs[0], input_basename)
        stems = {}
        for sn in stem_names:
            stem_path = os.path.join(stems_dir, f"{sn}.wav")
            if os.path.exists(stem_path):
                stems[sn] = f"/api/stem/{job_id}/{sn}.wav"
        return stems

    # Lock: only one Demucs runs at a time to prevent OOM (~3GB RAM each)
    async with _get_demucs_lock():
        stems = await asyncio.to_thread(run_demucs)
    stem_dirs[job_id] = {"path": tmp_dir, "created": time.time()}
    return {"job_id": job_id, "stems": stems}


@app.get("/api/stem/{job_id}/{stem_name}.wav")
async def get_stem(job_id: str, stem_name: str):
    """Serve a separated stem WAV file."""
    if job_id not in stem_dirs:
        return JSONResponse({"error": "Unknown job"}, status_code=404)
    if stem_name not in ("vocals", "drums", "bass", "other", "guitar", "piano"):
        return JSONResponse({"error": "Invalid stem"}, status_code=400)
    sep_base = os.path.join(stem_dirs[job_id]["path"], "separated")
    model_dirs = [d for d in os.listdir(sep_base) if os.path.isdir(os.path.join(sep_base, d))] if os.path.exists(sep_base) else []
    sep_dir = os.path.join(sep_base, model_dirs[0]) if model_dirs else sep_base
    subdirs = [d for d in os.listdir(sep_dir) if os.path.isdir(os.path.join(sep_dir, d))] if os.path.exists(sep_dir) else []
    if not subdirs:
        return JSONResponse({"error": "Stems not found"}, status_code=404)
    path = os.path.join(sep_dir, subdirs[0], f"{stem_name}.wav")
    if not os.path.exists(path):
        return JSONResponse({"error": "Stem not found"}, status_code=404)
    return FileResponse(path, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Stem temp-dir cleanup (remove entries older than 1 hour)
# ---------------------------------------------------------------------------

async def _cleanup_stem_dirs():
    while True:
        await asyncio.sleep(600)  # check every 10 minutes
        now = time.time()
        expired = [k for k, v in stem_dirs.items() if now - v["created"] > 3600]
        for k in expired:
            try:
                shutil.rmtree(stem_dirs[k]["path"], ignore_errors=True)
            except Exception:
                pass
            del stem_dirs[k]


@app.on_event("startup")
async def _start_cleanup():
    asyncio.create_task(_cleanup_stem_dirs())


# --- Static files (must be last so it doesn't shadow API routes) ---
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.endswith(('.js', '.css', '.html')):
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response

app.add_middleware(NoCacheMiddleware)
app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
