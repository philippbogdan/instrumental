"""FastAPI backend for INSTRUMENTAL — inverse synthesis web app."""

import sys
import os
import io
import uuid
import time
import asyncio
import json
import multiprocessing
from typing import Dict, Any, Optional

import numpy as np
import torch
import librosa
import soundfile as sf
import cma

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

# Ensure multiprocessing uses fork to avoid worker spawn overhead
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


def _detect_pitch(audio_np: np.ndarray, sr: int = 44100) -> float:
    """Detect fundamental frequency using librosa.pyin."""
    f0, voiced_flag, _ = librosa.pyin(
        audio_np,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) > 0:
        return float(np.median(voiced_f0))
    return 440.0


# Module-level globals for multiprocessing workers (set before pool.map)
_WORKER_TARGET_NOTES = []


def _evaluate_single(params_np):
    """Evaluate a single candidate. Runs in a worker process."""
    synth = SynthPatch()
    loss_fn = get_loss("matching")
    params = torch.tensor(params_np, dtype=torch.float32)
    total = 0.0
    for audio_np, freq, dur in _WORKER_TARGET_NOTES:
        target = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        gen = synth.render(params, f0_hz=freq, duration=dur, note_duration=dur * 0.9)
        ml = min(target.shape[1], gen.shape[1])
        loss = loss_fn(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
        total += loss.item()
    return total / len(_WORKER_TARGET_NOTES)


def _init_worker(target_notes):
    """Initializer for pool workers: set the shared target notes."""
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
    """
    Run CMA-ES optimization (blocking) with multiprocessing.
    Pushes progress to asyncio.Queue.

    target_notes: list of (audio_np, freq, duration) tuples
    """
    sigma0 = 0.3
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    popsize = max(16, n_cores * 2)  # at least 2 candidates per core

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        "bounds": [[0] * N_PARAMS, [1] * N_PARAMS],
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

            # Push progress every generation
            elapsed = time.time() - t0
            msg = {
                "type": "progress",
                "evals": evals,
                "total_evals": n_evals,
                "best_loss": round(best_loss, 4),
                "elapsed_seconds": round(elapsed, 1),
            }
            loop.call_soon_threadsafe(queue.put_nowait, msg)

    # Build param_defs for frontend
    param_defs = [{"name": name, "lo": lo, "hi": hi} for name, lo, hi in PARAM_DEFS]
    best_params = list(es.result.xbest)

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
    file: UploadFile = File(...),
    n_evals: int = Form(10000),
):
    audio_bytes = await file.read()
    audio_np, sr = sf.read(io.BytesIO(audio_bytes))
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    # Resample to 44100 if needed
    if sr != 44100:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=44100)
        sr = 44100

    # Detect pitch
    f0 = _detect_pitch(audio_np, sr)
    dur = len(audio_np) / sr

    # Spectral init
    init_15 = spectral_init(audio_np, sr)
    x0 = _pad_spectral_init(init_15).numpy()

    # Target notes: use full audio as single note
    target_notes = [(audio_np.astype(np.float32), f0, dur)]

    job_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    jobs[job_id] = {"queue": queue, "status": "running"}

    loop = asyncio.get_event_loop()
    asyncio.ensure_future(
        asyncio.to_thread(_run_cmaes, target_notes, x0, n_evals, queue, loop, None)
    )

    return {"job_id": job_id, "status": "started"}


@app.post("/api/match-sequence")
async def match_sequence(
    file: UploadFile = File(...),
    n_evals: int = Form(10000),
):
    audio_bytes = await file.read()
    audio_np, sr = sf.read(io.BytesIO(audio_bytes))
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
    jobs[job_id] = {"queue": queue, "status": "running"}

    loop = asyncio.get_event_loop()
    asyncio.ensure_future(
        asyncio.to_thread(_run_cmaes, representative, x0, n_evals, queue, loop, note_data)
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
                break
    except WebSocketDisconnect:
        pass


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


# --- Static files (must be last so it doesn't shadow API routes) ---
app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
