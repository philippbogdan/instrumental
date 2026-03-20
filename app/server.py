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


def _extract_notes(audio_np: np.ndarray, sr: int = 44100, max_notes: int = 3, max_note_dur: float = 0.5):
    """
    Extract individual notes from audio via onset detection + pitch tracking.
    Returns list of (audio_segment, freq_hz, duration) tuples for CMA-ES multi-pitch fitting.

    Addresses failure scenarios:
    - F1: Uses delta=0.07 for onset sensitivity (tuned for Demucs stems with reverb bleed)
    - F2: Skips notes shorter than 50ms and notes where pitch detection returns NaN
    - F3: Falls back to 1-second chunk at median pitch if no notes found
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
        return [(chunk, f0, len(chunk) / sr)]

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

        if len(seg_audio) < int(0.05 * sr):
            continue

        # Detect pitch for this segment using pyin
        f0_arr, voiced, _ = librosa.pyin(
            seg_audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        # F2: Filter NaN and get median of voiced frames
        voiced_f0 = f0_arr[voiced] if voiced is not None else np.array([])
        if len(voiced_f0) == 0:
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
        return [(chunk, f0, len(chunk) / sr)]

    # Filter out likely sub-harmonics: if most notes are above 150Hz, remove notes below 100Hz
    freqs = [n["freq"] for n in notes]
    median_freq = float(np.median(freqs))
    if median_freq > 150:
        notes = [n for n in notes if n["freq"] > 100]
    if len(notes) == 0:
        chunk = audio_np[:min(len(audio_np), sr)]
        f0 = _detect_pitch(chunk, sr)
        return [(chunk, f0, len(chunk) / sr)]

    # Select up to max_notes representative notes (one per unique semitone, prefer longer notes)
    seen_semitones = {}
    for nd in sorted(notes, key=lambda n: n["duration"], reverse=True):  # longest first
        semitone = round(12 * np.log2(nd["freq"] / 440.0))
        if semitone not in seen_semitones and len(seen_semitones) < max_notes:
            seen_semitones[semitone] = nd

    representative = [
        (nd["audio"], nd["freq"], nd["duration"])
        for nd in seen_semitones.values()
    ]

    print(f"[_extract_notes] Extracted {len(notes)} notes, selected {len(representative)} representatives "
          f"at pitches: {[f'{r[1]:.0f}Hz' for r in representative]}")
    return representative


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
    file: UploadFile = File(...),
    n_evals: int = Form(20000),
    stem_job_id: Optional[str] = Form(None),
    stem_name: Optional[str] = Form(None),
):
    # If stem params provided, load stem WAV instead of uploaded file
    if stem_job_id and stem_name and stem_job_id in stem_dirs:
        sep_dir = os.path.join(stem_dirs[stem_job_id]["path"], "separated", "htdemucs")
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

    # Cap audio to 10 seconds for note extraction (not 5s - need enough for onset detection)
    max_samples = 10 * sr
    if len(audio_np) > max_samples:
        audio_np = audio_np[:max_samples]

    # Extract individual notes via onset detection + pitch tracking
    # This is multi-pitch fitting as described in the paper (Section 2.3)
    target_notes = _extract_notes(audio_np, sr, max_notes=3, max_note_dur=0.5)

    # Spectral init from first representative note
    init_15 = spectral_init(target_notes[0][0], sr)
    x0 = _pad_spectral_init(init_15).numpy()

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
    n_evals: int = Form(20000),
    stem_job_id: Optional[str] = Form(None),
    stem_name: Optional[str] = Form(None),
):
    # If stem params provided, load stem WAV instead of uploaded file
    if stem_job_id and stem_name and stem_job_id in stem_dirs:
        sep_dir = os.path.join(stem_dirs[stem_job_id]["path"], "separated", "htdemucs")
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
# Demucs source separation
# ---------------------------------------------------------------------------

@app.post("/api/separate")
async def separate_stems(file: UploadFile = File(...)):
    """Run Demucs htdemucs source separation on uploaded audio."""
    job_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp(prefix="instrumental_")
    # Preserve original extension so Demucs/ffmpeg handles format correctly
    ext = os.path.splitext(file.filename or "input.wav")[1] or ".wav"
    input_path = os.path.join(tmp_dir, "input" + ext)

    audio_bytes = await file.read()
    with open(input_path, "wb") as f:
        f.write(audio_bytes)

    def run_demucs():
        out_dir = os.path.join(tmp_dir, "separated")
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(
            [
                sys.executable, "-m", "demucs",
                "-n", "htdemucs",
                "-d", "cpu",
                "--float32",
                "-o", out_dir,
                input_path,
            ],
            check=True,
            capture_output=True,
        )
        # Output lands in: {out_dir}/htdemucs/{input_basename}/{stem}.wav
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        stems_dir = os.path.join(out_dir, "htdemucs", input_basename)
        stems = {}
        for stem_name in ("vocals", "drums", "bass", "other"):
            stem_path = os.path.join(stems_dir, f"{stem_name}.wav")
            if os.path.exists(stem_path):
                stems[stem_name] = f"/api/stem/{job_id}/{stem_name}.wav"
        return stems

    stems = await asyncio.to_thread(run_demucs)
    stem_dirs[job_id] = {"path": tmp_dir, "created": time.time()}
    return {"job_id": job_id, "stems": stems}


@app.get("/api/stem/{job_id}/{stem_name}.wav")
async def get_stem(job_id: str, stem_name: str):
    """Serve a separated stem WAV file."""
    if job_id not in stem_dirs:
        return JSONResponse({"error": "Unknown job"}, status_code=404)
    if stem_name not in ("vocals", "drums", "bass", "other"):
        return JSONResponse({"error": "Invalid stem"}, status_code=400)
    sep_dir = os.path.join(stem_dirs[job_id]["path"], "separated", "htdemucs")
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
app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
