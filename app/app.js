/* ── app.js ── Main orchestrator ─────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let audioCtx = null;
  let synth = null;
  let keyboard = null;
  let waveformViz = null;
  let dawView = null;
  let selectedFile = null;
  let mode = 'single';
  let currentJobId = null;
  let ws = null;
  let completed = false;

  // DOM refs
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const matchBtn = document.getElementById('matchBtn');
  const modeSingle = document.getElementById('modeSingle');
  const modeSequence = document.getElementById('modeSequence');
  const fileInfo = document.getElementById('fileInfo');
  const fileName = document.getElementById('fileName');
  const clearFile = document.getElementById('clearFile');
  const progressSection = document.getElementById('progressSection');
  const progressFill = document.getElementById('progressFill');
  const evalCount = document.getElementById('evalCount');
  const currentLoss = document.getElementById('currentLoss');
  const elapsedTime = document.getElementById('elapsedTime');
  const resultSection = document.getElementById('resultSection');
  const waveformCanvas = document.getElementById('waveformCanvas');
  const keyboardContainer = document.getElementById('keyboardContainer');
  const dawSection = document.getElementById('dawSection');
  const dawContainer = document.getElementById('dawContainer');
  const dawPlayBtn = document.getElementById('dawPlayBtn');

  // ── Helper: lazy AudioContext + Synth init ──────────────────────────

  function ensureAudioCtx() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      synth = new InstrumentalSynth(audioCtx);
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();
  }

  // ── File selection helpers ──────────────────────────────────────────

  function handleFileSelected(file) {
    if (!file) return;
    selectedFile = file;
    fileName.textContent = file.name;
    fileInfo.classList.remove('hidden');
    matchBtn.disabled = false;
  }

  function resetFile() {
    selectedFile = null;
    fileName.textContent = '';
    fileInfo.classList.add('hidden');
    matchBtn.disabled = true;
  }

  // ── Drop zone handlers ─────────────────────────────────────────────

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    handleFileSelected(file);
  });

  dropZone.addEventListener('click', (e) => {
    // Don't trigger file picker when clicking buttons inside drop zone
    if (e.target.closest('button')) return;
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    handleFileSelected(fileInput.files[0]);
  });

  clearFile.addEventListener('click', (e) => {
    e.stopPropagation();
    resetFile();
  });

  // ── Mode toggle ────────────────────────────────────────────────────

  modeSingle.addEventListener('click', () => {
    mode = 'single';
    modeSingle.classList.add('active');
    modeSequence.classList.remove('active');
  });

  modeSequence.addEventListener('click', () => {
    mode = 'sequence';
    modeSequence.classList.add('active');
    modeSingle.classList.remove('active');
  });

  // ── Match button ───────────────────────────────────────────────────

  matchBtn.addEventListener('click', async () => {
    ensureAudioCtx();
    matchBtn.disabled = true;
    completed = false;

    const formData = new FormData();
    formData.append('file', selectedFile);

    const endpoint = mode === 'single' ? '/api/match-single' : '/api/match-sequence';

    try {
      const resp = await fetch(endpoint, { method: 'POST', body: formData });
      if (!resp.ok) throw new Error('Upload failed: ' + resp.statusText);
      const json = await resp.json();
      currentJobId = json.job_id;
    } catch (err) {
      matchBtn.disabled = false;
      console.error('Upload error:', err);
      return;
    }

    // Show progress, hide drop zone
    progressSection.classList.remove('hidden');
    dropZone.classList.add('hidden');

    // Connect WebSocket for progress
    const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(wsProto + '//' + location.host + '/ws/progress/' + currentJobId);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'progress') {
        const pct = (data.evals / data.total_evals * 100).toFixed(1);
        progressFill.style.width = pct + '%';
        evalCount.textContent = data.evals.toLocaleString() + ' / ' + data.total_evals.toLocaleString();
        if (data.best_loss !== undefined) {
          currentLoss.textContent = 'Loss: ' + data.best_loss.toFixed(4);
        }
        if (data.elapsed_seconds !== undefined) {
          elapsedTime.textContent = Math.round(data.elapsed_seconds) + 's';
        }
      } else if (data.type === 'complete') {
        completed = true;
        handleComplete(data);
      } else if (data.type === 'error') {
        console.error('Job error:', data.message);
        progressSection.classList.add('hidden');
        dropZone.classList.remove('hidden');
        matchBtn.disabled = false;
      }
    };

    ws.onclose = () => {
      if (!completed) {
        // Fallback: poll for result
        pollForResult(currentJobId);
      }
    };
  });

  // ── Poll fallback ──────────────────────────────────────────────────

  async function pollForResult(jobId) {
    try {
      const resp = await fetch('/api/result/' + jobId);
      if (resp.ok) {
        const data = await resp.json();
        if (data.type === 'complete' || data.params) {
          handleComplete(data);
          return;
        }
      }
      // Retry after a delay
      setTimeout(() => pollForResult(jobId), 2000);
    } catch (err) {
      console.error('Poll error:', err);
      setTimeout(() => pollForResult(jobId), 2000);
    }
  }

  // ── Handle completion ──────────────────────────────────────────────

  function handleComplete(data) {
    if (ws) {
      ws.close();
      ws = null;
    }

    // Hide progress, show result
    progressSection.classList.add('hidden');
    resultSection.classList.remove('hidden');
    resultSection.classList.add('fadeIn');

    // Load synth params
    if (data.param_defs) synth.setParamDefs(data.param_defs);
    if (data.params) synth.setParams(data.params);

    // Waveform visualization — construct synchronously so canvas size is set
    // before keyboard renders, preventing a layout jump on the second rAF.
    waveformViz = new WaveformViz(waveformCanvas, {
      color: '#000000',
      bgColor: 'transparent'
    });
    synth.renderOffline(440, 1.0).then((buf) => {
      waveformViz.drawWaveform(buf);
    });

    // Keyboard
    if (keyboard) keyboard.destroy();
    keyboard = new PianoKeyboard(keyboardContainer, synth, {
      startOctave: 3,
      numOctaves: 2
    });
    keyboard.render();

    // Preset button (bottom-right of keyboard)
    const presetBtn = document.createElement('button');
    presetBtn.className = 'preset-btn';
    presetBtn.title = 'Load best match (loss 2.09)';
    presetBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/><polyline points="12,6 12,12 16,14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    presetBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const resp = await fetch('/api/preset/best');
      const data = await resp.json();
      synth.setParamDefs(data.param_defs);
      synth.setParams(data.params);
      presetBtn.classList.add('active');
      // Re-render waveform
      synth.renderOffline(440, 1.0).then((buf) => {
        if (waveformViz) waveformViz.drawWaveform(buf);
      });
    });
    keyboardContainer.style.position = 'relative';
    keyboardContainer.appendChild(presetBtn);

    // DAW (sequence mode)
    if (data.notes && data.notes.length > 0) {
      dawSection.classList.remove('hidden');
      dawView = new PianoRollDAW(dawContainer);
      dawView.setNotes(data.notes);
    }
  }

  // ── DAW play/stop toggle ───────────────────────────────────────────

  dawPlayBtn.addEventListener('click', () => {
    ensureAudioCtx();
    if (!dawView) return;

    if (dawView.isPlaying()) {
      dawView.stop();
      dawPlayBtn.textContent = 'Play Sequence';
    } else {
      dawView.play(synth);
      dawPlayBtn.textContent = 'Stop';
    }
  });
});
