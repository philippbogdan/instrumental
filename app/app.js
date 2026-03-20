/* ── app.js ── Main orchestrator ─────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let audioCtx = null;
  let synth = null;
  let keyboard = null;
  let waveformViz = null;
  let dawView = null;
  let selectedFile = null;
  // mode removed - always use match-single with multi-note extraction
  let currentJobId = null;
  let ws = null;
  let completed = false;
  let player = null;
  let stemDisplay = null;
  let separationJobId = null;
  let separationStems = null;

  // DOM refs
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const matchBtn = document.getElementById('matchBtn');
  // mode toggle removed
  const fileInfo = document.getElementById('fileInfo');
  const fileName = document.getElementById('fileName');
  const clearFile = document.getElementById('clearFile');
  const progressSection = document.getElementById('progressSection');
  const progressFill = document.getElementById('progressFill');
  const progressPhase = document.getElementById('progressPhase');
  const evalCount = document.getElementById('evalCount');
  const currentLoss = document.getElementById('currentLoss');
  const elapsedTime = document.getElementById('elapsedTime');
  const resultSection = document.getElementById('resultSection');
  const waveformCanvas = document.getElementById('waveformCanvas');
  const keyboardContainer = document.getElementById('keyboardContainer');
  const dawSection = document.getElementById('dawSection');
  const dawContainer = document.getElementById('dawContainer');
  const dawPlayBtn = document.getElementById('dawPlayBtn');
  const searchSection = document.getElementById('searchSection');
  const searchInput = document.getElementById('searchInput');
  const searchBtn = document.getElementById('searchBtn');
  const searchResults = document.getElementById('searchResults');
  const playerSection = document.getElementById('playerSection');
  const stemSection = document.getElementById('stemSection');
  const stemContainer = document.getElementById('stemContainer');
  const dividerOr = document.getElementById('dividerOr');

  function formatETA(evals, totalEvals, elapsedSec) {
    if (evals < 10 || elapsedSec < 1) return '';
    const rate = evals / elapsedSec;
    const remaining = (totalEvals - evals) / rate;
    if (remaining < 60) return Math.round(remaining) + 's left';
    return Math.round(remaining / 60) + 'm ' + Math.round(remaining % 60) + 's left';
  }

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

    // Show audio player
    playerSection.classList.remove('hidden');
    if (player) player.destroy();
    player = new AudioPlayer(playerSection);
    player.load(URL.createObjectURL(file));
  }

  function resetFile() {
    selectedFile = null;
    fileName.textContent = '';
    fileInfo.classList.add('hidden');
    matchBtn.disabled = true;

    // Destroy player
    if (player) {
      player.destroy();
      player = null;
    }
    playerSection.classList.add('hidden');

    // Reset separation state
    separationJobId = null;
    separationStems = null;
    if (stemDisplay) {
      stemDisplay.destroy();
      stemDisplay = null;
    }
    stemSection.classList.add('hidden');
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

  // ── Deezer search ─────────────────────────────────────────────────

  async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    try {
      const resp = await fetch('/api/search?q=' + encodeURIComponent(query));
      if (!resp.ok) throw new Error('Search failed');
      const json = await resp.json();
      const tracks = json.data || [];

      if (tracks.length === 0) {
        searchResults.innerHTML = '<div class="search-no-results">No results found</div>';
        searchResults.classList.remove('hidden');
        return;
      }

      searchResults.innerHTML = '';
      for (const track of tracks.slice(0, 10)) {
        const item = document.createElement('div');
        item.className = 'search-result-item';

        const img = document.createElement('img');
        img.src = track.album && track.album.cover_small ? track.album.cover_small : '';
        img.alt = '';

        const info = document.createElement('div');
        info.className = 'result-info';

        const title = document.createElement('span');
        title.className = 'result-title';
        title.textContent = track.title || 'Unknown';

        const artist = document.createElement('span');
        artist.className = 'result-artist';
        artist.textContent = track.artist ? track.artist.name : 'Unknown';

        info.appendChild(title);
        info.appendChild(artist);
        item.appendChild(img);
        item.appendChild(info);
        searchResults.appendChild(item);

        item.addEventListener('click', () => selectSearchResult(track));
      }

      searchResults.classList.remove('hidden');
    } catch (err) {
      console.error('Search error:', err);
    }
  }

  async function selectSearchResult(track) {
    searchResults.classList.add('hidden');
    searchResults.innerHTML = '';

    if (!track.preview) {
      console.error('No preview URL for track');
      return;
    }

    try {
      const resp = await fetch('/api/preview?url=' + encodeURIComponent(track.preview));
      if (!resp.ok) throw new Error('Preview fetch failed');
      const blob = await resp.blob();
      const file = new File([blob], 'preview.mp3', { type: 'audio/mpeg' });
      handleFileSelected(file);
    } catch (err) {
      console.error('Preview fetch error:', err);
    }
  }

  searchBtn.addEventListener('click', performSearch);

  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') performSearch();
  });

  // ── Match button (multi-phase flow) ──────────────────────────────

  matchBtn.addEventListener('click', async () => {
    ensureAudioCtx();
    matchBtn.disabled = true;
    completed = false;

    // Stop audio playback
    if (player) player.pause();

    // Hide input sections
    dropZone.classList.add('hidden');
    searchSection.classList.add('hidden');
    playerSection.classList.add('hidden');
    if (dividerOr) dividerOr.classList.add('hidden');

    // Phase 1: Show skeleton stems while Demucs runs
    stemSection.classList.remove('hidden');
    stemContainer.innerHTML = '';
    for (const name of ['vocals', 'drums', 'bass', 'other']) {
      const row = document.createElement('div');
      row.className = 'stem-row loading';
      const label = document.createElement('span');
      label.className = 'stem-label';
      label.textContent = name;
      const wave = document.createElement('div');
      wave.className = 'stem-waveform';
      row.appendChild(label);
      row.appendChild(wave);
      stemContainer.appendChild(row);
    }

    let useStem = false;

    try {
      const sepFormData = new FormData();
      sepFormData.append('file', selectedFile);
      const sepResp = await fetch('/api/separate', { method: 'POST', body: sepFormData });

      if (sepResp.ok) {
        const sepJson = await sepResp.json();
        separationJobId = sepJson.job_id;
        separationStems = sepJson.stems;

        // Phase 2: Show stems for selection
        progressSection.classList.add('hidden');
        stemSection.classList.remove('hidden');

        if (stemDisplay) stemDisplay.destroy();
        stemDisplay = new StemDisplay(stemContainer, audioCtx);
        stemDisplay.setStems(separationStems);
        // Default selection is "other" (StemDisplay does this by default)

        // Auto-start matching after 2 seconds
        useStem = true;
        await new Promise(resolve => setTimeout(resolve, 2000));
      } else {
        console.warn('Separation failed, falling back to direct match');
      }
    } catch (err) {
      console.warn('Separation error, falling back to direct match:', err);
    }

    // Phase 3: Start matching
    stemSection.classList.add('hidden');
    progressSection.classList.remove('hidden');
    progressPhase.textContent = 'Optimizing...';
    progressFill.style.width = '0%';
    evalCount.textContent = '0 / 20,000';
    currentLoss.textContent = 'Loss: --';
    elapsedTime.textContent = '0s';

    const endpoint = '/api/match-single';

    const formData = new FormData();
    formData.append('file', selectedFile);

    if (useStem && separationJobId) {
      formData.append('stem_job_id', separationJobId);
      formData.append('stem_name', stemDisplay ? stemDisplay.getSelectedStem() : 'other');
    }

    try {
      const resp = await fetch(endpoint, { method: 'POST', body: formData });
      if (!resp.ok) throw new Error('Upload failed: ' + resp.statusText);
      const json = await resp.json();
      currentJobId = json.job_id;
    } catch (err) {
      matchBtn.disabled = false;
      progressSection.classList.add('hidden');
      dropZone.classList.remove('hidden');
      searchSection.classList.remove('hidden');
      if (dividerOr) dividerOr.classList.remove('hidden');
      console.error('Upload error:', err);
      return;
    }

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
          const eta = formatETA(data.evals, data.total_evals, data.elapsed_seconds);
          elapsedTime.textContent = Math.round(data.elapsed_seconds) + 's' + (eta ? ' (' + eta + ')' : '');
        }
      } else if (data.type === 'complete') {
        completed = true;
        handleComplete(data);
      } else if (data.type === 'error') {
        console.error('Job error:', data.message);
        progressSection.classList.add('hidden');
        dropZone.classList.remove('hidden');
        searchSection.classList.remove('hidden');
        if (dividerOr) dividerOr.classList.remove('hidden');
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
      const resp = await fetch('/api/job-result/' + jobId);
      const data = await resp.json();
      if (data.type === 'complete' || data.params) {
        completed = true;
        handleComplete(data);
        return;
      }
      if (data.type === 'progress') {
        const pct = (data.evals / data.total_evals * 100).toFixed(1);
        progressFill.style.width = pct + '%';
        evalCount.textContent = data.evals.toLocaleString() + ' / ' + data.total_evals.toLocaleString();
        if (data.best_loss !== undefined) {
          currentLoss.textContent = 'Loss: ' + data.best_loss.toFixed(4);
        }
        if (data.elapsed_seconds !== undefined) {
          const eta = formatETA(data.evals, data.total_evals, data.elapsed_seconds);
          elapsedTime.textContent = Math.round(data.elapsed_seconds) + 's' + (eta ? ' (' + eta + ')' : '');
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
    // Export to Vital button
    const exportBtn = document.createElement('button');
    exportBtn.className = 'export-btn';
    exportBtn.title = 'Export to Vital synth (.vital)';
    exportBtn.textContent = 'Export to Vital';
    exportBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const currentParams = synth._params ? Array.from(synth._params) : data.params;
      const resp = await fetch('/api/export/vital', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params: currentParams })
      });
      if (resp.ok) {
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'INSTRUMENTAL_Match.vital';
        a.click();
        URL.revokeObjectURL(url);
      }
    });

    keyboardContainer.style.position = 'relative';
    keyboardContainer.appendChild(presetBtn);
    keyboardContainer.appendChild(exportBtn);

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
