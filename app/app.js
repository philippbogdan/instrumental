/* ── app.js ── Main orchestrator ─────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
  // Analytics helper
  function _track(type, data) {
    fetch('/api/event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type, data: data || {} })
    }).catch(() => {});
  }
  _track('visit');

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
  let noteJobId = null;
  let allNotesForPlayback = [];
  let analyzedDuration = 10;
  let selectedNoteForMatch = 0;
  let separationStems = null;
  let isWildGround = false;
  let currentSongName = '';
  let _activeAudios = []; // track playing audio elements for cleanup

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
  const searchSection = document.getElementById('wildSection');
  const searchInput = document.getElementById('searchInput');
  const searchBtn = document.getElementById('searchBtn');
  const searchResults = document.getElementById('searchResults');
  const playerSection = document.getElementById('playerSection');
  const stemSection = document.getElementById('stemSection');
  const stemContainer = document.getElementById('stemContainer');
  const dividerOr = document.getElementById('dividerOr');
  const stemCount = 4;

  // ── Demo card click handler ─────────────────────────────────────────
  document.querySelectorAll('.demo-card').forEach(card => {
    card.addEventListener('click', async () => {
      const demoId = card.dataset.demo;
      if (!demoId) return;

      // Stop any playing audio
      _activeAudios.forEach(a => { a.pause(); a.src = ''; });
      _activeAudios = [];
      _track('cherrypick', { song: demoId });

      // Highlight active card
      document.querySelectorAll('.demo-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');

      // Hide other sections, show result
      if (dividerOr) dividerOr.classList.add('hidden');
      dropZone.classList.add('hidden');
      if (playerSection) playerSection.classList.add('hidden');
      if (stemSection) stemSection.classList.add('hidden');
      progressSection.classList.add('hidden');
      const wildSec = document.getElementById('wildSection');
      if (wildSec) wildSec.classList.add('hidden');
      resultSection.classList.remove('hidden');

      // Load pre-baked demo data
      const notesResp = await fetch('/demos/' + demoId + '_notes.json');
      const demoData = await notesResp.json();

      // Initialize synth with matched params
      ensureAudioCtx();
      if (!synth) synth = new InstrumentalSynth(audioCtx);
      synth.setParams(demoData.params);

      allNotesForPlayback = demoData.notes;
      analyzedDuration = demoData.duration;

      // Build result UI with pre-baked audio
      handleDemoResult(demoId, demoData);
    });
  });

  function handleDemoResult(demoId, demoData) {
    const container = resultSection;
    container.innerHTML = '';

    // Audio elements
    const cacheBust = '?v=21';
    const origAudio = new Audio('/demos/' + demoId + '_original.wav' + cacheBust);
    const matchAudio = new Audio('/demos/' + demoId + '_matched.wav' + cacheBust);
    _activeAudios = [origAudio, matchAudio];
    let masterDur = demoData.duration;

    // Track rows
    function makeTrackRow(label, audio) {
      const row = document.createElement('div');
      row.className = 'ab-track';
      row.innerHTML =
        '<span class="ab-label">' + label + '</span>' +
        '<div class="ab-bar"><div class="ab-bar-fill"></div></div>' +
        '<input type="range" class="ab-volume" min="0" max="100" value="100">';
      row.querySelector('.ab-volume').addEventListener('input', function() {
        audio.volume = this.value / 100;
      });
      return row;
    }

    container.appendChild(makeTrackRow('Original', origAudio));
    const matchRow = makeTrackRow('Matched', matchAudio);
    container.appendChild(matchRow);

    // Transport
    const transport = document.createElement('div');
    transport.className = 'ab-transport';
    transport.innerHTML =
      '<button class="ab-play-btn">Play</button>' +
      '<div class="ab-progress"><div class="ab-progress-fill"></div></div>' +
      '<span class="ab-time">0:00 / ' + fmtTime(masterDur) + '</span>';
    container.appendChild(transport);

    const playBtn = transport.querySelector('.ab-play-btn');
    const progressBar = transport.querySelector('.ab-progress');
    const progressFill = transport.querySelector('.ab-progress-fill');
    const timeDisplay = transport.querySelector('.ab-time');
    let isPlaying = false;

    playBtn.addEventListener('click', () => {
      if (isPlaying) {
        origAudio.pause(); matchAudio.pause();
        playBtn.textContent = 'Play';
        isPlaying = false;
        clearDemoHighlights();
      } else {
        origAudio.play(); matchAudio.play();
        playBtn.textContent = '||';
        isPlaying = true;
        _track('play', { demo: demoId });
        scheduleDemoHighlights(matchAudio);
      }
    });

    progressBar.addEventListener('click', (e) => {
      const frac = e.offsetX / progressBar.offsetWidth;
      const t = frac * masterDur;
      origAudio.currentTime = Math.min(t, origAudio.duration || t);
      matchAudio.currentTime = Math.min(t, matchAudio.duration || t);
      clearDemoHighlights();
      if (isPlaying) scheduleDemoHighlights(matchAudio);
    });

    origAudio.addEventListener('timeupdate', () => {
      if (!masterDur) return;
      if (origAudio.currentTime >= masterDur) {
        origAudio.pause(); matchAudio.pause();
        origAudio.currentTime = 0; matchAudio.currentTime = 0;
        playBtn.textContent = 'Play';
        isPlaying = false;
        progressFill.style.width = '0%';
        timeDisplay.textContent = '0:00 / ' + fmtTime(masterDur);
        clearDemoHighlights();
        return;
      }
      const pct = (origAudio.currentTime / masterDur) * 100;
      progressFill.style.width = Math.min(pct, 100) + '%';
      timeDisplay.textContent = fmtTime(origAudio.currentTime) + ' / ' + fmtTime(masterDur);
    });

    origAudio.addEventListener('loadedmetadata', () => {
      masterDur = Math.min(demoData.duration, origAudio.duration);
      timeDisplay.textContent = '0:00 / ' + fmtTime(masterDur);
    });

    // Keyboard
    const kbContainer = document.createElement('div');
    kbContainer.className = 'keyboard-container';
    container.appendChild(kbContainer);

    if (keyboard) keyboard.destroy();
    keyboard = new PianoKeyboard(kbContainer, synth, { startOctave: 2, numOctaves: 4 });
    keyboard.render();
    kbContainer.style.position = 'relative';
    // No scroll centering needed — keyboard fits in view

    // Export button
    const exportBtn = document.createElement('button');
    exportBtn.className = 'export-btn';
    exportBtn.textContent = 'Export to Vital';
    exportBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const resp = await fetch('/api/export/vital', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params: demoData.params })
      });
      if (resp.ok) {
        const blob = await resp.blob();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'INSTRUMENTAL_Match.vital';
        a.click();
        _track('export_vital');
      }
    });
    kbContainer.appendChild(exportBtn);

    // Keyboard animation
    let demoHighlights = [];
    function scheduleDemoHighlights(audioEl) {
      clearDemoHighlights();
      const startTime = audioEl.currentTime;
      for (const note of demoData.notes) {
        const onDelay = Math.max(0, (note.onset - startTime) * 1000);
        const offDelay = Math.max(0, (note.onset + note.duration - startTime) * 1000);
        const midi = note.midi || Math.round(12 * Math.log2(note.freq / 440) + 69);
        demoHighlights.push(setTimeout(() => keyboard.setActive(midi, true), onDelay));
        demoHighlights.push(setTimeout(() => keyboard.setActive(midi, false), offDelay));
      }
    }
    function clearDemoHighlights() {
      demoHighlights.forEach(t => clearTimeout(t));
      demoHighlights = [];
      if (keyboard && keyboard._keys) {
        keyboard._keys.forEach((el, midi) => keyboard.setActive(midi, false));
      }
    }
  }

  function fmtTime(s) {
    const m = Math.floor(s / 60);
    return m + ':' + String(Math.floor(s % 60)).padStart(2, '0');
  }

  function formatETA(evals, totalEvals, elapsedSec) {
    if (evals < 10 || elapsedSec < 1) return '';
    const rate = evals / elapsedSec;
    const remaining = (totalEvals - evals) / rate;
    if (remaining < 60) return Math.round(remaining) + 's left';
    return Math.round(remaining / 60) + 'm ' + Math.round(remaining % 60) + 's left';
  }

  // ── Live knobs during optimization ──────────────────────────────────

  // Which params to show as knobs (indices into PARAM_DEFS, most interesting ones)
  const KNOB_PARAMS = [
    { idx: 0, name: 'Saw' },
    { idx: 1, name: 'Square' },
    { idx: 5, name: 'Cutoff' },
    { idx: 6, name: 'Res' },
    { idx: 9, name: 'Sustain' },
    { idx: 11, name: 'Gain' },
  ];

  function _initKnobs() {
    const container = document.getElementById('knobsContainer');
    if (!container) return;
    container.innerHTML = '';
    for (const kp of KNOB_PARAMS) {
      const wrapper = document.createElement('div');
      wrapper.className = 'knob-wrapper';
      wrapper.innerHTML =
        '<div class="knob" data-knob-idx="' + kp.idx + '">' +
        '<div class="knob-indicator"></div>' +
        '</div>' +
        '<div class="knob-label">' + kp.name + '</div>' +
        '<div class="knob-value" data-knob-val="' + kp.idx + '">--</div>';
      container.appendChild(wrapper);
    }
  }

  function _updateKnobs(params) {
    if (!params || params.length === 0) return;
    for (const kp of KNOB_PARAMS) {
      const val = params[kp.idx] || 0; // normalized 0-1
      // Rotate indicator: 0 = -135deg (7 o'clock), 1 = +135deg (5 o'clock)
      const deg = -135 + val * 270;
      const indicator = document.querySelector('.knob[data-knob-idx="' + kp.idx + '"] .knob-indicator');
      if (indicator) indicator.style.transform = 'rotate(' + deg + 'deg)';
      const valEl = document.querySelector('[data-knob-val="' + kp.idx + '"]');
      if (valEl) valEl.textContent = val.toFixed(2);
    }
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
    _track('search', { query });

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
    isWildGround = true;
    _track('search_select', { title: track.title, artist: track.artist?.name });
    currentSongName = (track.artist?.name || '') + ' - ' + (track.title || 'Unknown');
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
    playerSection.classList.add('hidden');
    if (dividerOr) dividerOr.classList.add('hidden');
    const demoSec = document.getElementById('demoSection');
    const wildSec = document.getElementById('wildSection');
    if (demoSec) demoSec.classList.add('hidden');
    if (wildSec) wildSec.classList.add('hidden');

    // Phase 1: Show skeleton stems with ETA while Demucs runs
    stemSection.classList.remove('hidden');
    stemContainer.innerHTML = '';

    // ETA label
    const etaLabel = document.createElement('div');
    etaLabel.className = 'progress-phase';
    etaLabel.style.marginBottom = '8px';
    etaLabel.textContent = 'Separating stems...';
    stemContainer.appendChild(etaLabel);

    const stemNames = stemCount === 6
      ? ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']
      : ['vocals', 'drums', 'bass', 'other'];
    for (const name of stemNames) {
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

    // ETA countdown timer (~2s per second of audio for htdemucs_ft)
    const etaStart = Date.now();
    // Dynamic ETA: htdemucs_ft processes at ~2x realtime on Mini
    // Get audio duration from player if available, else estimate from file size
    let audioDuration = 30; // fallback
    if (player && player.audio && player.audio.duration && isFinite(player.audio.duration)) {
      audioDuration = player.audio.duration;
    }
    const estimatedSec = Math.round(audioDuration * 2); // ~2x realtime for htdemucs_ft
    const etaInterval = setInterval(() => {
      const elapsed = Math.round((Date.now() - etaStart) / 1000);
      const remaining = Math.max(0, estimatedSec - elapsed);
      if (remaining > 0) {
        etaLabel.textContent = 'Separating stems... ~' + remaining + 's';
      } else {
        etaLabel.textContent = 'Separating stems... ' + elapsed + 's (finishing up)';
      }
    }, 1000);

    let useStem = false;

    try {
      const sepFormData = new FormData();
      sepFormData.append('file', selectedFile);
      sepFormData.append('stem_count', stemCount.toString());
      console.log('[INSTRUMENTAL] Sending separation request...');
      const sepResp = await fetch('/api/separate', { method: 'POST', body: sepFormData });
      clearInterval(etaInterval);
      console.log('[INSTRUMENTAL] Separation response:', sepResp.status, sepResp.ok);

      if (sepResp.ok) {
        const sepJson = await sepResp.json();
        console.log('[INSTRUMENTAL] Separation result:', JSON.stringify(sepJson).substring(0, 200));
        separationJobId = sepJson.job_id;
        separationStems = sepJson.stems;

        // Phase 2: Show stems for user to select, with a Start button
        progressSection.classList.add('hidden');
        stemSection.classList.remove('hidden');

        if (stemDisplay) stemDisplay.destroy();
        stemDisplay = new StemDisplay(stemContainer, audioCtx);
        stemDisplay.songName = currentSongName;
        stemDisplay.setStems(separationStems);

        // Add "Start Matching" button below stems
        let startBtn = stemContainer.querySelector('.stem-start-btn');
        if (!startBtn) {
          startBtn = document.createElement('button');
          startBtn.className = 'match-btn stem-start-btn';
          startBtn.textContent = 'Match Selected Stem';
          startBtn.style.marginTop = '12px';
          stemContainer.appendChild(startBtn);
        }

        // Wait for user to click "Match Selected Stem"
        useStem = true;
        await new Promise(resolve => {
          startBtn.addEventListener('click', () => {
            if (stemDisplay) stemDisplay.stopPlayback();
            resolve();
          }, { once: true });
        });

        // Phase 2.5: Extract notes from selected stem and show preview
        stemSection.classList.add('hidden');
        progressSection.classList.remove('hidden');
        progressPhase.textContent = 'Extracting notes...';
        progressFill.style.width = '0%';
        evalCount.textContent = '';
        currentLoss.textContent = '';
        elapsedTime.textContent = '';

        const selectedStem = stemDisplay ? stemDisplay.getSelectedStem() : 'other';
        const noteFormData = new FormData();
        noteFormData.append('stem_job_id', separationJobId);
        noteFormData.append('stem_name', selectedStem);

        const noteResp = await fetch('/api/extract-notes', { method: 'POST', body: noteFormData });
        if (noteResp.ok) {
          const noteJson = await noteResp.json();
          noteJobId = noteJson.note_job_id;
          allNotesForPlayback = noteJson.all_notes || [];
          analyzedDuration = noteJson.analyzed_duration || 10;

          // Show extracted notes with play buttons
          progressSection.classList.add('hidden');
          stemSection.classList.remove('hidden');
          stemContainer.innerHTML = '';

          // Title - ask user to pick the cleanest note
          const noteTitle = document.createElement('div');
          noteTitle.className = 'progress-phase';
          noteTitle.style.marginBottom = '8px';
          noteTitle.textContent = noteJson.fallback
            ? 'No individual notes detected'
            : 'Which note sounds cleanest?';
          stemContainer.appendChild(noteTitle);

          let selectedNoteIndex = 0; // default to first (loudest)
          const noteRows = [];

          for (const note of noteJson.notes) {
            const row = document.createElement('div');
            row.className = 'stem-row loading' + (note.index === 0 ? ' selected' : '');

            const playBtn = document.createElement('button');
            playBtn.className = 'stem-play-btn';
            playBtn.textContent = '\u25B6';
            playBtn.disabled = true;

            const label = document.createElement('span');
            label.className = 'stem-label';
            label.textContent = note.name;

            const freqLabel = document.createElement('span');
            freqLabel.className = 'stem-label';
            freqLabel.style.width = '50px';
            freqLabel.style.opacity = '0.5';
            freqLabel.textContent = note.freq + 'Hz';

            const waveDiv = document.createElement('div');
            waveDiv.className = 'stem-waveform';
            const canvas = document.createElement('canvas');
            const playhead = document.createElement('div');
            playhead.className = 'stem-playhead';
            waveDiv.appendChild(canvas);
            waveDiv.appendChild(playhead);

            row.appendChild(playBtn);
            row.appendChild(label);
            row.appendChild(freqLabel);
            row.appendChild(waveDiv);
            stemContainer.appendChild(row);
            noteRows.push(row);

            // Click to select this note for optimization
            row.addEventListener('click', () => {
              noteRows.forEach(r => r.classList.remove('selected'));
              row.classList.add('selected');
              selectedNoteIndex = note.index;
            });

            // Load waveform async
            (async () => {
              try {
                const resp = await fetch(note.url);
                const buf = await resp.arrayBuffer();
                const audioBuf = await audioCtx.decodeAudioData(buf);
                const viz = new WaveformViz(canvas, {color: '#000000', bgColor: 'transparent', lineWidth: 1});
                const w = canvas.parentElement ? canvas.parentElement.clientWidth : 400;
                viz.setSize(w, 36);
                viz.drawWaveform(audioBuf);
                row.classList.remove('loading');
                playBtn.disabled = false;

                // Play button
                let playing = null;
                playBtn.addEventListener('click', (e) => {
                  e.stopPropagation();
                  if (playing) {
                    try { playing.stop(); } catch(_) {}
                    playBtn.textContent = '\u25B6';
                    playBtn.classList.remove('playing');
                    playhead.style.display = 'none';
                    playing = null;
                    return;
                  }
                  const src = audioCtx.createBufferSource();
                  src.buffer = audioBuf;
                  src.connect(audioCtx.destination);
                  src.start();
                  playBtn.textContent = '\u25A0';
                  playBtn.classList.add('playing');
                  playing = src;
                  const startT = audioCtx.currentTime;
                  const dur = audioBuf.duration;
                  const anim = () => {
                    if (!playing) return;
                    const pct = Math.min((audioCtx.currentTime - startT) / dur, 1);
                    playhead.style.left = (pct * 100) + '%';
                    playhead.style.display = 'block';
                    if (pct < 1) requestAnimationFrame(anim);
                  };
                  requestAnimationFrame(anim);
                  src.onended = () => {
                    playBtn.textContent = '\u25B6';
                    playBtn.classList.remove('playing');
                    playhead.style.display = 'none';
                    playing = null;
                  };
                });
              } catch(e) {
                row.classList.remove('loading');
                console.error('Note load failed:', e);
              }
            })();
          }

          // "Match This Note" button
          const optBtn = document.createElement('button');
          optBtn.className = 'match-btn';
          optBtn.textContent = 'Match This Note';
          optBtn.style.marginTop = '12px';
          stemContainer.appendChild(optBtn);

          await new Promise(resolve => {
            optBtn.addEventListener('click', resolve, { once: true });
          });

          // Store which note index was selected
          selectedNoteForMatch = selectedNoteIndex;
        }
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
    evalCount.textContent = '0 / 10,000';
    currentLoss.textContent = 'Loss: --';
    elapsedTime.textContent = '0s';
    _initKnobs();

    const endpoint = '/api/match-single';

    const formData = new FormData();
    formData.append('file', selectedFile);

    if (useStem && separationJobId) {
      formData.append('stem_job_id', separationJobId);
      formData.append('stem_name', stemDisplay ? stemDisplay.getSelectedStem() : 'other');
    }
    if (noteJobId) {
      formData.append('note_job_id', noteJobId);
      formData.append('selected_note', selectedNoteForMatch.toString());
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

      if (data.type === 'queued') {
        progressPhase.textContent = data.message || 'In queue...';
        progressFill.style.width = '0%';
        evalCount.textContent = '';
        currentLoss.textContent = '';
        elapsedTime.textContent = '';
      } else if (data.type === 'progress') {
        progressPhase.textContent = 'Optimizing...';
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
        if (data.params) _updateKnobs(data.params);
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
      if (data.type === 'complete') {
        completed = true;
        handleComplete(data);
        return;
      }
      if (data.type === 'queued') {
        progressPhase.textContent = data.message || 'In queue...';
        progressFill.style.width = '0%';
        evalCount.textContent = '';
        currentLoss.textContent = '';
        elapsedTime.textContent = '';
      } else if (data.type === 'progress') {
        progressPhase.textContent = 'Optimizing...';
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
        if (data.params) _updateKnobs(data.params);
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
    if (ws) { ws.close(); ws = null; }

    progressSection.classList.add('hidden');
    resultSection.classList.remove('hidden');

    // Load synth params
    if (data.param_defs) synth.setParamDefs(data.param_defs);
    if (data.params) synth.setParams(data.params);

    // Clear result section and rebuild
    const container = resultSection;
    container.innerHTML = '';

    // Audio elements for synced playback
    const origAudio = new Audio();
    const matchAudio = new Audio();
    let masterDur = analyzedDuration;
    let bothReady = 0;

    // --- Track rows ---
    function makeTrackRow(label, audio) {
      const row = document.createElement('div');
      row.className = 'ab-track';
      row.innerHTML =
        '<span class="ab-label">' + label + '</span>' +
        '<div class="ab-bar"><div class="ab-bar-fill"></div></div>' +
        '<input type="range" class="ab-volume" min="0" max="100" value="100">';
      const vol = row.querySelector('.ab-volume');
      vol.addEventListener('input', () => { audio.volume = vol.value / 100; });
      return row;
    }

    const origRow = makeTrackRow('Original', origAudio);
    container.appendChild(origRow);
    const matchRow = makeTrackRow('Matched', matchAudio);
    container.appendChild(matchRow);

    // --- Shared transport controls ---
    const transport = document.createElement('div');
    transport.className = 'ab-transport';
    transport.innerHTML =
      '<button class="ab-play-btn">Play</button>' +
      '<div class="ab-progress"><div class="ab-progress-fill"></div></div>' +
      '<span class="ab-time">0:00 / 0:00</span>';
    container.appendChild(transport);

    const playBtn = transport.querySelector('.ab-play-btn');
    const progressBar = transport.querySelector('.ab-progress');
    const progressFill = transport.querySelector('.ab-progress-fill');
    const timeDisplay = transport.querySelector('.ab-time');

    function fmtTime(s) {
      const m = Math.floor(s / 60);
      return m + ':' + String(Math.floor(s % 60)).padStart(2, '0');
    }

    function effectiveDur() {
      return masterDur;
    }

    // Play/pause both in sync
    let isPlaying = false;
    playBtn.addEventListener('click', () => {
      if (isPlaying) {
        origAudio.pause(); matchAudio.pause();
        playBtn.textContent = 'Play';
        isPlaying = false;
        clearKeyboardHighlights();
      } else {
        origAudio.play(); matchAudio.play();
        playBtn.textContent = '||';
        isPlaying = true;
        scheduleKeyboardHighlights(matchAudio);
      }
    });

    // Seek both
    progressBar.addEventListener('click', (e) => {
      const frac = e.offsetX / progressBar.offsetWidth;
      const t = frac * effectiveDur();
      origAudio.currentTime = Math.min(t, origAudio.duration || t);
      matchAudio.currentTime = Math.min(t, matchAudio.duration || t);
      clearKeyboardHighlights();
      if (isPlaying) scheduleKeyboardHighlights(matchAudio);
    });

    // Update shared progress from original (master)
    origAudio.addEventListener('timeupdate', () => {
      const dur = effectiveDur();
      if (!dur) return;
      if (origAudio.currentTime >= dur) {
        origAudio.pause(); matchAudio.pause();
        origAudio.currentTime = 0; matchAudio.currentTime = 0;
        playBtn.textContent = 'Play';
        isPlaying = false;
        progressFill.style.width = '0%';
        timeDisplay.textContent = '0:00 / ' + fmtTime(dur);
        clearKeyboardHighlights();
        return;
      }
      const pct = (origAudio.currentTime / dur) * 100;
      progressFill.style.width = Math.min(pct, 100) + '%';
      timeDisplay.textContent = fmtTime(origAudio.currentTime) + ' / ' + fmtTime(dur);

      // Update per-track progress bars
      const origFill = origRow.querySelector('.ab-bar-fill');
      origFill.style.width = Math.min(pct, 100) + '%';
      const matchDur = matchAudio.duration || dur;
      const matchPct = (matchAudio.currentTime / matchDur) * 100;
      const matchFill = matchRow.querySelector('.ab-bar-fill');
      matchFill.style.width = Math.min(matchPct, 100) + '%';
    });

    // Load audio
    if (separationJobId && stemDisplay) {
      const stemUrl = '/api/stem/' + separationJobId + '/' + (stemDisplay.getSelectedStem() || 'other') + '.wav';
      origAudio.src = stemUrl;
      origAudio.load();
      origAudio.addEventListener('loadedmetadata', () => {
        masterDur = Math.min(analyzedDuration, origAudio.duration);
        timeDisplay.textContent = '0:00 / ' + fmtTime(masterDur);
      });
    }

    if (noteJobId) {
      fetch('/api/render-sequence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params: data.params, note_job_id: noteJobId, total_duration: analyzedDuration })
      }).then(r => r.blob()).then(blob => {
        matchAudio.src = URL.createObjectURL(blob);
        matchAudio.load();
      });
    }

    // --- Keyboard ---
    const kbContainer = document.createElement('div');
    kbContainer.className = 'keyboard-container';
    container.appendChild(kbContainer);

    if (keyboard) keyboard.destroy();
    keyboard = new PianoKeyboard(kbContainer, synth, { startOctave: 2, numOctaves: 4 });
    keyboard.render();
    kbContainer.style.position = 'relative';
    // No scroll centering needed — keyboard fits in view

    // Export button
    const exportBtn = document.createElement('button');
    exportBtn.className = 'export-btn';
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
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'INSTRUMENTAL_Match.vital';
        a.click();
        _track('export_vital');
      }
    });
    kbContainer.appendChild(exportBtn);

    // --- Keyboard animation helpers ---
    let highlightTimeouts = [];

    function scheduleKeyboardHighlights(audioEl) {
      clearKeyboardHighlights();
      const startTime = audioEl.currentTime;
      for (const note of allNotesForPlayback) {
        const onDelay = Math.max(0, (note.onset - startTime) * 1000);
        const offDelay = Math.max(0, (note.onset + note.duration - startTime) * 1000);
        const midi = note.midi || round12(note.freq);
        highlightTimeouts.push(setTimeout(() => keyboard.setActive(midi, true), onDelay));
        highlightTimeouts.push(setTimeout(() => keyboard.setActive(midi, false), offDelay));
      }
    }

    function clearKeyboardHighlights() {
      highlightTimeouts.forEach(t => clearTimeout(t));
      highlightTimeouts = [];
      if (keyboard && keyboard._keys) {
        keyboard._keys.forEach((el, midi) => keyboard.setActive(midi, false));
      }
    }

    function round12(freq) {
      return Math.round(12 * Math.log2(freq / 440) + 69);
    }

    // Quality notice for wild ground runs
    if (isWildGround) {
      const notice = document.createElement('div');
      notice.className = 'quality-notice';
      notice.innerHTML =
        '<div class="quality-notice-header">' +
        '? Why does it sound different from the original?' +
        '</div>' +
        '<div class="quality-notice-body">' +
        'This method is extremely sensitive to the quality of the input notes. ' +
        'When we separate a full song into stems, even small amounts of bleed from ' +
        'vocals, drums, or bass contaminate the synth audio. The optimizer then tries ' +
        'to match this noise instead of the actual synth sound.' +
        '<br><br>' +
        'The cherrypicked examples on the landing page use clean, manually recorded ' +
        'synth samples — that\'s why they sound much closer to the original.' +
        '<br><br>' +
        'For best results, upload a clean recording of just the synth part.' +
        '</div>';
      notice.addEventListener('click', () => notice.classList.toggle('open'));
      container.appendChild(notice);
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
