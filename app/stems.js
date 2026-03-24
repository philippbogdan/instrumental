class StemDisplay {
  constructor(containerElement, audioContext) {
    this.container = containerElement;
    this.audioCtx = audioContext;
    this._selected = 'other';
    this._callback = null;
    this._rows = {};
    this._buffers = {};
    this._checked = new Set();
    this._playing = null;
    this._dlBar = null;
    this.songName = '';
  }

  _safeName() {
    return (this.songName || 'track').replace(/[^a-zA-Z0-9_\- ]/g, '').replace(/\s+/g, '_').toLowerCase();
  }

  setStems(stemUrls) {
    this.container.innerHTML = '';
    this.stopPlayback();
    this._checked.clear();
    const stemOrder = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'];

    // Header row
    const header = document.createElement('div');
    header.className = 'stem-header';
    header.innerHTML = '<div class="stem-header-spacer"></div><div class="stem-header-dl">\u2913</div>';
    this.container.appendChild(header);

    for (const name of stemOrder) {
      const url = stemUrls[name];
      if (!url) continue;

      const row = document.createElement('div');
      row.className = 'stem-row loading';

      const playBtn = document.createElement('button');
      playBtn.className = 'stem-play-btn';
      playBtn.textContent = '\u25B6';
      playBtn.title = 'Preview ' + name;
      playBtn.disabled = true;
      playBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this._togglePlay(name, playBtn);
      });

      const label = document.createElement('span');
      label.className = 'stem-label';
      label.textContent = name;

      const waveDiv = document.createElement('div');
      waveDiv.className = 'stem-waveform';
      const canvas = document.createElement('canvas');
      const playhead = document.createElement('div');
      playhead.className = 'stem-playhead';
      waveDiv.appendChild(canvas);
      waveDiv.appendChild(playhead);

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.className = 'stem-checkbox';
      cb.title = 'Select for download';
      cb.addEventListener('click', (e) => e.stopPropagation());
      cb.addEventListener('change', () => {
        if (cb.checked) {
          this._checked.add(name);
        } else {
          this._checked.delete(name);
        }
        this._updateDlBar();
      });

      row.appendChild(playBtn);
      row.appendChild(label);
      row.appendChild(waveDiv);

      const outer = document.createElement('div');
      outer.className = 'stem-outer';
      outer.appendChild(row);
      outer.appendChild(cb);
      this.container.appendChild(outer);

      this._rows[name] = { el: row, playBtn, waveDiv, playhead, canvas, url };

      row.addEventListener('click', () => {
        Object.values(this._rows).forEach(r => r.el.classList.remove('selected'));
        row.classList.add('selected');
        this._selected = name;
        if (this._callback) this._callback(name);
      });

      this._loadWaveform(url, canvas, name);
    }

    // Download bar (hidden initially)
    this._dlBar = document.createElement('div');
    this._dlBar.className = 'stem-dl-bar hidden';
    this._dlBar.innerHTML =
      '<span class="stem-dl-bar-label">Download selected stems</span>' +
      '<button class="stem-dl-bar-btn">Download .wav</button>';
    this._dlBar.querySelector('.stem-dl-bar-btn').addEventListener('click', () => this._downloadMix());
    this.container.appendChild(this._dlBar);
  }

  _updateDlBar() {
    if (!this._dlBar) return;
    if (this._checked.size > 0) {
      const names = Array.from(this._checked).join(' + ');
      this._dlBar.querySelector('.stem-dl-bar-label').textContent = names;
      this._dlBar.classList.remove('hidden');
    } else {
      this._dlBar.classList.add('hidden');
    }
  }

  async _downloadMix() {
    const names = Array.from(this._checked);
    if (names.length === 0) return;

    const base = this._safeName();

    // If only one stem, download directly
    if (names.length === 1) {
      const row = this._rows[names[0]];
      if (row && row.url) {
        const a = document.createElement('a');
        a.href = row.url;
        a.download = base + '_' + names[0] + '.wav';
        a.click();
      }
      return;
    }

    // Mix multiple stems client-side
    const buffers = names.map(n => this._buffers[n]).filter(Boolean);
    if (buffers.length === 0) return;

    const sr = buffers[0].sampleRate;
    const maxLen = Math.max(...buffers.map(b => b.length));
    const mixed = new Float32Array(maxLen);

    for (const buf of buffers) {
      const ch = buf.getChannelData(0);
      for (let i = 0; i < ch.length; i++) {
        mixed[i] += ch[i];
      }
    }

    // Normalize
    let peak = 0;
    for (let i = 0; i < mixed.length; i++) {
      const abs = Math.abs(mixed[i]);
      if (abs > peak) peak = abs;
    }
    if (peak > 0.99) {
      const scale = 0.95 / peak;
      for (let i = 0; i < mixed.length; i++) mixed[i] *= scale;
    }

    // Encode as WAV
    const wavBlob = this._encodeWav(mixed, sr);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(wavBlob);
    a.download = base + '_' + names.join('_') + '.wav';
    a.click();
  }

  _encodeWav(samples, sampleRate) {
    const len = samples.length;
    const buffer = new ArrayBuffer(44 + len * 2);
    const view = new DataView(buffer);
    const writeStr = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + len * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, len * 2, true);

    for (let i = 0; i < len; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
  }

  async _loadWaveform(url, canvas, name) {
    try {
      const resp = await fetch(url);
      const arrayBuf = await resp.arrayBuffer();
      const audioBuf = await this.audioCtx.decodeAudioData(arrayBuf);
      this._buffers[name] = audioBuf;
      const viz = new WaveformViz(canvas, {color: '#000000', bgColor: 'transparent', lineWidth: 1});
      const w = canvas.parentElement ? canvas.parentElement.clientWidth : 500;
      viz.setSize(w, 36);
      viz.drawWaveform(audioBuf);
      const row = this._rows[name];
      if (row) {
        row.el.classList.remove('loading');
        row.playBtn.disabled = false;
      }
    } catch (e) {
      console.error('Failed to load stem waveform:', e);
      const row = this._rows[name];
      if (row) row.el.classList.remove('loading');
    }
  }

  _togglePlay(name, btn) {
    if (this._playing && this._playing.name === name) {
      this.stopPlayback();
      return;
    }
    this.stopPlayback();
    const buf = this._buffers[name];
    if (!buf) return;

    const source = this.audioCtx.createBufferSource();
    source.buffer = buf;
    source.connect(this.audioCtx.destination);
    source.start();

    btn.textContent = '\u25A0';
    btn.classList.add('playing');

    const row = this._rows[name];
    const duration = buf.duration;
    const startTime = this.audioCtx.currentTime;

    this._playing = { name, source, btn, startTime, duration };

    const animate = () => {
      if (!this._playing || this._playing.name !== name) return;
      const elapsed = this.audioCtx.currentTime - startTime;
      const pct = Math.min(elapsed / duration, 1.0);
      if (row && row.playhead) {
        row.playhead.style.left = (pct * 100) + '%';
        row.playhead.style.display = 'block';
      }
      if (pct < 1.0) {
        this._playing.rafId = requestAnimationFrame(animate);
      }
    };
    this._playing.rafId = requestAnimationFrame(animate);

    source.onended = () => {
      if (this._playing && this._playing.name === name) {
        this._resetPlayState(name);
      }
    };
  }

  _resetPlayState(name) {
    if (this._playing && this._playing.rafId) {
      cancelAnimationFrame(this._playing.rafId);
    }
    const row = this._rows[name];
    if (row && row.playhead) {
      row.playhead.style.display = 'none';
      row.playhead.style.left = '0%';
    }
    if (this._playing) {
      this._playing.btn.textContent = '\u25B6';
      this._playing.btn.classList.remove('playing');
    }
    this._playing = null;
  }

  stopPlayback() {
    if (this._playing) {
      try { this._playing.source.stop(); } catch (_) {}
      this._resetPlayState(this._playing.name);
    }
  }

  getSelectedStem() { return this._selected; }
  onSelect(cb) { this._callback = cb; }
  destroy() {
    this.stopPlayback();
    this.container.innerHTML = '';
    this._rows = {};
    this._buffers = {};
    this._checked.clear();
  }
}
