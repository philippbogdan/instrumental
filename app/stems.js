class StemDisplay {
  constructor(containerElement, audioContext) {
    this.container = containerElement;
    this.audioCtx = audioContext;
    this._selected = 'other';
    this._callback = null;
    this._rows = {};
    this._buffers = {};
    this._playing = null;  // {name, source, btn, startTime, duration, rafId}
  }

  setStems(stemUrls) {
    this.container.innerHTML = '';
    this.stopPlayback();
    const stemOrder = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'];
    // Only show stems that the server returned

    for (const name of stemOrder) {
      const url = stemUrls[name];
      if (!url) continue;

      const row = document.createElement('div');
      row.className = 'stem-row loading';  // Start as loading skeleton

      const playBtn = document.createElement('button');
      playBtn.className = 'stem-play-btn';
      playBtn.textContent = '\u25B6';
      playBtn.title = 'Preview ' + name;
      playBtn.disabled = true;  // Disabled until waveform loads
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
      // Playhead overlay
      const playhead = document.createElement('div');
      playhead.className = 'stem-playhead';
      waveDiv.appendChild(canvas);
      waveDiv.appendChild(playhead);

      row.appendChild(playBtn);
      row.appendChild(label);
      row.appendChild(waveDiv);
      this.container.appendChild(row);

      this._rows[name] = { el: row, playBtn, waveDiv, playhead, canvas };

      row.addEventListener('click', () => {
        Object.values(this._rows).forEach(r => r.el.classList.remove('selected'));
        row.classList.add('selected');
        this._selected = name;
        if (this._callback) this._callback(name);
      });

      this._loadWaveform(url, canvas, name);
    }
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
      // Remove loading state and enable play
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

    // Animate playhead
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
  }
}
