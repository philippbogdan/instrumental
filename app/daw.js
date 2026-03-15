/* ── daw.js ── Mini Piano-Roll DAW ─────────────────────────────────── */

class PianoRollDAW {
  constructor(containerElement, options = {}) {
    this.container = containerElement;
    const defaults = {
      colors: {
        bg: '#000',
        note: '#E52B50',
        grid: 'rgba(194,178,128,0.1)',
        text: '#C2B280',
        playhead: '#E52B50'
      }
    };
    this.colors = Object.assign({}, defaults.colors, (options.colors || {}));
    this.notes = [];
    this.timeRange = [0, 4];
    this.freqRange = [200, 600];
    this._playing = false;
    this._animId = null;
    this._timeouts = [];
    this._synth = null;
    this._playStartTime = 0;
    this.keyWidth = 60;

    // Create canvas
    this.canvas = document.createElement('canvas');
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.canvas.style.display = 'block';
    this.container.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');

    // Handle resize
    this._resizeObserver = new ResizeObserver(() => this._resize());
    this._resizeObserver.observe(this.container);
    this._resize();
  }

  /* ── Helpers ────────────────────────────────────────────────────── */

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.container.getBoundingClientRect();
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = rect.width;
    this.h = rect.height;
    this._draw();
  }

  static _noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

  static freqToMidi(freq) {
    return Math.round(12 * Math.log2(freq / 440) + 69);
  }

  static midiToNoteName(midi) {
    const octave = Math.floor(midi / 12) - 1;
    const name = PianoRollDAW._noteNames[midi % 12];
    return name + octave;
  }

  static midiToFreq(midi) {
    return 440 * Math.pow(2, (midi - 69) / 12);
  }

  static freqToNoteName(freq) {
    return PianoRollDAW.midiToNoteName(PianoRollDAW.freqToMidi(freq));
  }

  _timeToX(t) {
    const drawW = this.w - this.keyWidth;
    return this.keyWidth + ((t - this.timeRange[0]) / (this.timeRange[1] - this.timeRange[0])) * drawW;
  }

  _midiToY(midi) {
    // Higher pitch = higher on screen (lower Y)
    const frac = (midi - this.midiMin) / (this.midiMax - this.midiMin);
    return this.h - (frac * this.h);
  }

  /* ── setNotes ───────────────────────────────────────────────────── */

  setNotes(noteArray) {
    this.notes = noteArray || [];
    if (this.notes.length === 0) {
      this.timeRange = [0, 4];
      this.midiMin = 48;
      this.midiMax = 84;
      this._draw();
      return;
    }

    // Compute time range
    const maxEnd = Math.max(...this.notes.map(n => n.onset + n.duration));
    this.timeRange = [0, maxEnd + 0.5];

    // Compute pitch range in MIDI
    const midis = this.notes.map(n => PianoRollDAW.freqToMidi(n.freq));
    this.midiMin = Math.min(...midis) - 2;
    this.midiMax = Math.max(...midis) + 2;

    // Compute freq range for compatibility
    this.freqRange = [
      PianoRollDAW.midiToFreq(this.midiMin),
      PianoRollDAW.midiToFreq(this.midiMax)
    ];

    this._draw();
  }

  /* ── Drawing ────────────────────────────────────────────────────── */

  _draw(playheadTime) {
    const ctx = this.ctx;
    const w = this.w;
    const h = this.h;
    if (!w || !h) return;

    // Background
    ctx.fillStyle = this.colors.bg;
    ctx.fillRect(0, 0, w, h);

    const drawW = w - this.keyWidth;
    const midiSpan = this.midiMax - this.midiMin;
    if (midiSpan <= 0) return;
    const noteH = h / midiSpan;

    // Grid: horizontal lines per semitone
    ctx.strokeStyle = this.colors.grid;
    ctx.lineWidth = 1;
    for (let midi = this.midiMin; midi <= this.midiMax; midi++) {
      const y = this._midiToY(midi);
      ctx.beginPath();
      ctx.moveTo(this.keyWidth, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Grid: vertical lines every 0.5s
    const tRange = this.timeRange[1] - this.timeRange[0];
    const step = 0.5;
    for (let t = 0; t <= this.timeRange[1]; t += step) {
      const x = this._timeToX(t);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    // Time labels along top
    ctx.fillStyle = this.colors.text;
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    for (let t = 0; t <= this.timeRange[1]; t += 1) {
      const x = this._timeToX(t);
      ctx.fillText(t.toFixed(1) + 's', x, 10);
    }

    // Piano key labels on left edge
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillStyle = this.colors.text;
    for (let midi = this.midiMin; midi <= this.midiMax; midi++) {
      const name = PianoRollDAW.midiToNoteName(midi);
      const y = this._midiToY(midi);
      // Draw label centered on the semitone row
      ctx.fillText(name, this.keyWidth - 4, y + 3);

      // Highlight black keys with a subtle bar
      const noteIdx = midi % 12;
      if ([1, 3, 6, 8, 10].includes(noteIdx)) {
        ctx.fillStyle = 'rgba(194,178,128,0.05)';
        ctx.fillRect(0, y - noteH / 2, this.keyWidth, noteH);
        ctx.fillStyle = this.colors.text;
      }
    }

    // Separator line between keys and main area
    ctx.strokeStyle = this.colors.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(this.keyWidth, 0);
    ctx.lineTo(this.keyWidth, h);
    ctx.stroke();

    // Draw notes as rounded rectangles
    for (const note of this.notes) {
      const midi = PianoRollDAW.freqToMidi(note.freq);
      const x = this._timeToX(note.onset);
      const nw = (note.duration / tRange) * drawW;
      const y = this._midiToY(midi) - noteH / 2;

      ctx.fillStyle = this.colors.note;
      this._roundRect(ctx, x, y, Math.max(nw, 2), noteH * 0.8, 3);
      ctx.fill();
    }

    // Playhead
    if (playheadTime !== undefined && playheadTime >= 0) {
      const px = this._timeToX(playheadTime);
      ctx.strokeStyle = this.colors.playhead;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, h);
      ctx.stroke();
    }
  }

  _roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
  }

  /* ── Playback ───────────────────────────────────────────────────── */

  play(synth) {
    if (this._playing) this.stop();
    this._synth = synth;
    this._playing = true;
    this._timeouts = [];

    const totalDuration = this.timeRange[1];
    this._playStartTime = performance.now();

    // Schedule each note
    const sorted = [...this.notes].sort((a, b) => a.onset - b.onset);
    for (const note of sorted) {
      const tid = setTimeout(() => {
        synth.playNote(note.freq, note.duration, 0.8);
      }, note.onset * 1000);
      this._timeouts.push(tid);
    }

    // Animate playhead
    const animate = () => {
      const elapsed = (performance.now() - this._playStartTime) / 1000;
      if (elapsed > totalDuration) {
        this.stop();
        return;
      }
      this._draw(elapsed);
      this._animId = requestAnimationFrame(animate);
    };
    this._animId = requestAnimationFrame(animate);
  }

  stop() {
    this._playing = false;

    // Cancel animation
    if (this._animId !== null) {
      cancelAnimationFrame(this._animId);
      this._animId = null;
    }

    // Clear scheduled timeouts
    for (const tid of this._timeouts) {
      clearTimeout(tid);
    }
    this._timeouts = [];

    // Stop synth
    if (this._synth && typeof this._synth.stopAll === 'function') {
      this._synth.stopAll();
    }
    this._synth = null;

    // Redraw without playhead
    this._draw();
  }

  isPlaying() {
    return this._playing;
  }
}
