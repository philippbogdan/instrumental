/* ───────────────────────────────────────────────────
   WaveformViz – Canvas-based waveform visualization
   ─────────────────────────────────────────────────── */

class WaveformViz {
  /**
   * @param {HTMLCanvasElement} canvasElement
   * @param {Object} options
   */
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');

    this.color = options.color || '#C2B280';
    this.bgColor = options.bgColor || '#000000';
    this.lineWidth = options.lineWidth || 1.5;

    this.dpr = window.devicePixelRatio || 1;

    // Simple one-time sizing: use parent width, fixed 120px height
    const w = this.canvas.parentElement?.clientWidth || 672;
    const h = 120;
    this._applySize(w, h);
  }

  /* ── Public API ─────────────────────────────────── */

  /**
   * Draw a time-domain waveform from an AudioBuffer.
   * For each pixel column, computes min/max sample values and draws
   * a vertical line between them — the classic "overview" waveform style.
   * @param {AudioBuffer} audioBuffer
   */
  drawWaveform(audioBuffer) {
    const ctx = this.ctx;
    const w = this._logicalWidth();
    const h = this._logicalHeight();
    const data = audioBuffer.getChannelData(0);
    const length = data.length;

    this.clear();

    // Subtle center line
    ctx.strokeStyle = this._dimColor(this.color, 0.25);
    ctx.lineWidth = 0.5 * this.dpr;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    // Waveform
    const ampScale = 0.8; // fill 80 % of canvas height
    const halfH = h / 2;
    ctx.strokeStyle = this.color;
    ctx.lineWidth = this.lineWidth * this.dpr;

    ctx.beginPath();
    for (let x = 0; x < w; x++) {
      const startSample = Math.floor((x / w) * length);
      const endSample = Math.floor(((x + 1) / w) * length);

      let min = 1;
      let max = -1;
      for (let i = startSample; i < endSample; i++) {
        const s = data[i];
        if (s < min) min = s;
        if (s > max) max = s;
      }

      const yMin = halfH - max * halfH * ampScale;
      const yMax = halfH - min * halfH * ampScale;

      ctx.moveTo(x, yMin);
      ctx.lineTo(x, yMax);
    }
    ctx.stroke();
  }

  /**
   * Draw a frequency-domain spectrum from an AudioBuffer.
   * Uses a simple DFT magnitude estimation and renders a filled area
   * from the bottom of the canvas in amaranth (#E52B50) at 50 % opacity.
   * @param {AudioBuffer} audioBuffer
   */
  drawSpectrum(audioBuffer) {
    const ctx = this.ctx;
    const w = this._logicalWidth();
    const h = this._logicalHeight();
    const data = audioBuffer.getChannelData(0);
    const length = data.length;

    // Use a power-of-two FFT size capped at 4096 for performance
    const fftSize = Math.min(4096, this._nextPow2(length));
    const real = new Float32Array(fftSize);
    const imag = new Float32Array(fftSize);

    // Copy samples (windowed with Hann)
    for (let i = 0; i < fftSize; i++) {
      const hann = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
      real[i] = (i < length ? data[i] : 0) * hann;
    }

    // In-place Cooley-Tukey FFT
    this._fft(real, imag);

    // Compute magnitude spectrum (only first half — positive frequencies)
    const halfFFT = fftSize / 2;
    const magnitudes = new Float32Array(halfFFT);
    let maxMag = 0;
    for (let i = 0; i < halfFFT; i++) {
      magnitudes[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
      if (magnitudes[i] > maxMag) maxMag = magnitudes[i];
    }

    // Normalise
    if (maxMag > 0) {
      for (let i = 0; i < halfFFT; i++) magnitudes[i] /= maxMag;
    }

    // Draw filled area from bottom
    ctx.fillStyle = 'rgba(229, 43, 80, 0.5)'; // amaranth at 50 %
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let x = 0; x < w; x++) {
      // Map x to a log-scaled frequency bin for perceptual balance
      const ratio = x / w;
      const bin = Math.floor(Math.pow(ratio, 2) * (halfFFT - 1));
      const barH = magnitudes[bin] * h * 0.9;
      ctx.lineTo(x, h - barH);
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fill();
  }

  /** Clear the canvas and fill with bgColor. */
  clear() {
    const w = this._logicalWidth();
    const h = this._logicalHeight();
    this.ctx.fillStyle = this.bgColor;
    this.ctx.fillRect(0, 0, w, h);
  }

  /**
   * Resize the canvas to the given CSS (logical) dimensions.
   * @param {number} width  CSS pixels
   * @param {number} height CSS pixels
   */
  setSize(width, height) {
    this._applySize(width, height);
  }

  /* ── Internal helpers ───────────────────────────── */

  _applySize(cssW, cssH) {
    if (!cssW || !cssH) return;
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = cssW * this.dpr;
    this.canvas.height = cssH * this.dpr;
    this.canvas.style.width = cssW + 'px';
    this.canvas.style.height = cssH + 'px';
    this.ctx.setTransform(1, 0, 0, 1, 0, 0); // reset before scaling
  }

  _logicalWidth() { return this.canvas.width; }
  _logicalHeight() { return this.canvas.height; }

  /** Return a colour string with reduced opacity. */
  _dimColor(hex, opacity) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + opacity + ')';
  }

  _nextPow2(n) {
    let p = 1;
    while (p < n) p <<= 1;
    return p;
  }

  /** In-place iterative Cooley-Tukey radix-2 FFT. */
  _fft(real, imag) {
    const n = real.length;
    // Bit-reversal permutation
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      while (j & bit) { j ^= bit; bit >>= 1; }
      j ^= bit;
      if (i < j) {
        let tmp = real[i]; real[i] = real[j]; real[j] = tmp;
        tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
      }
    }
    // Butterfly stages
    for (let len = 2; len <= n; len <<= 1) {
      const halfLen = len >> 1;
      const angle = -2 * Math.PI / len;
      const wR = Math.cos(angle);
      const wI = Math.sin(angle);
      for (let i = 0; i < n; i += len) {
        let curR = 1, curI = 0;
        for (let j = 0; j < halfLen; j++) {
          const tR = curR * real[i + j + halfLen] - curI * imag[i + j + halfLen];
          const tI = curR * imag[i + j + halfLen] + curI * real[i + j + halfLen];
          real[i + j + halfLen] = real[i + j] - tR;
          imag[i + j + halfLen] = imag[i + j] - tI;
          real[i + j] += tR;
          imag[i + j] += tI;
          const nextR = curR * wR - curI * wI;
          curI = curR * wI + curI * wR;
          curR = nextR;
        }
      }
    }
  }
}
