class StemDisplay {
  constructor(containerElement, audioContext) {
    this.container = containerElement;
    this.audioCtx = audioContext;
    this._selected = 'other';
    this._callback = null;
    this._rows = {};
  }

  setStems(stemUrls) {
    this.container.innerHTML = '';
    const stemOrder = ['vocals', 'drums', 'bass', 'other'];

    for (const name of stemOrder) {
      const url = stemUrls[name];
      if (!url) continue;

      const row = document.createElement('div');
      row.className = 'stem-row' + (name === this._selected ? ' selected' : '');

      const label = document.createElement('span');
      label.className = 'stem-label';
      label.textContent = name;

      const waveDiv = document.createElement('div');
      waveDiv.className = 'stem-waveform';
      const canvas = document.createElement('canvas');
      waveDiv.appendChild(canvas);

      row.appendChild(label);
      row.appendChild(waveDiv);
      this.container.appendChild(row);

      this._rows[name] = row;

      row.addEventListener('click', () => {
        Object.values(this._rows).forEach(r => r.classList.remove('selected'));
        row.classList.add('selected');
        this._selected = name;
        if (this._callback) this._callback(name);
      });

      this._loadWaveform(url, canvas);
    }
  }

  async _loadWaveform(url, canvas) {
    try {
      const resp = await fetch(url);
      const arrayBuf = await resp.arrayBuffer();
      const audioBuf = await this.audioCtx.decodeAudioData(arrayBuf);
      const viz = new WaveformViz(canvas, {color: '#000000', bgColor: 'transparent', lineWidth: 1});
      viz.drawWaveform(audioBuf);
    } catch (e) {
      console.error('Failed to load stem waveform:', e);
    }
  }

  getSelectedStem() { return this._selected; }
  onSelect(cb) { this._callback = cb; }
  destroy() { this.container.innerHTML = ''; this._rows = {}; }
}
