/**
 * AudioPlayer — lightweight audio playback controller.
 * Expects a container element with the following child elements:
 *   .player-play-btn   — button to toggle play/pause
 *   .player-progress    — clickable progress bar track
 *   .player-progress-fill — inner fill element for progress
 *   .player-time        — text element showing current / total time
 */
class AudioPlayer {
  constructor(containerElement) {
    this.container = containerElement;
    this.audio = new Audio();

    this.playBtn = containerElement.querySelector('.player-play-btn');
    this.progressBar = containerElement.querySelector('.player-progress');
    this.progressFill = containerElement.querySelector('.player-progress-fill');
    this.timeDisplay = containerElement.querySelector('.player-time');

    this._maxDuration = null;
    this._handlers = {};
    this._bindEvents();
  }

  /** Cap playback at a maximum duration (seconds). Progress and time display respect this cap. */
  setMaxDuration(dur) {
    this._maxDuration = dur;
  }

  _effectiveDuration() {
    if (this._maxDuration && this.audio.duration) {
      return Math.min(this._maxDuration, this.audio.duration);
    }
    return this.audio.duration || 0;
  }

  load(url) {
    this.audio.src = url;
    this.audio.load();
    this.audio.onerror = () => console.error('AudioPlayer: failed to load', url);
  }

  play() {
    this.audio.play();
    this.playBtn.textContent = '||';
  }

  pause() {
    this.audio.pause();
    this.playBtn.textContent = 'Play';
  }

  toggle() {
    this.audio.paused ? this.play() : this.pause();
  }

  seek(fraction) {
    if (!this.audio.duration) return;
    this.audio.currentTime = Math.max(0, Math.min(1, fraction)) * this.audio.duration;
  }

  _bindEvents() {
    this._handlers.toggle = () => this.toggle();
    this.playBtn.addEventListener('click', this._handlers.toggle);

    this._handlers.timeupdate = () => {
      const dur = this._effectiveDuration();
      if (!dur) return;
      // Stop at max duration
      if (this._maxDuration && this.audio.currentTime >= this._maxDuration) {
        this.audio.pause();
        this.audio.currentTime = 0;
        this.playBtn.textContent = 'Play';
        this.progressFill.style.width = '0%';
        this.timeDisplay.textContent = '0:00 / ' + this._formatTime(dur);
        return;
      }
      const pct = (this.audio.currentTime / dur) * 100;
      this.progressFill.style.width = Math.min(pct, 100) + '%';
      this.timeDisplay.textContent =
        this._formatTime(this.audio.currentTime) + ' / ' + this._formatTime(dur);
    };
    this.audio.addEventListener('timeupdate', this._handlers.timeupdate);

    this._handlers.loadedmetadata = () => {
      const dur = this._effectiveDuration();
      this.timeDisplay.textContent = '0:00 / ' + this._formatTime(dur);
    };
    this.audio.addEventListener('loadedmetadata', this._handlers.loadedmetadata);

    this._handlers.ended = () => {
      const dur = this._effectiveDuration();
      this.playBtn.textContent = 'Play';
      this.progressFill.style.width = '0%';
      this.timeDisplay.textContent = '0:00 / ' + this._formatTime(dur);
    };
    this.audio.addEventListener('ended', this._handlers.ended);

    this._handlers.seek = (e) => {
      const fraction = e.offsetX / this.progressBar.offsetWidth;
      const dur = this._effectiveDuration();
      if (dur) this.audio.currentTime = Math.max(0, Math.min(1, fraction)) * dur;
    };
    this.progressBar.addEventListener('click', this._handlers.seek);
  }

  _formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m + ':' + String(s).padStart(2, '0');
  }

  destroy() {
    this.audio.pause();
    this.audio.src = '';
    this.playBtn.removeEventListener('click', this._handlers.toggle);
    this.audio.removeEventListener('timeupdate', this._handlers.timeupdate);
    this.audio.removeEventListener('loadedmetadata', this._handlers.loadedmetadata);
    this.audio.removeEventListener('ended', this._handlers.ended);
    this.progressBar.removeEventListener('click', this._handlers.seek);
  }
}
