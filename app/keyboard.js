/**
 * PianoKeyboard — 2-octave on-screen piano keyboard (Vital-style)
 * Depends on InstrumentalSynth (synth.js loaded before this file).
 */

class PianoKeyboard {
  constructor(containerElement, synth, options = {}) {
    this.container = containerElement;
    this.synth = synth;
    this.startOctave = options.startOctave ?? 3;
    this.numOctaves = options.numOctaves ?? 2;

    // MIDI range
    this.startNote = (this.startOctave + 1) * 12; // C3 = 48
    this.endNote = this.startNote + this.numOctaves * 12 - 1; // B4 = 71

    // State
    this._keys = new Map();          // midiNote -> DOM element
    this._activeNotes = new Set();   // currently pressed midi notes
    this._mouseDown = false;
    this._activeTouch = new Map();   // touchId -> midiNote
    this._boundListeners = [];

    // Note names
    this._noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    this._whiteNoteIndices = [0, 2, 4, 5, 7, 9, 11]; // C D E F G A B
    this._blackNoteIndices = [1, 3, 6, 8, 10];        // C# D# F# G# A#

    // Computer keyboard mapping
    this._keyMap = this._buildKeyMap();
  }

  _buildKeyMap() {
    // DAW-style layout: home row = white keys, top row = sharps
    // H = C4, so counting back: A=E3, S=F3, D=G3, F=A3, G=B3
    // Forward: H=C4, J=D4, K=E4, L=F4, ;=G4
    // Bottom row for lowest: Z=C3, X=D3
    // Top row sharps positioned above gaps between white keys
    const base = this.startNote; // C3 = 48
    return {
      // Bottom row: C3, D3
      'z': base + 0,   // C3
      'x': base + 2,   // D3
      // Home row: white keys E3 through G4
      'a': base + 4,   // E3
      's': base + 5,   // F3
      'd': base + 7,   // G3
      'f': base + 9,   // A3
      'g': base + 11,  // B3
      'h': base + 12,  // C4
      'j': base + 14,  // D4
      'k': base + 16,  // E4
      'l': base + 17,  // F4
      ';': base + 19,  // G4
      "'": base + 21,  // A4
      // Top row: sharps
      'w': base + 1,   // C#3
      'e': base + 3,   // D#3
      't': base + 6,   // F#3
      'y': base + 8,   // G#3
      'u': base + 10,  // A#3
      'i': base + 13,  // C#4
      'o': base + 15,  // D#4
      'p': base + 18,  // F#4
      '[': base + 20,  // G#4
      ']': base + 22,  // A#4
      // Extra: B4
      '\\': base + 23, // B4
    };
  }

  _midiToFreq(midiNote) {
    return 440 * Math.pow(2, (midiNote - 69) / 12);
  }

  _noteName(midiNote) {
    const name = this._noteNames[midiNote % 12];
    const octave = Math.floor(midiNote / 12) - 1;
    return name + octave;
  }

  _isBlack(midiNote) {
    return this._blackNoteIndices.includes(midiNote % 12);
  }

  render() {
    this.container.innerHTML = '';
    this.container.classList.add('keyboard-container');

    const wrapper = document.createElement('div');
    wrapper.style.position = 'relative';
    wrapper.style.display = 'flex';
    wrapper.style.width = '100%';
    this.container.appendChild(wrapper);

    // First pass: create white keys in flex order
    const whiteKeys = [];
    for (let note = this.startNote; note <= this.endNote; note++) {
      if (!this._isBlack(note)) {
        const key = document.createElement('div');
        key.className = 'key white';
        key.dataset.note = note;

        const label = document.createElement('span');
        label.className = 'key-label';
        label.textContent = this._noteName(note);
        key.appendChild(label);

        wrapper.appendChild(key);
        this._keys.set(note, key);
        whiteKeys.push({ note, el: key });
      }
    }

    // Second pass: create black keys positioned absolutely
    // Black keys sit between white keys. We position them based on the
    // preceding white key's index in the flex row.
    const whiteKeyWidth = 100 / whiteKeys.length; // percentage width

    for (let note = this.startNote; note <= this.endNote; note++) {
      if (this._isBlack(note)) {
        // Find the index of the white key just before this black key
        const prevWhiteIdx = whiteKeys.findIndex(wk => wk.note === note - 1);
        if (prevWhiteIdx < 0) continue;

        const key = document.createElement('div');
        key.className = 'key black';
        key.dataset.note = note;

        // Position: right edge of previous white key, offset left by half black key width
        const blackKeyWidthPct = whiteKeyWidth * 0.58;
        const leftPct = (prevWhiteIdx + 1) * whiteKeyWidth - blackKeyWidthPct / 2;
        key.style.left = leftPct + '%';
        key.style.width = blackKeyWidthPct + '%';

        wrapper.appendChild(key);
        this._keys.set(note, key);
      }
    }

    this._attachMouseListeners(wrapper);
    this._attachTouchListeners(wrapper);
    this._attachKeyboardListeners();
  }

  // --- Note on/off ---

  _noteOn(midiNote) {
    if (this._activeNotes.has(midiNote)) return;
    this._activeNotes.add(midiNote);

    const freq = this._midiToFreq(midiNote);
    if (this.synth && typeof this.synth.playNote === 'function') {
      this.synth.playNote(freq, 0.15, 0.8);
    }
    this.setActive(midiNote, true);
  }

  _noteOff(midiNote) {
    if (!this._activeNotes.has(midiNote)) return;
    this._activeNotes.delete(midiNote);
    this.setActive(midiNote, false);
  }

  setActive(midiNote, isActive) {
    const el = this._keys.get(midiNote);
    if (!el) return;
    if (isActive) {
      el.classList.add('active');
    } else {
      el.classList.remove('active');
    }
  }

  // --- Mouse interaction ---

  _getNoteFromElement(el) {
    const keyEl = el.closest('.key');
    if (!keyEl || !keyEl.dataset.note) return null;
    return parseInt(keyEl.dataset.note, 10);
  }

  _attachMouseListeners(wrapper) {
    const onMouseDown = (e) => {
      e.preventDefault();
      this._mouseDown = true;
      const note = this._getNoteFromElement(e.target);
      if (note !== null) this._noteOn(note);
    };

    const onMouseUp = () => {
      this._mouseDown = false;
      // Release all mouse-held notes
      for (const note of [...this._activeNotes]) {
        // Only release if not held by keyboard
        if (!this._isHeldByKeyboard(note)) {
          this._noteOff(note);
        }
      }
    };

    const onMouseEnter = (e) => {
      if (!this._mouseDown) return;
      const note = this._getNoteFromElement(e.target);
      if (note !== null) this._noteOn(note);
    };

    const onMouseLeave = (e) => {
      if (!this._mouseDown) return;
      const note = this._getNoteFromElement(e.target);
      if (note !== null && !this._isHeldByKeyboard(note)) {
        this._noteOff(note);
      }
    };

    wrapper.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mouseup', onMouseUp);

    // Per-key enter/leave for drag
    for (const [, el] of this._keys) {
      el.addEventListener('mouseenter', onMouseEnter);
      el.addEventListener('mouseleave', onMouseLeave);
    }

    this._boundListeners.push(
      { target: wrapper, event: 'mousedown', fn: onMouseDown },
      { target: document, event: 'mouseup', fn: onMouseUp }
    );
    for (const [, el] of this._keys) {
      this._boundListeners.push(
        { target: el, event: 'mouseenter', fn: onMouseEnter },
        { target: el, event: 'mouseleave', fn: onMouseLeave }
      );
    }
  }

  // --- Touch interaction ---

  _attachTouchListeners(wrapper) {
    const getNotFromTouch = (touch) => {
      const el = document.elementFromPoint(touch.clientX, touch.clientY);
      if (!el) return null;
      return this._getNoteFromElement(el);
    };

    const onTouchStart = (e) => {
      e.preventDefault();
      for (const touch of e.changedTouches) {
        const note = getNotFromTouch(touch);
        if (note !== null) {
          this._activeTouch.set(touch.identifier, note);
          this._noteOn(note);
        }
      }
    };

    const onTouchMove = (e) => {
      e.preventDefault();
      for (const touch of e.changedTouches) {
        const newNote = getNotFromTouch(touch);
        const oldNote = this._activeTouch.get(touch.identifier);
        if (oldNote !== undefined && oldNote !== newNote) {
          if (!this._isHeldByKeyboard(oldNote)) this._noteOff(oldNote);
        }
        if (newNote !== null && newNote !== oldNote) {
          this._activeTouch.set(touch.identifier, newNote);
          this._noteOn(newNote);
        } else if (newNote === null) {
          this._activeTouch.delete(touch.identifier);
        }
      }
    };

    const onTouchEnd = (e) => {
      e.preventDefault();
      for (const touch of e.changedTouches) {
        const note = this._activeTouch.get(touch.identifier);
        if (note !== undefined) {
          if (!this._isHeldByKeyboard(note)) this._noteOff(note);
          this._activeTouch.delete(touch.identifier);
        }
      }
    };

    wrapper.addEventListener('touchstart', onTouchStart, { passive: false });
    wrapper.addEventListener('touchmove', onTouchMove, { passive: false });
    wrapper.addEventListener('touchend', onTouchEnd, { passive: false });

    this._boundListeners.push(
      { target: wrapper, event: 'touchstart', fn: onTouchStart },
      { target: wrapper, event: 'touchmove', fn: onTouchMove },
      { target: wrapper, event: 'touchend', fn: onTouchEnd }
    );
  }

  // --- Computer keyboard interaction ---

  _isHeldByKeyboard(midiNote) {
    return this._keyboardHeld && this._keyboardHeld.has(midiNote);
  }

  _attachKeyboardListeners() {
    this._keyboardHeld = new Set();

    const onKeyDown = (e) => {
      if (e.repeat) return;
      const note = this._keyMap[e.key.toLowerCase()];
      if (note !== undefined) {
        e.preventDefault();
        this._keyboardHeld.add(note);
        this._noteOn(note);
      }
    };

    const onKeyUp = (e) => {
      const note = this._keyMap[e.key.toLowerCase()];
      if (note !== undefined) {
        e.preventDefault();
        this._keyboardHeld.delete(note);
        this._noteOff(note);
      }
    };

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);

    this._boundListeners.push(
      { target: document, event: 'keydown', fn: onKeyDown },
      { target: document, event: 'keyup', fn: onKeyUp }
    );
  }

  // --- Cleanup ---

  destroy() {
    for (const { target, event, fn } of this._boundListeners) {
      target.removeEventListener(event, fn);
    }
    this._boundListeners = [];
    this._keys.clear();
    this._activeNotes.clear();
    this._activeTouch.clear();
    if (this._keyboardHeld) this._keyboardHeld.clear();
    this.container.innerHTML = '';
  }
}
