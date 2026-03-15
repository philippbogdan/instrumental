/**
 * InstrumentalSynth - WebAudio subtractive synthesizer
 *
 * Mirrors the Python synth from src/synth.py.
 * Signal chain: Oscillators -> Filter (cascaded LP) -> EQ -> ADSR -> Gain -> Reverb -> Out
 *
 * 28 parameters, all normalized [0,1].
 */

const DEFAULT_PARAM_DEFS = [
  { name: "saw_mix",          lo: 0.0,   hi: 1.0 },
  { name: "square_mix",       lo: 0.0,   hi: 1.0 },
  { name: "sine_mix",         lo: 0.0,   hi: 1.0 },
  { name: "noise_mix",        lo: 0.0,   hi: 0.5 },
  { name: "detune",           lo: -24.0, hi: 24.0 },
  { name: "filter_cutoff",    lo: 200.0, hi: 16000.0 },
  { name: "filter_resonance", lo: 0.0,   hi: 0.95 },
  { name: "attack",           lo: 0.001, hi: 0.5 },
  { name: "decay",            lo: 0.001, hi: 1.0 },
  { name: "sustain",          lo: 0.0,   hi: 1.0 },
  { name: "release",          lo: 0.001, hi: 1.0 },
  { name: "gain",             lo: 0.0,   hi: 1.0 },
  { name: "filter_env",       lo: -1.0,  hi: 1.0 },
  { name: "reverb_size",      lo: 0.01,  hi: 0.99 },
  { name: "reverb_mix",       lo: 0.0,   hi: 0.8 },
  { name: "unison_voices",    lo: 1.0,   hi: 7.0 },
  { name: "unison_spread",    lo: 0.0,   hi: 0.5 },
  { name: "noise_floor",      lo: 0.0,   hi: 0.1 },
  { name: "filter_attack",    lo: 0.001, hi: 0.5 },
  { name: "filter_decay",     lo: 0.001, hi: 1.0 },
  { name: "filter_sustain",   lo: 0.0,   hi: 1.0 },
  { name: "filter_release",   lo: 0.001, hi: 1.0 },
  { name: "pulse_width",      lo: 0.1,   hi: 0.9 },
  { name: "filter_slope",     lo: 4.0,   hi: 48.0 },
  { name: "eq1_freq",         lo: 500.0, hi: 4500.0 },
  { name: "eq1_gain",         lo: -6.0,  hi: 6.0 },
  { name: "eq2_freq",         lo: 2000.0,hi: 10000.0 },
  { name: "eq2_gain",         lo: -6.0,  hi: 6.0 },
];

class InstrumentalSynth {
  constructor(audioContext) {
    this.ctx = audioContext;
    this._paramDefs = DEFAULT_PARAM_DEFS;
    this._params = new Float32Array(28).fill(0.5);
    this._activeNotes = [];
    this._lastBuffer = null;
  }

  /* --- Public API --- */

  setParams(paramsArray) {
    for (let i = 0; i < 28; i++) {
      this._params[i] = Math.max(0, Math.min(1, paramsArray[i] ?? 0.5));
    }
  }

  setParamDefs(defs) {
    this._paramDefs = defs;
  }

  playNote(freq, duration = 1.0, velocity = 1.0) {
    const p = this._denorm();
    const t0 = this.ctx.currentTime;
    const noteRef = { nodes: [], stop: null };

    const dest = this.ctx.destination;
    const chain = this._buildChain(this.ctx, p, freq, duration, velocity, t0, dest);

    noteRef.nodes = chain.allNodes;
    noteRef.stop = () => {
      chain.allNodes.forEach(n => {
        try { n.stop ? n.stop() : null; } catch (_) { /* already stopped */ }
      });
      chain.allNodes.forEach(n => {
        try { n.disconnect(); } catch (_) {}
      });
    };

    // Schedule auto-stop after full duration + release + tail
    const totalTime = duration + p.release + 0.1;
    const stopTimeout = setTimeout(() => {
      noteRef.stop();
      const idx = this._activeNotes.indexOf(noteRef);
      if (idx >= 0) this._activeNotes.splice(idx, 1);
    }, totalTime * 1000);
    noteRef._timeout = stopTimeout;

    this._activeNotes.push(noteRef);
    return noteRef;
  }

  stopAll() {
    this._activeNotes.forEach(ref => {
      clearTimeout(ref._timeout);
      ref.stop();
    });
    this._activeNotes = [];
  }

  async renderOffline(freq, duration = 1.0) {
    const sampleRate = this.ctx.sampleRate || 44100;
    const p = this._denorm();
    const totalDuration = duration + p.release + 0.5;
    const length = Math.ceil(totalDuration * sampleRate);
    const offCtx = new OfflineAudioContext(1, length, sampleRate);
    const t0 = 0;

    this._buildChain(offCtx, p, freq, duration, 1.0, t0, offCtx.destination);

    const buffer = await offCtx.startRendering();
    this._lastBuffer = buffer;
    return buffer;
  }

  getLastBuffer() {
    return this._lastBuffer;
  }

  /* --- Internal: denormalize params --- */

  _denorm() {
    const result = {};
    for (let i = 0; i < this._paramDefs.length; i++) {
      const { name, lo, hi } = this._paramDefs[i];
      const v = Math.max(0, Math.min(1, this._params[i]));
      result[name] = v * (hi - lo) + lo;
    }
    return result;
  }

  /* --- Internal: build the full audio graph --- */

  _buildChain(ctx, p, freq, duration, velocity, t0, destination) {
    const allNodes = [];
    const sampleRate = ctx.sampleRate || 44100;

    // --- Unison configuration ---
    const nVoices = Math.max(1, Math.round(p.unison_voices));
    const spread = p.unison_spread;
    const voiceDetunes = [];
    if (nVoices === 1) {
      voiceDetunes.push(0);
    } else {
      for (let i = 0; i < nVoices; i++) {
        voiceDetunes.push(spread * (2 * i / (nVoices - 1) - 1));
      }
    }

    // --- Mixer node (all oscillators sum here) ---
    const mixerNode = ctx.createGain();
    mixerNode.gain.value = 1.0;
    allNodes.push(mixerNode);

    // Normalize mix levels
    const totalMix = Math.max(0.01, p.saw_mix + p.square_mix + p.sine_mix + p.noise_mix);

    // --- Oscillators per voice ---
    const totalDuration = duration + p.release + 0.5;

    for (let vi = 0; vi < nVoices; vi++) {
      const totalDetune = (p.detune + voiceDetunes[vi]) * 100; // cents

      // Sawtooth
      if (p.saw_mix > 0.001) {
        const osc = ctx.createOscillator();
        osc.type = "sawtooth";
        osc.frequency.value = freq;
        osc.detune.value = totalDetune;
        const g = ctx.createGain();
        g.gain.value = (p.saw_mix / totalMix) / nVoices;
        osc.connect(g).connect(mixerNode);
        osc.start(t0);
        osc.stop(t0 + totalDuration);
        allNodes.push(osc, g);
      }

      // Pulse (via PeriodicWave)
      if (p.square_mix > 0.001) {
        const osc = ctx.createOscillator();
        const pw = this._createPulseWave(ctx, p.pulse_width);
        osc.setPeriodicWave(pw);
        osc.frequency.value = freq;
        osc.detune.value = totalDetune;
        const g = ctx.createGain();
        g.gain.value = (p.square_mix / totalMix) / nVoices;
        osc.connect(g).connect(mixerNode);
        osc.start(t0);
        osc.stop(t0 + totalDuration);
        allNodes.push(osc, g);
      }

      // Sine
      if (p.sine_mix > 0.001) {
        const osc = ctx.createOscillator();
        osc.type = "sine";
        osc.frequency.value = freq;
        osc.detune.value = totalDetune;
        const g = ctx.createGain();
        g.gain.value = (p.sine_mix / totalMix) / nVoices;
        osc.connect(g).connect(mixerNode);
        osc.start(t0);
        osc.stop(t0 + totalDuration);
        allNodes.push(osc, g);
      }
    }

    // --- Noise ---
    const noiseMix = p.noise_mix + p.noise_floor;
    if (noiseMix > 0.0001) {
      const noiseLen = Math.ceil(totalDuration * sampleRate);
      const noiseBuf = ctx.createBuffer(1, noiseLen, sampleRate);
      const data = noiseBuf.getChannelData(0);
      for (let i = 0; i < noiseLen; i++) {
        data[i] = Math.random() * 2 - 1;
      }
      const noiseSrc = ctx.createBufferSource();
      noiseSrc.buffer = noiseBuf;
      const noiseGain = ctx.createGain();
      noiseGain.gain.value = noiseMix / totalMix;
      noiseSrc.connect(noiseGain).connect(mixerNode);
      noiseSrc.start(t0);
      noiseSrc.stop(t0 + totalDuration);
      allNodes.push(noiseSrc, noiseGain);
    }

    // --- Cascaded lowpass filters ---
    const nFilters = Math.max(1, Math.min(4, Math.round(p.filter_slope / 12)));
    let filterChainIn = mixerNode;
    const filters = [];

    for (let fi = 0; fi < nFilters; fi++) {
      const f = ctx.createBiquadFilter();
      f.type = "lowpass";
      f.frequency.value = p.filter_cutoff;
      f.Q.value = fi === 0 ? p.filter_resonance * 20 : 0.707;
      filterChainIn.connect(f);
      filterChainIn = f;
      filters.push(f);
      allNodes.push(f);
    }

    // --- Filter envelope (modulates cutoff of all filters) ---
    this._scheduleFilterEnvelope(filters, p, t0, duration);

    // --- EQ: two peaking filters ---
    const eq1 = ctx.createBiquadFilter();
    eq1.type = "peaking";
    eq1.frequency.value = p.eq1_freq;
    eq1.gain.value = p.eq1_gain;
    eq1.Q.value = 2.0;
    filterChainIn.connect(eq1);
    allNodes.push(eq1);

    const eq2 = ctx.createBiquadFilter();
    eq2.type = "peaking";
    eq2.frequency.value = p.eq2_freq;
    eq2.gain.value = p.eq2_gain;
    eq2.Q.value = 2.0;
    eq1.connect(eq2);
    allNodes.push(eq2);

    // --- Amplitude ADSR envelope ---
    const envGain = ctx.createGain();
    envGain.gain.value = 0;
    this._scheduleADSR(envGain.gain, p.attack, p.decay, p.sustain, p.release, t0, duration);
    eq2.connect(envGain);
    allNodes.push(envGain);

    // --- Master gain ---
    const masterGain = ctx.createGain();
    masterGain.gain.value = p.gain * velocity;
    envGain.connect(masterGain);
    allNodes.push(masterGain);

    // --- Reverb (convolver) ---
    const dryGain = ctx.createGain();
    dryGain.gain.value = 1.0 - p.reverb_mix;
    masterGain.connect(dryGain);
    dryGain.connect(destination);
    allNodes.push(dryGain);

    if (p.reverb_mix > 0.001) {
      const irLength = Math.max(1024, Math.ceil(sampleRate * Math.max(0.01, p.reverb_size) * 2));
      const irBuffer = ctx.createBuffer(1, irLength, sampleRate);
      const irData = irBuffer.getChannelData(0);
      const decayRate = -3.0 / (p.reverb_size * sampleRate);
      for (let i = 0; i < irLength; i++) {
        irData[i] = (Math.random() * 2 - 1) * Math.exp(decayRate * i);
      }

      const convolver = ctx.createConvolver();
      convolver.buffer = irBuffer;
      const wetGain = ctx.createGain();
      wetGain.gain.value = p.reverb_mix;
      masterGain.connect(convolver);
      convolver.connect(wetGain);
      wetGain.connect(destination);
      allNodes.push(convolver, wetGain);
    }

    return { allNodes };
  }

  /* --- Internal: ADSR scheduling --- */

  _scheduleADSR(param, attack, decay, sustain, release, t0, noteDuration) {
    const a = Math.max(0.001, attack);
    const d = Math.max(0.001, decay);
    const s = Math.max(0, Math.min(1, sustain));
    const r = Math.max(0.001, release);

    param.setValueAtTime(0, t0);
    param.linearRampToValueAtTime(1.0, t0 + a);
    param.linearRampToValueAtTime(s, t0 + a + d);
    // Hold sustain until note off
    const noteOff = t0 + noteDuration;
    param.setValueAtTime(s, noteOff);
    param.linearRampToValueAtTime(0, noteOff + r);
  }

  /* --- Internal: filter envelope --- */

  _scheduleFilterEnvelope(filters, p, t0, noteDuration) {
    const baseCutoff = p.filter_cutoff;
    const envAmount = p.filter_env;
    const a = Math.max(0.001, p.filter_attack);
    const d = Math.max(0.001, p.filter_decay);
    const s = Math.max(0, Math.min(1, p.filter_sustain));
    const r = Math.max(0.001, p.filter_release);

    // Envelope modulates cutoff: base * (1 + envAmount * (envValue - 0.5))
    // envValue goes 0 -> 1 -> sustain -> 0
    const startFreq  = baseCutoff * (1 + envAmount * (0 - 0.5));
    const peakFreq   = baseCutoff * (1 + envAmount * (1 - 0.5));
    const susFreq    = baseCutoff * (1 + envAmount * (s - 0.5));
    const releaseEnd = baseCutoff * (1 + envAmount * (0 - 0.5));

    const clampFreq = (f) => Math.max(20, Math.min(20000, f));

    const noteOff = t0 + noteDuration;

    for (const f of filters) {
      f.frequency.setValueAtTime(clampFreq(startFreq), t0);
      f.frequency.linearRampToValueAtTime(clampFreq(peakFreq), t0 + a);
      f.frequency.linearRampToValueAtTime(clampFreq(susFreq), t0 + a + d);
      f.frequency.setValueAtTime(clampFreq(susFreq), noteOff);
      f.frequency.linearRampToValueAtTime(clampFreq(releaseEnd), noteOff + r);
    }
  }

  /* --- Internal: pulse wave via PeriodicWave --- */

  _createPulseWave(ctx, width) {
    const n = 64;
    const real = new Float32Array(n);
    const imag = new Float32Array(n);
    for (let k = 1; k < n; k++) {
      real[k] = 2 * Math.sin(k * Math.PI * width) / (k * Math.PI);
    }
    return ctx.createPeriodicWave(real, imag, { disableNormalization: true });
  }
}
