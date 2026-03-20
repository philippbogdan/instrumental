# INSTRUMENTAL Export Strategy: Synthesis & Recommendations

**Date:** 2026-03-20
**Status:** Final
**Input:** 5 research files (VST presets, Vital format, MIDI/audio export, WebAudio-to-DAW bridges, Surge XT)

---

## 1. Comparison Matrix

INSTRUMENTAL recovers **28 subtractive synth parameters** (defined in `src/synth.py:17-46`):
saw_mix, square_mix, sine_mix, noise_mix, detune, filter_cutoff, filter_resonance,
attack, decay, sustain, release, gain, filter_env, reverb_size, reverb_mix,
unison_voices, unison_spread, noise_floor, filter_attack, filter_decay, filter_sustain,
filter_release, pulse_width, filter_slope, eq1_freq, eq1_gain, eq2_freq, eq2_gain.

All parameters are normalized 0-1 internally, then denormalized to real ranges.

### Approach Comparison

| Approach | Feasibility | Effort | Sound Quality | DAW Compatibility | User Experience |
|---|---|---|---|---|---|
| **A. Vital .vital preset** | **Excellent** - JSON format, `json` stdlib only | **1-2 days** | **Exact** - direct param mapping, 28/800 params | Vital (free, cross-platform) | **Great** - download .vital, open in Vital, play |
| **B. SFZ multi-sampling** | **Good** - render WAVs + text SFZ file | **3-5 days** | **Excellent** - pre-rendered audio captures everything | Ableton Sampler, Reaper, Studio One, FL Studio (via plugin) | **Good** - import SFZ + samples folder into DAW sampler |
| **C. VST3 .vstpreset** | **Medium** - requires plugin-specific binary state blob | **5-10 days** | **Varies** - depends on target synth mapping quality | All major DAWs (Ableton, Logic, Cubase, etc.) | **Medium** - must own the target synth plugin |
| **D. Surge XT .fxp** | **Poor** - no preset save API, fragile XML-in-binary format | **10+ days** | **Approximate** - 28 params vs hundreds, lossy mapping | DAWs with Surge XT installed (free) | **Poor** - requires Surge XT, mapping is lossy |
| **E. MIDI + WAV** | **Excellent** - trivial to implement | **0.5 days** | **Exact** - pre-rendered audio | Every DAW | **Poor** - not a reusable instrument, just audio + notes |
| **F. Faust DSP rewrite** | **Poor** - requires rewriting synth in functional DSP language | **20+ days** | **Exact** - same DSP, compiled natively | All DAWs (compiles to VST3/AU/CLAP) | **Excellent** - native plugin, zero friction |
| **G. CLAP/WAM plugin** | **Poor** - requires C++/Rust rewrite | **30+ days** | **Exact** | CLAP: Bitwig, Reaper, Studio One; WAM: web DAWs only | **Excellent** (if built) |
| **H. DecentSampler .dspreset** | **Good** - XML format, same samples as SFZ | **1 day** (after SFZ done) | **Excellent** - same as SFZ | Any DAW via free DecentSampler plugin | **Good** - requires free plugin install |

### Scoring Summary (1-5, higher = better)

| Approach | Feasibility | Effort (inverse) | Sound Quality | DAW Compat | UX | **Total** |
|---|---|---|---|---|---|---|
| **A. Vital preset** | 5 | 5 | 4 | 3 | 5 | **22** |
| **B. SFZ sampling** | 4 | 4 | 5 | 4 | 4 | **21** |
| **E. MIDI + WAV** | 5 | 5 | 5 | 5 | 2 | **22** |
| **H. DecentSampler** | 4 | 4 | 5 | 4 | 4 | **21** |
| **C. VST3 preset** | 3 | 3 | 3 | 5 | 3 | **17** |
| **F. Faust rewrite** | 1 | 1 | 5 | 5 | 5 | **17** |
| **D. Surge XT** | 2 | 2 | 2 | 3 | 2 | **11** |
| **G. CLAP/WAM** | 1 | 1 | 5 | 3 | 5 | **15** |

---

## 2. Top 2 Recommendations

### Recommendation 1: Vital .vital Preset Export (PRIMARY)

**What to build:** A Python function that maps INSTRUMENTAL's 28 recovered parameters to Vital's JSON preset format and writes a `.vital` file the user can download from the web app.

**Why Vital wins:**
- **JSON format** - `.vital` files are plain JSON text. No binary encoding, no reverse engineering, no fragile format.
- **Direct parameter mapping** - Our 28 subtractive synth params (oscillators, filter, ADSR, reverb, EQ) map naturally to Vital's parameter set. Vital is a subtractive/wavetable synth with the same architecture.
- **Free and cross-platform** - Vital is free (basic tier), runs on macOS/Windows/Linux. Zero cost barrier for users.
- **Proven approach** - Syntheon (our vendored dependency at `vendor/syntheon/`) already demonstrates audio-to-Vital parameter export via `vital_inferencer.py`.
- **Vita Python API** exists (`pip install vital`) for programmatic control and audio rendering if needed for verification.

**Specific code changes needed:**

1. **New file: `src/vital_export.py`**
   - Function `instrumental_to_vital(params: dict) -> dict` mapping our 28 params to Vital JSON keys
   - Function `write_vital_preset(params: dict, path: str)` serializing to `.vital` file
   - Parameter mapping table (see below)

2. **Update: `app/server.py`**
   - Add `/api/export/vital` endpoint
   - Takes current recovered parameters, generates `.vital` file, returns as download

3. **Update: frontend**
   - Add "Export to Vital" button on the results page
   - Triggers download of `.vital` file

**Parameter mapping (INSTRUMENTAL -> Vital):**

| INSTRUMENTAL param | Range | Vital param | Notes |
|---|---|---|---|
| `saw_mix` | 0-1 | `osc_1_level` | Osc 1 = saw wavetable |
| `square_mix` | 0-1 | `osc_2_level` | Osc 2 = square wavetable |
| `sine_mix` | 0-1 | `osc_3_level` | Osc 3 = sine wavetable |
| `noise_mix` | 0-0.5 | `sample_level` | Noise via sampler section |
| `detune` | -24 to 24 | `osc_1_transpose` | Semitone detune |
| `filter_cutoff` | 200-16000 Hz | `filter_1_cutoff` | Map Hz to Vital's MIDI note scale |
| `filter_resonance` | 0-0.95 | `filter_1_resonance` | Direct 0-1 mapping |
| `attack` | 0.001-0.5 | `env_1_attack` | Seconds, may need log scaling |
| `decay` | 0.001-1.0 | `env_1_decay` | Seconds |
| `sustain` | 0-1 | `env_1_sustain` | Direct mapping |
| `release` | 0.001-1.0 | `env_1_release` | Seconds |
| `gain` | 0-1 | `volume` | Master output |
| `filter_env` | -1 to 1 | Modulation routing | Env -> filter cutoff amount |
| `reverb_size` | 0.01-0.99 | `reverb_size` | Direct mapping |
| `reverb_mix` | 0-0.8 | `reverb_dry_wet` | Direct mapping |
| `unison_voices` | 1-7 | `osc_1_unison_voices` | Integer cast |
| `unison_spread` | 0-0.5 | `osc_1_unison_detune` | Detune spread |
| `noise_floor` | 0-0.1 | (add to noise level) | Additive noise floor |
| `filter_attack` | 0.001-0.5 | `env_2_attack` | Filter envelope |
| `filter_decay` | 0.001-1.0 | `env_2_decay` | Filter envelope |
| `filter_sustain` | 0-1 | `env_2_sustain` | Filter envelope |
| `filter_release` | 0.001-1.0 | `env_2_release` | Filter envelope |
| `pulse_width` | 0.1-0.9 | `osc_2_frame` | Wavetable frame position for square |
| `filter_slope` | 4-48 dB/oct | `filter_1_model` | Map to filter type (12/24/48 dB) |
| `eq1_freq` | 500-4500 Hz | `eq_low_cutoff` | EQ band 1 |
| `eq1_gain` | -6 to 6 dB | `eq_low_gain` | EQ band 1 |
| `eq2_freq` | 2000-10000 Hz | `eq_high_cutoff` | EQ band 2 |
| `eq2_gain` | -6 to 6 dB | `eq_high_gain` | EQ band 2 |

**Python libraries:**
- `json` (stdlib) - serialize preset to `.vital` file
- Reference: `vendor/syntheon/syntheon/inferencer/vital/vital_inferencer.py` for authoritative Vital param names

**Expected output:** `preset_name.vital` - a JSON file loadable in Vital synth.

**DAW compatibility:** Any DAW where Vital is installed as a VST3/AU/CLAP plugin: Ableton Live, Logic Pro, FL Studio, Reaper, Cubase, Bitwig, Studio One.

---

### Recommendation 2: SFZ Multi-Sample Export (SECONDARY)

**What to build:** A Python pipeline that renders the matched synth across the keyboard (every semitone or every 3rd semitone, 2-3 velocity layers), packages the WAV samples with an SFZ mapping file, and lets the user download a ZIP.

**Why SFZ:**
- **No external synth required** - Works with any DAW's built-in sampler (Ableton Sampler, Reaper ReaSampler, Studio One Sampler).
- **Perfect sound reproduction** - Pre-rendered audio captures EXACTLY what the user heard, with all 28 parameters baked in.
- **Text-based format** - `.sfz` files are human-readable text defining sample-to-key mappings.
- **Covers the "no Vital" case** - Users who don't want to install Vital can still use the matched sound.

**Specific code changes needed:**

1. **New file: `src/sfz_export.py`**
   - Function `render_samples(params: dict, sr=44100) -> dict[int, np.ndarray]` - render synth at each MIDI note
   - Function `write_sfz(samples_dir: str, output_path: str)` - generate SFZ mapping text
   - Function `package_sfz_zip(params: dict, output_path: str)` - render + map + ZIP

2. **Update: `app/server.py`**
   - Add `/api/export/sfz` endpoint
   - Generates samples, creates SFZ, ZIPs everything, returns as download

3. **Rendering strategy:**
   - Render every 3 semitones (C, Eb, F#, A) across 5 octaves = ~20 samples
   - Each sample: 2-second note (attack + sustain + release tail)
   - 24-bit 44.1kHz WAV files
   - SFZ maps each sample to its +-1 semitone range

**Python libraries:**
- `numpy` / `torch` - render synth audio (already in codebase via `src/synth.py`)
- `soundfile` or `scipy.io.wavfile` - write WAV files
- `zipfile` (stdlib) - package SFZ + samples
- `sf-creator` (optional) - auto-generate SFZ from named samples

**Expected output:** `instrumental_export.zip` containing:
```
instrumental_export/
  preset.sfz
  samples/
    C2.wav
    Eb2.wav
    ...
    A6.wav
```

**DAW compatibility:** Ableton Live (Sampler), Reaper (ReaSampler), Studio One (Sampler), FL Studio (via Sforzando/DirectWave), Logic Pro (via free Sforzando AU). Kontakt can also import SFZ.

---

## 3. What NOT to Pursue (and Why)

### Do NOT build: Surge XT preset export
- **No preset save API** - surgepy can manipulate parameters but cannot save `.fxp` files programmatically.
- **Fragile format** - Binary FXP with embedded XML; format is undocumented, unsupported for external editing, and will change to msgpack in Surge XT 2.
- **Lossy mapping** - Surge has hundreds of parameters; mapping our 28 to its architecture requires arbitrary decisions that degrade sound quality.
- **Effort vs. reward** - 10+ days of reverse engineering for a format that's actively being deprecated.

### Do NOT build: VST3 .vstpreset for arbitrary synths
- **Plugin-specific state blobs** - Each VST3 plugin (Serum, Massive, Pigments) defines its own binary parameter encoding within the `.vstpreset` container. There is no universal parameter format.
- **Reverse engineering per synth** - Supporting Serum alone would require reverse-engineering its proprietary `.SerumPreset` format (partially done by community, but fragile and legally gray).
- **Vital already IS a VST3** - If we export to Vital's native `.vital` format, users load it in any DAW via Vital's VST3/AU plugin. We get VST3 DAW compatibility for free.

### Do NOT build: Faust DSP rewrite or native CLAP/VST plugin
- **Wrong scope for a hackathon** - Rewriting the WebAudio/PyTorch synth in Faust or C++ is a 20-30 day project.
- **Diminishing returns** - The synth is a 28-parameter subtractive engine. A native plugin is overkill when Vital preset export gives the same result.
- **Future option** - If INSTRUMENTAL grows beyond hackathon scope, Faust rewrite is the right long-term path for cross-platform native plugins. Not now.

### Do NOT build: WAM (Web Audio Module) packaging
- **Web DAWs only** - WAM plugins only run in browser-based DAWs (openDAW, WebDAW). No musician uses these for production.
- **Our synth already runs in the browser** - The web app IS the instrument. WAM packaging adds complexity for zero new capability.

### Do NOT build: SF2 SoundFont
- **SFZ is strictly better** - Text-based, more flexible, better Python tooling. SF2 is binary legacy format with no advantages over SFZ for our use case.

---

## 4. Implementation Priority

| Priority | What | Effort | Impact |
|---|---|---|---|
| **P0** | Vital `.vital` preset export | 1-2 days | High - reusable instrument in any DAW |
| **P1** | SFZ multi-sample export | 3-5 days | Medium - covers non-Vital users |
| **P2** | MIDI + WAV quick export | 0.5 days | Low - fallback, not reusable |
| **P3** | DecentSampler `.dspreset` | 1 day (after SFZ) | Low - niche audience |

**For the hackathon: build P0 (Vital export).** It delivers the most value in the least time. SFZ is the logical next step post-hackathon.

---

## 5. Key Technical Insights

### Why Vital is the ideal target
1. **Architecture match** - Vital is a subtractive synth with wavetable oscillators, multimode filter, dual ADSR, effects. Our 28 parameters map to a natural subset of Vital's ~800 params.
2. **JSON format** - No binary serialization. `json.dump()` is the entire export pipeline.
3. **Syntheon precedent** - Our vendored `vendor/syntheon/` package already implements audio->Vital parameter inference, proving the mapping works.
4. **Free tier** - Vital Basic is free. No paywall for users to use our export.

### SFZ as the universal fallback
1. **Pre-rendered = perfect** - No parameter mapping loss. The exported samples ARE the matched sound.
2. **No external synth needed** - Works with built-in DAW samplers.
3. **Trade-off** - Larger file size (20-50 MB ZIP vs. 10 KB `.vital` file), less editable.

---

*Synthesis complete. Recommendations: (1) Vital preset export as primary, (2) SFZ multi-sampling as secondary.*
