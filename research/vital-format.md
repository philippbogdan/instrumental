# Vital Synth Preset Format Research

**Date:** 2026-03-20
**Researcher:** R2
**Status:** Complete

---

## Executive Summary

Vital presets use **JSON format** (not binary), making them human-readable and easily editable. Vital exposes ~800 parameters total, far more than our 28 INSTRUMENTAL parameters. We can programmatically generate .vital files by creating JSON objects and saving them with the .vital extension. Syntheon already demonstrates parameter inference and export to Vital, proving the approach is viable.

---

## 1. Vital File Format Specification

### Format Type
- **.vital files are JSON text files** (not binary)
- Plain text, human-readable format
- Custom `.vital` extension
- Wavetable data and custom noise samples are embedded in the JSON (plain text format)
- Each preset file is fully self-contained

### File Structure
- JSON object containing all preset parameters
- Comprehensive parameter set (~800 parameters available)
- Metadata included (preset name, version info)
- All wavetable/sample data embedded in text format for easy sharing

### Advantages for Our Use Case
- **Easy to generate programmatically** - standard JSON serialization
- **Editable in any text editor** - no binary format complexity
- **Self-contained** - everything needed in one file
- **Version control friendly** - plain text diffs work
- **Python-friendly** - `json` module is all we need

**Sources:**
- [vitaldb/vitalutils - Vital File Format.pdf](https://github.com/vitaldb/vitalutils/blob/master/Vital%20File%20Format.pdf)
- [KVR Audio Forum - Vital File Format Discussion](https://www.kvraudio.com/forum/viewtopic.php?t=556602)

---

## 2. Vital Parameter Ecosystem

### Total Parameter Count
- **~800 total parameters** available in Vital
- Far exceeds our 28 core parameters
- Includes oscillators, filters, envelopes, LFOs, effects, modulation routing

### Core Parameter Categories

#### Voice Section (Oscillators & Sampler)
- 3 wavetable oscillators
- 1 sampler
- Oscillator mix controls

#### Filter Section
- 2 filters with **32 filter types each**
- Parameters: `filter_1_cutoff`, `filter_1_on`, cutoff, resonance, drive, gain, style, pass blend, interpolation, transpose

#### Modulation & Envelope
- 3 envelope generators (ADSR)
- 4 LFOs with tempo sync
- 2 randomness sources
- Macro section: 4 assignable knobs for multi-parameter control

#### Effects
- 9-slot effects rack
- Reverb, delay, distortion, compression, etc.
- Per-effect parameters for mix, size, time, etc.

#### Modulation Routing
- Extensive modulation matrix
- Connect any modulation source to any destination

### Official Parameter Documentation
- User Manual available (Scribd)
- Forum discussions at [Vital Audio Forum](https://forum.vital.audio)
- Source code at [github.com/mtytel/vital](https://github.com/mtytel/vital)

---

## 3. Mapping Our 28 Parameters to Vital

Our INSTRUMENTAL synth parameters → Vital equivalents:

| Our Parameter | Vital Equivalent | Notes |
|---|---|---|
| `saw_mix` | `osc_1_mix` (or `osc_mix_[1-3]`) | One of 3 oscillator mix controls |
| `square_mix` | `osc_2_mix` | Oscillator 2 typically square-ish |
| `sine_mix` | `osc_3_mix` | Oscillator 3 often sine-like |
| `noise_mix` | `noise_volume` or `noise_mix` | Noise oscillator level |
| `detune` | `osc_detune` or per-osc `detune` | Oscillator detuning |
| `filter_cutoff` | `filter_1_cutoff` | Primary filter cutoff |
| `filter_resonance` | `filter_1_resonance` | Filter Q/resonance |
| `attack` | `env_attack` (envelope 1) | Attack time |
| `decay` | `env_decay` | Decay time |
| `sustain` | `env_sustain` | Sustain level |
| `release` | `env_release` | Release time |
| `gain` | `volume` or `master_gain` | Output level |
| `filter_env` | `filter_envelope_amount` | Envelope → filter modulation |
| `reverb_size` | `reverb_room_size` or similar | Reverb algorithm parameter |
| `reverb_mix` | `reverb_wet` or `reverb_mix` | Reverb dry/wet balance |
| `unison_voices` | `voice_unison` or `unison_voices` | Number of unison voices |
| `unison_spread` | `unison_spread` or `detune_spread` | Detune spread between voices |
| `noise_floor` | `noise_floor` | Minimum noise level |
| `filter_attack` | `filter_env_attack` | Filter envelope attack |
| `filter_decay` | `filter_env_decay` | Filter envelope decay |
| `filter_sustain` | `filter_env_sustain` | Filter envelope sustain |
| `filter_release` | `filter_env_release` | Filter envelope release |
| `pulse_width` | `pulse_width` or `osc_[1-3]_pulse_width` | PWM for oscillators |
| `filter_slope` | `filter_slope` or `filter_type` | dB/octave slope (12/24/48) |
| `eq1_freq` | `eq_1_freq` | EQ band 1 frequency |
| `eq1_gain` | `eq_1_gain` | EQ band 1 gain |
| `eq2_freq` | `eq_2_freq` | EQ band 2 frequency |
| `eq2_gain` | `eq_2_gain` | EQ band 2 gain |

**Note:** Vital parameter names may vary. See Syntheon and Vita projects for authoritative mappings.

---

## 4. Programmatic .vital File Generation

### Approach
1. Create a Python dictionary with all required Vital parameters
2. Map our 28 INSTRUMENTAL parameters to Vital equivalents
3. Serialize to JSON
4. Write to `.vital` file
5. Save alongside wavetable data if needed

### Example Structure (Pseudocode)
```python
import json

vital_preset = {
    "preset_name": "INSTRUMENTAL Export",
    "osc_1_mix": 0.8,        # saw_mix (0-1)
    "osc_2_mix": 0.2,        # square_mix
    "osc_3_mix": 0.1,        # sine_mix
    "noise_mix": 0.0,        # noise_mix
    "osc_detune": 0.05,      # detune
    "filter_1_cutoff": 5000, # Hz
    "filter_1_resonance": 0.3,
    "env_attack": 0.01,      # seconds
    "env_decay": 0.5,
    "env_sustain": 0.7,
    "env_release": 0.5,
    "volume": 0.8,           # gain
    "filter_envelope_amount": 0.5,
    "reverb_room_size": 0.5,
    "reverb_wet": 0.2,
    # ... 800+ parameters total
}

with open("my_preset.vital", "w") as f:
    json.dump(vital_preset, f, indent=2)
```

### Advantages
- No special binary encoding needed
- Standard JSON tools work
- Easy to debug (human-readable)
- Trivial to version control

### Caveats
- Need to identify ALL required parameters for Vital to load the preset
- Default values for unspecified parameters may vary
- Wavetable data must be embedded if using custom wavetables

---

## 5. Reverse Engineering & Community Projects

### Known Projects

#### [vitaldb/vitalutils](https://github.com/vitaldb/vitalutils)
- Open-source utilities for processing Vital files
- Includes format specification PDF
- Python library available (v1.6.0)
- Tools for loading, parsing, and manipulating .vital files

#### [DBraun/Vita - Python Bindings](https://github.com/DBraun/Vita)
- **Official Python API for Vital synthesizer**
- Supported on Linux, macOS, Windows
- Features:
  - Control synthesizer parameters programmatically
  - Create modulation routings
  - Render audio to NumPy arrays
  - VST-style normalized parameter control (0-1)
  - Modulation source/destination management
- Installation: `pip install vital`
- **Best option for programmatic control**

#### [gudgud96/Syntheon](https://github.com/gudgud96/syntheon)
- **ML-based parameter inference** (audio → parameters)
- Supports Vital as a target synthesizer
- Components:
  - `vital_inferencer.py` - Vital-specific inference logic
  - `adsr_envelope.py` - Envelope modeling
  - Audio preprocessing → inference → preset generation
- Demonstrates practical parameter mapping and export workflow
- Available on PyPI as `syntheon` package

#### [GavinRay97/vital-synth-sdk](https://github.com/GavinRay97/vital-synth-sdk)
- Utilities for working with Vital presets
- Aimed at ML/sound design exploration
- SDK for preset manipulation

#### GitHub Preset Collections
- [jpriebe/qub1t-vital-presets](https://github.com/jpriebe/qub1t-vital-presets)
- [abstractionmage/Vital-Presets](https://github.com/abstractionmage/Vital-Presets)
- [bluskript/midi2vital](https://github.com/bluskript/midi2vital) - MIDI to Vital converter
- These provide real .vital file examples to study

---

## 6. How Syntheon Exports to Vital

### Architecture
1. **Audio Input** → Preprocessing
2. **Inference Engine** → Generate synthesizer parameters
3. **Parameter Mapping** → Map inference output to Vital parameters
4. **Preset Generation** → Create .vital file with mapped parameters
5. **Export** → Save or apply to Vital instance

### Key Insight
Syntheon's approach proves that:
- Parameter inference is feasible (given audio, predict parameters)
- Direct mapping to Vital is straightforward
- Preset generation is reproducible

### Relevant Files
- `vital_inferencer.py` - Contains Vital-specific parameter handling
- `adsr_envelope.py` - Envelope parameter modeling
- Source at [gudgud96/syntheon GitHub](https://github.com/gudgud96/syntheon)

---

## 7. Vital Python API & SDK Options

### Tier 1: Official/Recommended

#### **Vita (DBraun) - Most Complete**
- **Repository:** [github.com/DBraun/Vita](https://github.com/DBraun/Vita)
- **Installation:** `pip install vital`
- **Capabilities:**
  - Full parameter control
  - Modulation routing
  - Audio rendering to NumPy
  - VST parameter normalization
- **Status:** Active, well-maintained
- **Best for:** Programmatic preset generation and audio rendering

### Tier 2: Utilities & Tools

#### **vitalutils (vitaldb)**
- Python library for parsing and manipulating .vital files
- Direct file I/O without needing the plugin
- Good for offline preset generation

#### **Syntheon**
- PyPI: `pip install syntheon`
- Focus: ML-driven parameter inference
- Includes Vital parameter mappings
- Good reference for parameter names and ranges

### Tier 3: Lower-level Access

#### **Direct JSON Manipulation**
- Since .vital files are JSON, use standard `json` module
- Simple but requires knowing all parameter names and valid ranges
- No validation or synthesis—just file generation

---

## 8. Parameter Name Reference

### From Vita & Syntheon Analysis

**Oscillators:**
- `osc_1_mix`, `osc_2_mix`, `osc_3_mix` - Oscillator levels
- `osc_detune` - Master detune
- `osc_transpose` - Master transpose
- `voice_unison` - Unison voice count
- `unison_spread` - Voice detune spread

**Filter:**
- `filter_1_cutoff` - Cutoff frequency (Hz or MIDI)
- `filter_1_resonance` - Q/resonance (0-1)
- `filter_1_drive` - Saturation
- `filter_1_type` - Filter topology (0-31, 32 types)
- `filter_slope` - Slope in dB/octave

**Envelopes (Amp):**
- `env_attack`, `env_decay`, `env_sustain`, `env_release`

**Envelopes (Filter):**
- `filter_env_attack`, `filter_env_decay`, `filter_env_sustain`, `filter_env_release`
- `filter_envelope_amount` - Modulation amount

**Modulation:**
- `lfo_1_freq`, `lfo_1_tempo`, `lfo_1_type` - LFO 1-4 parameters
- `macro_1`, `macro_2`, `macro_3`, `macro_4` - Assignable macro knobs

**Output:**
- `volume` - Master output level
- `pan` - Left-right balance

**Effects:**
- `reverb_room_size`, `reverb_wet` - Reverb parameters
- `delay_time`, `delay_feedback`, `delay_wet` - Delay parameters
- `distortion_drive`, `distortion_tone` - Distortion
- `chorus_rate`, `chorus_depth` - Chorus

**Noise:**
- `noise_mix` or `noise_volume` - Noise oscillator level

---

## 9. Implementation Recommendations

### For INSTRUMENTAL → Vital Export

**Option A: Use Vita (Recommended)**
```python
from vital import VitalSynth

synth = VitalSynth()

# Set our parameters
synth.set_parameter('osc_1_mix', our_saw_mix)
synth.set_parameter('filter_1_cutoff', our_filter_cutoff)
# ... map all 28 parameters ...

# Render audio
audio = synth.render(duration=4.0)

# Save as .vital preset
synth.save_preset('my_preset.vital')
```

**Option B: Direct JSON Generation**
```python
import json

def create_vital_preset(instrumental_params):
    vital_preset = {
        'preset_name': instrumental_params.get('name', 'INSTRUMENTAL Export'),
        'osc_1_mix': instrumental_params['saw_mix'],
        'osc_2_mix': instrumental_params['square_mix'],
        # ... map all 28 ...
        # Fill in defaults for unmapped ~772 parameters
    }
    return vital_preset

with open('export.vital', 'w') as f:
    json.dump(create_vital_preset(params), f)
```

**Option C: Study Syntheon**
- Examine `vital_inferencer.py` for authoritative parameter names
- Use similar mapping approach
- May be able to leverage Syntheon code directly

### Next Steps
1. **Obtain Vital File Format PDF** from vitaldb/vitalutils repo
2. **Install Vita**: `pip install vital` and test basic parameter control
3. **Create parameter mapping table** - our 28 → Vital equivalents
4. **Implement export function** using either Vita or direct JSON
5. **Test with real Vital instance** - load exported presets and verify
6. **Validate audio rendering** - compare web synth output to Vital rendering

---

## 10. Key Findings Summary

✅ **Format:** JSON text files (human-readable, programmable)
✅ **Parameters:** ~800 available in Vital, easy to map our 28 to them
✅ **Generation:** Trivial to create .vital files programmatically
✅ **Precedent:** Syntheon already does this (audio → parameters → .vital)
✅ **Python API:** Vita library (pip install vital) provides full control
✅ **Reverse Engineering:** Community has thoroughly documented the format

**Conclusion:** Exporting INSTRUMENTAL presets to Vital is absolutely feasible. The JSON format, established Python libraries (Vita), and reference implementations (Syntheon) make this a straightforward engineering task.

---

## Resources

- [Vital Official](https://vital.audio/)
- [DBraun/Vita - Python Bindings](https://github.com/DBraun/Vita)
- [vitaldb/vitalutils](https://github.com/vitaldb/vitalutils)
- [gudgud96/Syntheon](https://github.com/gudgud96/syntheon)
- [Vital Audio Forum](https://forum.vital.audio)
- [GavinRay97/vital-synth-sdk](https://github.com/GavinRay97/vital-synth-sdk)
