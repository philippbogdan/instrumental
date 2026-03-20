# Surge XT Research: Preset Format, Parameter Mapping & Export Strategy

**Date:** 2026-03-20
**Researcher:** R5
**Status:** Complete

---

## 1. Surge XT Preset Format

### File Extensions
- **Main presets**: `.fxp` format (standard VST/AU preset format)
- **Effects presets**: `.srgfx` format
- **Internal structure**: Binary container with embedded XML data

### Format Details
- **Binary header**: FXP parser checks binary header, then unpacks XML and wavetables
- **XML embedding**: XML is embedded within a binary FXP container
- **Not fully documented**: The format is not a public API and subject to change in future versions
- **Future direction**: Surge XT 2 will move away from human-readable XML toward msgpack serialization of C++ objects

### Editing Considerations
- XML can be edited with text editors, but requires careful handling
- Issues arise with special characters, XML formatting, quotes, line breaks, and tag spacing
- If crashes occur with custom XML, debugging with a debug build is necessary
- **Important caveat**: Direct .fxp modification is unsupported and may break in future Surge XT versions

---

## 2. Programmatic Patch Creation

### Python API: surgepy
- **Yes**, Surge XT has a Python API via **surgepy** (Python bindings for Surge)
- **Repository**: [surge-synthesizer/surge-python](https://github.com/surge-synthesizer/surge-python)
- **What it provides**:
  - Complete control of Surge through Python programs and Jupyter notebooks
  - Examples and scripts for developing signal processing algorithms
  - No specific public documentation for preset *generation*, but full parameter manipulation is possible

### Command-Line / Headless Mode
- **Surge XT 1.3+ includes CLI**: `surge-xt-cli` (Linux), `cli` executable (Mac), separate install (Windows)
- **Features**:
  - Runs without GUI (headless operation)
  - OSC (Open Sound Control) support for remote control
  - MIDI Program Changes to load patches
  - Can run as daemon without stdin checks (suitable for embedded systems)
  - Introduced December 2023

### Implications for Export
- **Surgepy**: Can control parameters programmatically, but creating/saving presets from vectors requires manual XML manipulation or reverse-engineering the save format
- **CLI**: Could potentially load presets and render audio in headless workflows, but preset *creation* would still require external tools
- **No off-the-shelf tool found** for generating Surge XT presets from parameter vectors

---

## 3. Surge XT Parameter Architecture

### Oscillators
- **Count**: 3 oscillators per scene
- **Algorithms**: 12 types: Classic, Modern, Wavetable, Window, Sine, FM2, FM3, String, Twist, Alias, S&H Noise, Audio Input
- **Unison**: Up to 16-voice unison at oscillator level (except FM2, FM3, Twist, Audio Input)
- **Modulation**: Hard sync, sub-oscillator options

### Filters
- **Count**: 2 filter units (8 different routing configurations, feedback in 7)
- **Algorithms**: Multimode, K35, Diode Ladder, OB-Xd 12/24 dB/oct, Cutoff Warp, Resonance Warp, Tri-Pole
- **Features**: Self-oscillation with excitation, fast cutoff frequency response

### Envelopes & Modulation
- **Envelope types**:
  - DAHDSR on every LFO
  - Multi-Segment Envelope Generator (MSEG): up to 128 nodes with various curves
  - Formula modulator: Lua scripting for custom outputs
- **LFO count**: 12 total (6 per-voice, 6 global per scene)
- **LFO waveforms**: 7 deformable shapes, step sequencer
- **Modulation depth**: Up to 8 outputs per formula modulator

### Estimated Total Parameter Count
Surge XT has **hundreds of parameters** (far exceeding INSTRUMENTAL's 28). Key categories:
- Oscillator morphing, unison, pitch/modulation
- Filter cutoff, resonance, waveform selection
- Filter envelope modulation depth
- 12 LFOs with shape, rate, depth, phase, key tracking
- Global effects chains

---

## 4. Parameter Mapping: Surge XT → INSTRUMENTAL (28 params)

### Core Mapping Strategy
Surge XT's parameter space is **orders of magnitude larger** than INSTRUMENTAL's 28 parameters. A practical mapping would require:

| INSTRUMENTAL Param | Surge XT Equivalent(s) | Notes |
|---|---|---|
| **osc_type** | Oscillator algorithm (Classic, Modern, Wavetable, etc.) | 12 choices |
| **osc_pitch** | Oscillator pitch + detuning | Can use unison width |
| **osc_morph** | Wavetable morph or oscillator morph | Classic/Modern morph parameter |
| **filter_type** | Filter algorithm selection | 8+ types available |
| **filter_cutoff** | Filter cutoff frequency | Standard parameter |
| **filter_resonance** | Filter resonance/Q | Standard parameter |
| **filter_env_amount** | Filter envelope modulation depth | Envelope → filter cutoff via MSEG |
| **env_attack** | DAHDSR attack time | Standard ADSR analogue |
| **env_decay** | DAHDSR decay time | Standard ADSR analogue |
| **env_sustain** | DAHDSR sustain level | Standard ADSR analogue |
| **env_release** | DAHDSR release time | Standard ADSR analogue |
| **lfo_rate** | LFO rate (1 of 12 LFOs) | Can use multiple LFOs for complexity |
| **lfo_depth** | LFO depth/amount | Standard parameter |
| **reverb_amount** | Global reverb effect | Part of Surge's effect chain |
| Remaining 14 params | Various: effect chains, global modulation, unison spread, etc. | Many-to-one mapping needed |

### Key Insights
1. **Not 1:1 mappable**: Surge XT's parameter space is vastly larger; many INSTRUMENTAL params would map to a small subset of Surge
2. **Reduction needed**: Export would require intelligent parameter selection/decimation (e.g., only use 1-2 of the 12 LFOs)
3. **No academic precedent found**: Hayes 2025 FlowSynth (see section 6) does not appear to export to Surge XT directly

---

## 5. Python API & Integration Potential

### surgepy Capabilities
- Direct parameter access from Python
- Runs Surge in embedded/standalone mode from Jupyter notebooks
- Can control synthesis in real-time
- Examples available but sparse documentation on preset generation

### Integration Approach for INSTRUMENTAL
```python
# Conceptual workflow (not tested)
import surgepy

synth = surgepy.Surge()  # Initialize Surge instance
synth.set_parameter('osc[0].type', 0)  # Set oscillator 0 to Classic
synth.set_parameter('filter[0].cutoff', 4000)  # Set filter cutoff
synth.set_parameter('amp_env.a', 0.1)  # Set attack time
# ... set 26 more parameters ...
synth.save_patch('my_patch.fxp')  # Not confirmed API
```

**Challenge**: No documented method to *save* patches programmatically via surgepy. Parameter manipulation is possible, but exporting as .fxp requires either:
1. XML manipulation directly (fragile, unsupported)
2. Using Surge's GUI to save (defeats the purpose of programmatic export)
3. Reverse-engineering the FXP format (high maintenance burden)

---

## 6. FlowSynth (Hayes 2025): Analysis

### What We Found
- **FlowSynth** is a neural generative model for *timbre synthesis*, not a traditional DAW export tool
- **Paper**: "FlowSynth: Instrument Generation Through Distributional Flow Matching"
- **Focus**: Generating individual notes with controlled pitch, velocity, and timbre using diffusion transformers
- **Target**: Virtual instrument generation (WAV/audio output), not preset format export

### Does NOT Target Surge XT Export
- FlowSynth is a generative *audio* model, not a preset format tool
- No evidence of Surge XT-specific export functionality
- The academic work focuses on timbre quality and consistency, not synth parameter abstraction

### Relevant to INSTRUMENTAL
- **If Hayes 2025 is being referenced**, it may be for:
  - Audio quality benchmarks (timbre consistency)
  - Generative synthesis research (not directly relevant to preset export)
  - Possible misattribution in team context (confirm with team lead)

---

## 7. Preset Generation from Parameter Vectors

### Current State
- **No off-the-shelf tool found** that generates Surge XT presets from parameter vectors
- surgepy exists but lacks documented preset save functionality
- Academic literature does not address this specific problem

### What Would Be Needed
1. **FXP format reverse-engineering**: Detailed XML schema documentation
2. **surgepy enhancement**: Add `save_preset()` or similar method (would require PR to Surge project)
3. **Custom serializer**: Write layer to convert parameter vectors → XML → FXP binary

### Surge Community Status
- Surge XT has an active GitHub community and Discord
- Requests for preset format documentation exist ([Issue #6627](https://github.com/surge-synthesizer/surge/issues/6627))
- Future versions moving away from XML toward msgpack (less human-friendly but more stable)

---

## 8. Summary & Recommendations for INSTRUMENTAL

### ✅ Viable
- **Surge XT is open-source** and has a CLI for headless operation
- **Parameter range** (hundreds) is compatible with machine learning training
- **Active community** with available source code

### ⚠️ Challenging
- **No documented API** for programmatic preset creation
- **Direct FXP editing** is unsupported and brittle
- **Parameter mapping** from 28 INSTRUMENTAL params → Surge XT requires intelligent dimensionality reduction
- **Parameter vectors → presets** requires custom serialization work

### Recommended Path Forward
1. **For research**: Use surgepy to set parameters programmatically + render audio via CLI
   - Viable for training/evaluation workflows
   - Cannot export user-editable presets

2. **For production presets**:
   - Contribute surgepy enhancement upstream (add preset save method)
   - Document FXP/XML format more thoroughly
   - Or switch to Vital (has documented JSON preset format) or another synth with better preset APIs

3. **For parameter mapping**:
   - Implement intelligent reduction: select top 28 parameters from Surge's hierarchy
   - Use perceptually-motivated mapping (e.g., cutoff, resonance, envelope times are universal)
   - Document mapping decisions in code

---

## Sources

- [Surge XT Official](https://surge-synthesizer.github.io/)
- [Surge XT User Manual](https://surge-synthesizer.github.io/manual-xt/)
- [surge-synthesizer/surge GitHub](https://github.com/surge-synthesizer/surge)
- [surge-synthesizer/surge-python GitHub](https://github.com/surge-synthesizer/surge-python)
- [FlowSynth: Instrument Generation Through Distributional Flow Matching](https://arxiv.org/pdf/2510.21667)
- [Update patch file format to not use .fxp · Issue #6627](https://github.com/surge-synthesizer/surge/issues/6627)
- [CDM: Surge XT 1.3 adds OSC, command line, new effects](https://cdm.link/free-surge-xt-1-3/)
