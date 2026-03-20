# VST/AU Plugin Preset Formats Research

**Date**: 2026-03-20
**Context**: INSTRUMENTAL recovers 28 synth parameters from audio. This research explores how to export these parameters into formats that musicians can load into their DAW.

## Executive Summary

INSTRUMENTAL can export its 28 recovered parameters in multiple formats:

1. **VST3 `.vstpreset`** (recommended modern standard)
   - XML-based, well-documented by Steinberg
   - Python library available (`vstpreset`)
   - Cross-platform support
   - Future-proof (VST2 is deprecated)

2. **VST2 `.fxp`** (legacy but widely used)
   - Binary format, plugin-specific state blobs
   - Reverse-engineered for Serum
   - Python parser libraries exist
   - Limited generalization across synths

3. **AU `.aupreset`** (macOS native)
   - Plist-based, Apple's Audio Unit standard
   - Limited to macOS
   - Well-documented by Apple

4. **Open format (JSON)** (plugin-agnostic)
   - Store parameters in human-readable JSON
   - Let users import into their synth of choice
   - Most flexible, plugin-agnostic approach

---

## 1. VST3 Preset Format (`.vstpreset`)

### Overview
The modern VST3 standard introduced by Steinberg. File extension: `.vstpreset`

### File Structure
```
VST 3 Preset File Format Definition
===================================

0   +---------------------------+
    | HEADER                    |
    | header id ('VST3')        |       4 Bytes
    | version                   |       4 Bytes (int32)
    | ASCII-encoded class id    |       32 Bytes
 +--| offset to chunk list      |       8 Bytes (int64)
 |  +---------------------------+
 |  | DATA AREA                 |<-+
 |  | data of chunks 1..n       |  |
 |  ...                       ...  |
 |  |                           |  |
 +->+---------------------------+  |
    | CHUNK LIST                |  |
    | list id ('List')          |  |    4 Bytes
    | entry count               |  |    4 Bytes (int32)
    +---------------------------+  |
```

### Key Points
- **Chunks**: Each chunk has a type identifier, size, and data
- **Plugin ID**: The `class id` (32 bytes) uniquely identifies which plugin this preset is for
- **State data**: Plugins store their parameter state in chunk format
- **Host-managed**: The host DAW manages loading/saving, plugins implement `IComponent::getState()`

### Python Libraries
- **`janminor/python-vstpreset`** (GitHub, 11 stars)
  - Actively maintained (last push 2026-03-11)
  - Can read/write VST3 preset files
  - Can convert VST2 `.fxp` to `.vstpreset`
  - MIT/GPL2 licensed
  - Usage: `vstpreset.read()`, `vstpreset.write()`

### Advantages
- ✅ Official Steinberg standard
- ✅ Well-documented
- ✅ Cross-platform
- ✅ Modern, future-proof
- ✅ Python library available
- ✅ Extensible chunk system

### Disadvantages
- ❌ Plugin must implement the preset interface
- ❌ Each plugin defines its own parameter encoding within chunks
- ❌ Not human-readable (binary format)

---

## 2. VST2 Preset Format (`.fxp` / `.fxb`)

### Overview
Legacy Steinberg VST2 format. Still widely used despite being deprecated.
- `.fxp` = Single preset/patch
- `.fxb` = Bank of presets

### File Structure (Simplified)
```
[Header]
- 'CcnK' magic (for chunk data) or 'FxCk' (for parameter-based)
- Version number (usually 1)
- Plugin ID (4-byte unique identifier, e.g., "XfsX" for Serum)
- Size

[Data]
- Plugin-specific blob (state data)
OR
- Parameter array (if not using chunks)

[Optional]
- Plugin name
- Preset name
```

### Magic Numbers
- `'FxCk'`: Preset with chunk data
- `'FPCh'`: Preset with parameters
- `'FxBk'`: Bank with chunk data
- `'FBCh'`: Bank with parameters

### Key Characteristic
**Plugin-specific opaque data**: The `.fxp` file contains a binary blob that only the plugin knows how to deserialize. This makes generalized preset generation difficult.

### Python Libraries
1. **`demberto/fxp`** (GitHub, 10 stars)
   - VST2.x FXP preset parser
   - MIT licensed
   - Can parse headers and metadata
   - Cannot reconstruct opaque plugin state without reverse-engineering

2. **`CharlesHolbrow/vst2-preset-parser`** (GitHub, 10 stars)
   - JavaScript/Node.js
   - Can parse FXP/FXB structure
   - Returns magic bytes, version, plugin ID, parameter data

### Serum-Specific Reverse Engineering
**`0xdevalias/reverse-engineering-serum-preset`** (GitHub gist, comprehensive)
- Serum v1: `.fxp` with opaque binary state
- Serum v2: `.SerumPreset` (new proprietary format)
- Both documented and partially reverse-engineered
- Parameter order, default values documented
- Cannot be easily mapped to other synths

### Limitations
- ❌ Plugin-specific binary encoding (no universal standard)
- ❌ Requires reverse-engineering per synth
- ❌ Deprecated (VST2 is end-of-life)
- ❌ No standardized parameter mapping

---

## 3. Audio Unit (AU) Preset Format (`.aupreset`)

### Overview
Apple's native plugin format for macOS. Stored as `.aupreset` files in `~/Library/Audio/Presets/[manufacturer]/[plugin]/`

### File Structure
- **Plist-based** (XML property list)
- Contains plugin name, version, and state data
- `aupreset` files are actually plist files with binary data

### Presets Storage Locations
```
~/Library/Audio/Presets/[manufacturer]/[plugin-name]/
```

### Apple Documentation
- **Technical Note TN2157**: Official guide for saving/restoring AU presets
- Presets have two forms:
  1. **Factory presets**: Hard-coded in the plugin binary
  2. **User presets**: Stored as `.aupreset` files on disk

### Python Support
- Limited direct Python libraries
- Can be manipulated via plistlib (Python stdlib)
- Ruby tool exists: `jipumarino/presets2aupreset` for converting other formats to AU

### Advantages
- ✅ Native macOS integration
- ✅ Standardized by Apple
- ✅ Plist format is text-readable
- ✅ Well-documented

### Disadvantages
- ❌ macOS only
- ❌ Still plugin-specific state blobs (binary data within plist)
- ❌ Limited cross-platform use
- ❌ Fewer Python tools than VST

---

## 4. Vital Synth Preset Format (`.vital`)

### Overview
Vital is an open-source spectral wavetable synthesizer. Vital preset files are JSON-based.

### Format
- **File extension**: `.vital`
- **Content**: JSON with nested structure
- **Human-readable**: Yes, can be edited in a text editor
- **Version tracking**: Includes synth version and patch format version strings

### Structure (Approximation)
```json
{
  "synth_version": "1.0.5",
  "patch_format_version": "1.0",
  "settings": {
    "oscillators": [...],
    "filters": [...],
    "envelopes": [...],
    "lfos": [...],
    "effects": [...],
    "modulation": [...]
  }
}
```

### Advantages
- ✅ Open-source synth (code available)
- ✅ JSON format (human-readable, parseable)
- ✅ Full parameter documentation available
- ✅ Can be edited and regenerated programmatically
- ✅ Free and widely available

### Limitations
- ❌ Requires Vital to be installed to load
- ❌ Not a universal synth (specific to Vital)
- ❌ Format versioning required for compatibility

### Reverse Engineering
- Vital file format documented in `vitaldb/vitalutils` GitHub repo
- PDF specification available
- Forum post shows JSON structure can be manually edited
- Parameters are well-named and logical

### Use Case for INSTRUMENTAL
**Best for prototyping/testing**: Generate `.vital` presets to test if INSTRUMENTAL's 28 parameters can be meaningfully mapped to a real synth. Vital's open-source nature makes it ideal for proof-of-concept.

---

## 5. Serum Preset Formats

### Serum v1 (Legacy)
- **Format**: `.fxp` (VST2 preset)
- **Status**: Deprecated (Serum 2 released 2025)
- **Reverse-engineering**: Possible but complex
- **Header**: Starts with `43 63 6E 4B 00 00 11` (hex)
- **Parameters**: ~60+ parameters (oscillator mix, filter cutoff, ADSR, effects, modulation)

### Serum v2 (Current)
- **Format**: `.SerumPreset` (new proprietary format)
- **Status**: Current standard
- **Structure**: Partially reverse-engineered by `0xdevalias`
- **Python tools**: `KennethWussmann/serum-preset-packager` can convert `.SerumPreset` to/from JSON

### Parameter Mapping
Serum has ~60-80 parameters across:
- 3 oscillators (wavetable, spectral, sampler modes)
- 2 filters (cutoff, resonance, drive)
- 2 filter envelopes
- 2 LFOs
- Effects chain (reverb, delay, distortion, etc.)
- Global controls (pitch, unison, pan, mix)

### Challenges
- ❌ Proprietary formats
- ❌ Limited official documentation
- ❌ Reverse-engineering required
- ❌ Updates break compatibility

---

## 6. Python Libraries Summary

### For VST3 (Recommended)
| Library | Stars | Language | Status | Use Case |
|---------|-------|----------|--------|----------|
| `janminor/python-vstpreset` | 11 | Python | Active (2026) | Read/write `.vstpreset` files |

### For VST2
| Library | Stars | Language | Status | Use Case |
|---------|-------|----------|--------|----------|
| `demberto/fxp` | 10 | Python | Stable | Parse FXP headers/metadata |
| `CharlesHolbrow/vst2-preset-parser` | 10 | JavaScript | Stable | Parse FXP/FXB structure |
| `dafaronbi/Serum-FXP-Python-Reader` | 2 | Python | Minimal | Read Serum FXP files |

### For VST Hosting/Rendering
| Library | Stars | Language | Purpose |
|---------|-------|----------|---------|
| `spotify/pedalboard` | 7.5k+ | Python/C++ | Load/process VST files programmatically |
| `dmorgan/DawDreamer` | 500+ | Python | Render MIDI + VST chains to audio |
| `VENOM` | 80 | C++/Python | Build VST3 plugins using Python (pybind11) |

### For File Format Parsing
| Library | Language | Purpose |
|---------|----------|---------|
| `PyFLP` | Python | FL Studio project/preset parser (shows JSON pattern) |
| `plistlib` | Python stdlib | Parse AU `.aupreset` files (macOS) |

---

## 7. Practical Approaches for INSTRUMENTAL

### Approach A: VST3 JSON Export (Recommended)
**Goal**: Export 28 parameters as VST3 preset that any modern DAW can load.

**Steps**:
1. Map INSTRUMENTAL's 28 params → VST3 parameter IDs
2. Use `python-vstpreset` to create `.vstpreset` file
3. User opens in Ableton Live, Logic Pro, Studio One, Cubase, etc.

**Pros**: Universal, modern, well-supported
**Cons**: Requires mapping to a specific synth's VST plugin

### Approach B: JSON + Plugin-Agnostic Format
**Goal**: Store parameters in human-readable JSON, let users import.

**Steps**:
1. Export `instrumental_preset.json` with 28 parameters
2. Provide import utilities for popular synths:
   - Serum: Convert JSON → `.SerumPreset`
   - Vital: Convert JSON → `.vital`
   - Pigments: Convert JSON → Arturia format
   - Generic: Instructions for manual mapping

**Pros**: Plugin-agnostic, human-readable, flexible
**Cons**: Requires per-synth converters

### Approach C: Open-Source Synth Integration (Vital)
**Goal**: Generate Vital presets directly.

**Steps**:
1. Map 28 params → Vital's parameter set (~40 params)
2. Generate `.vital` JSON files
3. Users load in Vital (free, open-source)

**Pros**: Single synth, proven format, users can share/remix
**Cons**: Limited to Vital users

### Approach D: Multi-Format Export
**Goal**: Support multiple formats simultaneously.

**Steps**:
1. Core: Generic 28-param JSON format
2. Generate VST3, Vital, and Serum presets
3. Provide Python utilities for each format

**Pros**: Maximum compatibility
**Cons**: Most complex, requires reverse-engineering per synth

---

## 8. Parameter Mapping Challenge

INSTRUMENTAL recovers **28 parameters**:
- Oscillator mix (3 oscs)
- Filter cutoff, resonance
- ADSR (attack, decay, sustain, release)
- Reverb, delay, distortion
- Modulation depth/rate
- Plus others

**Real synths have 50-100+ parameters**. The challenge is mapping intelligently:

### Option 1: Direct 1:1 Mapping (If Synth Has ~28 Params)
- Best case: Vital or Pigments
- Not applicable: Serum (too many params)

### Option 2: Parameter Subset Mapping
- Map INSTRUMENTAL's 28 → Synth's core 28-30 params
- Leave advanced params at defaults
- Example: Serum → Use only main osc, filter, ADSR, 1 effect

### Option 3: Interpolation/Scaling
- If synth param has different range than INSTRUMENTAL param, scale/interpolate
- Example: Filter cutoff (0-1 in INSTRUMENTAL) → 20-20000 Hz in synth

### Option 4: Plugin-Specific Logic
- Each synth gets custom mapping rules
- Documented in code with comments

---

## 9. Reverse-Engineering Notes

### Serum v1 (.fxp)
- **Binary format**: Magic bytes + opaque state blob
- **Reverse-engineering difficulty**: High (proprietary)
- **Tools available**: `0xdevalias` gist with partial parameter mappings
- **Parameter order**: Known for Serum (shared in GitHub issues)

### Vital (.vital)
- **JSON format**: Fully transparent
- **Reverse-engineering difficulty**: Low (open-source)
- **Tools available**: Full format spec in vitaldb/vitalutils
- **Parameter names**: Human-readable, well-documented

### Pigments (Arturia)
- **Format**: Proprietary (`.aut` or embedded in project)
- **Reverse-engineering difficulty**: Medium
- **Documentation**: Limited public info
- **Alternative**: Use Arturia's C++ API (requires plugin development)

---

## 10. Recommended Next Steps

### Phase 1: Proof-of-Concept (Vital)
1. ✅ Document Vital's ~40 core parameters
2. ✅ Map INSTRUMENTAL's 28 params → Vital subset
3. ✅ Generate `.vital` JSON presets programmatically
4. ✅ Test in Vital (free download)
5. ✅ Validate audio quality

### Phase 2: VST3 Standard Export
1. Install `python-vstpreset` library
2. Choose target synth (Serum, Pigments, Vital as VST3)
3. Map parameters and create `.vstpreset` files
4. Test in major DAWs (Ableton, Logic, Cubase)

### Phase 3: Multi-Format Support
1. Create JSON schema for INSTRUMENTAL presets
2. Implement converters:
   - JSON → `.vstpreset`
   - JSON → `.vital`
   - JSON → Serum JSON (via `serum-preset-packager`)
3. CLI tool: `instrumental export --format=vital|vst3|serum`

### Phase 4: Web Audio → DAW Bridge (Future)
- Explore Spotify `pedalboard` library for offline rendering
- Consider DawDreamer for MIDI + VST chain rendering
- Not critical for preset export, but enables sound verification

---

## 11. Tools & Resources

### Essential Repositories
- **`janminor/python-vstpreset`**: VST3 preset R/W (Python)
- **`demberto/fxp`**: VST2 FXP parser (Python)
- **`0xdevalias/reverse-engineering-serum`**: Serum format docs (gist)
- **`KennethWussmann/serum-preset-packager`**: Serum JSON converter (Python)
- **`vitaldb/vitalutils`**: Vital format spec (C + PDF)

### Official Documentation
- **Steinberg VST3 Developer Portal**: https://steinbergmedia.github.io/vst3_dev_portal/
- **Apple Technical Note TN2157**: AU preset save/restore
- **Vital Documentation**: https://davidmvogel.com/docs/Vital/

### Community Resources
- **KVR Audio Forum**: Discussions on FXP/FXB format
- **Vital Forums**: Format discussions and preset sharing
- **0xdevalias Gist**: Comprehensive notes on synth preset generation

---

## 12. Conclusion

**Recommended Strategy for INSTRUMENTAL**:

1. **Primary**: VST3 `.vstpreset` format
   - Modern, standardized, cross-platform
   - Use `python-vstpreset` library
   - Requires mapping to specific synth (Vital recommended for POC)

2. **Secondary**: JSON export
   - Plugin-agnostic, human-readable
   - Allows users to import into their preferred synth
   - Enables community contributions for synth-specific mappers

3. **Prototype**: Vital `.vital` format
   - Fastest to implement (open-source, JSON-based)
   - Proves parameter mapping works
   - Low barrier to user testing (Vital is free)

4. **Future**: Multi-synth support
   - Serum, Pigments, Massive X as secondary targets
   - Expand ecosystem post-launch

**Success metric**: INSTRUMENTAL users can load recovered presets into their DAW and hear the reconstructed sound in real-time.
