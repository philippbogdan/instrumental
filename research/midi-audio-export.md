# MIDI + Audio Rendering Export Research

**Research Date:** 2026-03-20
**Objective:** Investigate approaches for exporting INSTRUMENTAL synth matches as DAW-compatible instruments using MIDI + rendered audio.

---

## Executive Summary

Four viable export strategies exist, ranging from simple to sophisticated:

1. **MIDI + WAV (Simplest)** – Export MIDI notes + pre-rendered WAV of the synth. Works immediately in any DAW.
2. **SFZ Format (Best for Open)** – Text-based sampled instrument format with strong DAW support. Excellent Python library ecosystem.
3. **SF2 SoundFont (Universal)** – Long-established binary format. Works in virtually all DAWs. Libraries exist but less mature.
4. **Decent Sampler .dspreset (Modern, Free)** – XML-based, open format gaining adoption. Growing ecosystem, good for experimental workflows.

**Recommendation:** Start with **SFZ export** (multi-sampling with SFZ) as the primary strategy, with **MIDI + WAV** as a quick fallback. Both leverage existing mature Python tooling.

---

## Q1: MIDI + WAV Export (Simple Approach)

### Overview
Export a standard MIDI file (containing note timings) alongside a pre-rendered WAV file of the matched synth sound.

### Mechanics
- **MIDI file** = note pitch, timing, velocity
- **WAV file** = rendered audio of the synth (e.g., 30 seconds of the original signal or a loop)
- **DAW workflow** = Musician imports MIDI, then imports WAV as audio track

### Pros & Cons
| Pros | Cons |
|------|------|
| Requires no special format knowledge | Not a "reusable instrument" - just audio + notes |
| Works in every DAW immediately | No timbre variation across pitches (uses single recording) |
| Easy to implement (2 files) | Musician must manually sync notes to audio |
| Musician keeps original performances | No per-note control over timbre/expression |

### Python Libraries for MIDI Export
- **[MIDIUtil](https://pypi.org/project/MIDIUtil/)** – Pure Python, multi-track MIDI generation. Universal DAW compatibility. Mature & stable.
- **[Mido](https://mido.readthedocs.io/)** – Modern MIDI library. Can read/write MIDI files and interact with MIDI ports.

### Verdict
**Viable as fallback**, but not the primary recommendation. Good for musicians who want to "re-synthesize" the sound using their own synth patches with the original timing.

---

## Q2: SFZ Format (Text-Based Sampled Instrument)

### Overview
**SFZ** = "SoundFont Z" – a text-based, open standard for defining sampled instruments. Supported by most DAWs and free sample players.

### File Structure
```
// Example SFZ header
<control>
default_path=samples/

<group>
key=60  // Middle C (MIDI note 60)
sample=C4_velocity100.wav
ampeg_attack=0.01
ampeg_release=0.3

<group>
key=61  // C# (MIDI note 61)
sample=Cs4_velocity100.wav
ampeg_attack=0.01
ampeg_release=0.3
```

### Features
- **Text-based** – Human-readable, version-controllable
- **Relative paths** – References audio files alongside the .sfz file
- **Flexible mapping** – Map samples to key ranges, velocity layers, round-robin alternation
- **Effects/modulation** – Basic AHDSR envelope, filters, pitch bend, modulation wheel

### DAW Support
| DAW | Support | Notes |
|-----|---------|-------|
| **Ableton Live** | ✅ Sampler | Full SFZ support via Sampler plugin |
| **FL Studio** | ✅ (via free players) | SFZ Player works well |
| **Logic Pro** | ⚠️ Limited | Best via 3rd-party (e.g., Sforzando) |
| **Reaper** | ✅ ReaSampler | Full support |
| **Studio One** | ✅ Sampler | Good SFZ support |
| **Kontakt** | ❌ (proprietary) | Native format is .nki |

### Python Libraries for SFZ Creation

#### 1. **[pysfz](https://github.com/ajjackson/pysfz)** ⭐
- Purpose: Python interface for SFZ files
- Status: Early-stage (3 commits), but clean architecture
- Planned features: Create SFZ files from scratch
- **Caveat**: Not yet complete for generation; mainly import/modify

#### 2. **[jSfzTools](https://github.com/jlearman/jSfzTools)** ⭐⭐⭐
- **jCutSamps.py** – Segments multi-note recordings into individual samples
  - Auto-detects pitch via auto-correlation
  - Names files with MIDI note numbers
- **jMap.py** – Maps samples to keyboard ranges, generates SFZ file
- **Best for**: Workflows starting from recorded acoustic instruments (piano, bass)
- Works well with 16-bit, 44.1kHz/48kHz WAV files

#### 3. **[sf-creator](https://github.com/paulwellnerbou/sf-creator)** ⭐⭐
- Command: `python main.py sfz <sample-directory>`
- Auto-detects note names from filenames (A0 to C8)
- Fills gaps automatically across keyboard range
- **Supports both SFZ and DecentSampler output** (dual-format)
- Simpler than jSfzTools; good for pre-named samples

#### 4. **[SFZGen](https://github.com/ozikazina/SFZGen)** ⭐
- Sforzando file generator
- GUI or CLI option
- May require less Python integration

#### 5. **[sf2-to-sfz](https://github.com/bashexploe/sf2-to-sfz)** (Conversion)
- Converts SF2 SoundFont → SFZ format
- Useful if legacy SF2 exists

### Recommended SFZ Workflow for INSTRUMENTAL
1. **Generate samples** – Render synth at each MIDI note (C0 to C8, 2-3 velocity layers)
2. **Name files** – `C4_vel100.wav`, `Cs4_vel100.wav`, etc.
3. **Use sf-creator** – `python main.py sfz ./samples/` → generates `soundfont.sfz`
4. **Test in DAW** – Load in Ableton Sampler, Reaper, or free Sforzando

### Verdict
**Best choice for maximum compatibility & flexibility.** SFZ is open, text-based, and enjoys strong DAW support. Python ecosystem is maturing. One-time setup for sample generation, then musician gets full control.

---

## Q3: SF2 SoundFont (Binary Standard)

### Overview
**SF2** (SoundFont 2) – Binary format dating to the 1990s. Universal DAW support but more rigid than SFZ.

### Features
- **Binary format** – Not human-editable
- **Well-established** – Long industry standard
- **DAW support** – Every major DAW supports SF2 import

### Python Libraries
- **[sf-creator](https://github.com/paulwellnerbou/sf-creator)** – Generates both SFZ and SF2 from samples (primary tool)
- **[pysfz](https://github.com/ajjackson/pysfz)** – Limited SF2 support; mostly SFZ-focused

### Pros & Cons
| Pros | Cons |
|------|------|
| Universal DAW support | Binary format – can't inspect/edit easily |
| Extremely stable format | Python tooling less mature than SFZ |
| No licensing issues | Larger file size than SFZ |
| Long track record | Less flexible for advanced workflows |

### Verdict
**Viable secondary option**, but SFZ is preferred. SF2 is more limiting and requires binary tools to inspect. Worth supporting if users specifically request it.

---

## Q4: Decent Sampler .dspreset (Modern, Open)

### Overview
**DecentSampler** (.dspreset) – Modern, free plugin with open XML-based preset format. Growing adoption among sample librarians.

### File Structure
```xml
<?xml version="1.0" encoding="UTF-8"?>
<DecentSampler minVersion="1.0.0">
  <ui>
    <label text="My Synth Sample"/>
  </ui>

  <groups>
    <group>
      <sample path="C4_vel100.wav" loNote="60" hiNote="60" loVel="100" hiVel="100"/>
      <sample path="Cs4_vel100.wav" loNote="61" hiNote="61" loVel="100" hiVel="100"/>
    </group>
  </groups>
</DecentSampler>
```

### Advantages
- **Open format** – XML, human-readable, version-controllable
- **Free plugin** – DecentSampler is free (indie-friendly)
- **Growing ecosystem** – Gaining traction for custom sample libraries
- **Community tools** – [DecentSampler schema](https://github.com/praashie/DecentSampler-schema) & [sample projects](https://github.com/J0rgeSerran0/Decent-Sampler-Samples)
- **Documentation** – [Official guide](https://decentsampler-developers-guide.readthedocs.io/)

### DAW Integration
- **Standalone plugin** – Works as VST, AU, AAX in any DAW
- **Not sampler-dependent** – Unlike EXS24 or Kontakt, DecentSampler is independent software

### Limitations
- **Smaller installed base** – Not everyone has it (but it's free, easy to install)
- **Less mature** – Younger than SF2/SFZ
- **Plugin requirement** – Users must install DecentSampler plugin (vs. SFZ which works in native samplers)

### Verdict
**Excellent for open-source workflows & experimental projects.** Pair with SFZ for maximum reach. DecentSampler is ideal if you want to provide a complete, packaged solution (plugin + samples + preset).

---

## Q5: DAW Sampler Support & Multi-Sampling

### Native Samplers Across DAWs

| DAW | Sampler | Format Support | Multisampling | Notes |
|-----|---------|-----------------|-------------------|-------|
| **Ableton** | Sampler | SFZ, WAV, AIFF | ✅ Yes | Excellent SFZ support; easy sample mapping |
| **FL Studio** | Sampler | SFZ (via plugins) | ✅ Yes | Native sampler + free 3rd-party SFZ players |
| **Logic Pro** | EXS24 | EXS (native), AIFF | ✅ Yes | Proprietary; harder to generate programmatically |
| **Reaper** | ReaSampler | SFZ, WAV, AIFF | ✅ Yes | Full SFZ support; scriptable |
| **Studio One** | Sampler | SFZ, WAV | ✅ Yes | Good multisampling workflow |
| **Kontakt** | Kontakt | .nki (proprietary) | ✅ Yes | Can import EXS/SFZ but converts to .nki |

### Multi-Sampling Best Practices
1. **Sample density** – 1 sample per note for monophonic sounds is minimum; 2-3 velocity layers for better expressivity
2. **Sample quality** – 24-bit, 44.1kHz or 48kHz WAV files are standard
3. **Looping** – For sustained synth sounds, add loop points (attack + sustain loop)
4. **Automation** – AHDSR envelope (Attack, Decay, Sustain, Release) maps to sampler built-ins

### Kontakt & EXS24 Specifics
- **EXS24 format** – Apple Logic proprietary; uses .exs patch files + AIFF samples
- **Kontakt format** – Native Instruments proprietary; uses .nki files
- **Conversion barriers** – EXS → Kontakt requires manual conversion (no standard tool)
- **Workaround** – Export SFZ, load in Kontakt → Kontakt will convert to .nki

### Verdict
**SFZ is most portable.** Ableton Sampler, Reaper, and Studio One all support it natively. For Kontakt users, SFZ acts as a bridge format (Kontakt can read & convert).

---

## Q6: Recommended Export Strategy (Priority-Ordered)

### 🥇 Primary: SFZ Multi-Sampling
**When to use:** For maximum DAW compatibility, flexibility, and future-proofing.

**Workflow:**
1. Render synth samples at each MIDI note (C0–C8) with 2-3 velocity layers
2. Save as `C4_vel100.wav`, `Cs4_vel100.wav`, etc.
3. Use `sf-creator` to auto-generate `soundfont.sfz`
4. Package: `.sfz` file + `samples/` folder
5. User imports into DAW's sampler

**Pros:**
- Works in Ableton, Reaper, Studio One, FL Studio (with plugins), Logic (with 3rd-party)
- Text-based – can inspect/edit sample mappings
- Minimal file size
- No external plugin required (most DAWs have samplers)

**Cons:**
- Requires pre-rendered samples (computational cost upfront)

---

### 🥈 Secondary: Decent Sampler .dspreset
**When to use:** As a free, packaged alternative with built-in plugin.

**Workflow:**
1. Render same samples as SFZ
2. Generate `.dspreset` XML file with sample mappings
3. Package: `.dspreset` + `samples/` folder
4. User downloads DecentSampler plugin (free) and loads preset

**Pros:**
- Complete solution (plugin included)
- Modern XML format
- Open-source ecosystem
- Works across all DAWs as plugin

**Cons:**
- Requires plugin installation
- Smaller user base than SFZ

---

### 🥉 Tertiary: MIDI + WAV (Quick Fallback)
**When to use:** For users who prefer to re-synthesize with their own synth, or for preview/demo purposes.

**Workflow:**
1. Generate MIDI file of the original performance
2. Render full WAV of the synth output
3. Package: `.mid` + `.wav`
4. User imports both into DAW

**Pros:**
- No special format knowledge needed
- Works in every DAW immediately
- Musician retains creative control (can resynthesized with different patch)

**Cons:**
- Not a reusable instrument
- No timbre variation across pitches
- Musician must manually sync notes to audio

---

### ❌ Not Recommended: SF2 SoundFont
**Why:** SFZ is superior (text-based, cleaner tooling). SF2 is legacy. Unless users specifically request SF2 support, prioritize SFZ.

---

## Implementation Roadmap

### Phase 1: Proof of Concept (SFZ)
1. Render test samples (C4, D4, E4 at 1 velocity) from current synth
2. Write simple Python script using `sf-creator` to generate `.sfz`
3. Test in Ableton Sampler

### Phase 2: Full SFZ Export
1. Extend to full keyboard range (C0–C8)
2. Add 2-3 velocity layers per note
3. Optimize sample quality & looping
4. Generate `.sfz` dynamically from matched synth parameters

### Phase 3: Decent Sampler (Optional)
1. Generate `.dspreset` XML alongside SFZ
2. Test in DecentSampler plugin
3. Provide both formats in download

### Phase 4: Ableton .adg Support (Advanced)
1. Research Ableton Sampler's `.adg` preset format (proprietary)
2. If feasible, auto-generate Ableton-specific sampler presets

---

## Python Libraries Checklist

| Library | Purpose | Status | Recommendation |
|---------|---------|--------|-----------------|
| **MIDIUtil** | MIDI file generation | ✅ Stable | Use for MIDI export |
| **Mido** | MIDI read/write | ✅ Active | Use for MIDI export |
| **sf-creator** | SFZ/DecentSampler generation | ✅ Functional | ⭐ Primary choice |
| **jSfzTools** | Advanced SFZ mapping from multi-note recordings | ✅ Functional | Alternative if acoustic source |
| **pysfz** | SFZ Python interface | ⚠️ Early-stage | Monitor for maturity |
| **SFZGen** | SFZ GUI/CLI generator | ✅ Functional | Alternative UI |

---

## References

- [DecentSampler Documentation](https://decentsampler-developers-guide.readthedocs.io/en/latest/introduction.html)
- [DecentSampler Schema (GitHub)](https://github.com/praashie/DecentSampler-schema)
- [pysfz (GitHub)](https://github.com/ajjackson/pysfz)
- [jSfzTools (GitHub)](https://github.com/jlearman/jSfzTools)
- [sf-creator (GitHub)](https://github.com/paulwellnerbou/sf-creator)
- [MIDIUtil (PyPI)](https://pypi.org/project/MIDIUtil/)
- [Mido Documentation](https://mido.readthedocs.io/en/latest/)
- [SoundOnSound: Multisampling with EXS24](https://www.soundonsound.com/techniques/multisampling-exs24)
- [Ableton Sampler Multisampling Guide](https://help.ableton.com/hc/en-us/articles/115001318670-How-To-Multisampling-with-Sampler)

---

## Conclusion

**INSTRUMENTAL should prioritize SFZ export** with MIDI + WAV as a fallback option. SFZ offers:
- ✅ Text-based format (version control, inspection)
- ✅ Wide DAW support (Ableton, Reaper, Studio One, others)
- ✅ Mature Python ecosystem (sf-creator is production-ready)
- ✅ Flexibility (multisampling, velocity layers, custom envelopes)
- ✅ Open, patent-free standard

Optionally provide **DecentSampler .dspreset** as a free, packaged alternative with no external dependencies.

**Secondary formats (SF2, EXS24, Kontakt) are less important.** Focus engineering effort on the SFZ pipeline first.
