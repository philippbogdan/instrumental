# OpenDAW Integration Notes

## What is OpenDAW
- Browser-based DAW (TypeScript, Web Audio API, AudioWorklets)
- Open source (AGPL v3), ~1300 GitHub stars
- No VST/AU host — browser can't run native plugins
- SDK: `npm install @opendaw/studio-sdk` (pre-release)
- GitHub: github.com/andremichelle/openDAW

## Integration Paths

### Path 1: Vital Preset File (works today, any DAW)
Our tool outputs `.vital` files (JSON, 776 params). Users:
1. Download the `.vital` file
2. Drop into `~/Documents/Vital/User/Presets/`
3. Open Vital in any DAW → preset appears

### Path 2: Audio Sample Import (works today, OpenDAW)
1. Our tool renders audio from the matched params (via vita)
2. User imports the .wav into OpenDAW as a sample
3. Not ideal — loses playability, just a static audio clip

### Path 3: DawDreamer Bridge
1. Use DawDreamer (Python) to load Vital as VST headlessly
2. Set params programmatically
3. Render MIDI → audio
4. Export for OpenDAW or any DAW

### Path 4: Custom OpenDAW Box (future)
- OpenDAW has `@opendaw/studio-boxes` — extensible instrument layer
- Build a web-native synth that reads our param format
- Would need WebAssembly AudioWorklet implementation
- Or contribute to OpenDAW's Web Audio Modules 2.0 (WAM) support

## .vital File Format
- Plain JSON
- Keys: author, comments, preset_name, preset_style, settings (776 params), synth_version
- Settings include: osc params, filter params, envelope ADSR, LFO, mod matrix, effects
- Can embed base64-encoded wavetable data

## Vital Preset Folder Locations
- macOS: `~/Documents/Vital/User/Presets/`
- Windows: `C:\Users\<user>\Documents\Vital\User\Presets\`
- Linux: `~/.local/share/vital/User/Presets/`
