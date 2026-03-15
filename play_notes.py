"""Play the 22 detected notes as simple sine tones through the laptop speaker."""

import subprocess
import tempfile
import numpy as np
import soundfile as sf

SR = 44100
NOTE_DUR = 0.148  # 148ms per note (median from detection)

# The 22 notes from detection
notes_hz = [
    294.5, 278.0, 220.6, 294.5, 278.0, 278.0, 220.6,
    292.8, 278.0, 292.8, 278.0, 292.8, 278.0, 220.6,
    294.5, 278.0, 220.6, 292.8, 278.0, 220.6, 294.5, 292.8,
]

# Render all notes into one waveform
audio = np.zeros(0)
for freq in notes_hz:
    t = np.linspace(0, NOTE_DUR, int(SR * NOTE_DUR), endpoint=False)
    # Simple saw + sine blend with a quick fade in/out to avoid clicks
    saw = 2.0 * (t * freq % 1.0) - 1.0
    sine = np.sin(2 * np.pi * freq * t)
    tone = 0.6 * saw + 0.4 * sine
    # Quick 5ms fade in/out
    fade = int(0.005 * SR)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    tone *= 0.3  # volume
    audio = np.concatenate([audio, tone])

# Save and play
outpath = "/Users/MOPOLLIKA/coding/hackathons/instrumental/output/synth_playback.wav"
sf.write(outpath, audio.astype(np.float32), SR)
print(f"Saved {len(notes_hz)} notes to {outpath}")
print(f"Duration: {len(audio)/SR:.2f}s")
print("Playing...")
subprocess.run(["afplay", outpath])
print("Done.")
