"""
Export INSTRUMENTAL's 28 synth parameters to a Vital .vital preset file.

Uses the init.vital template from Syntheon as a base, then overwrites
the parameters we control and generates proper saw/square/sine wavetables.
"""

import json
import math
import os
import copy
import struct
import base64
import numpy as np

# Path to the Vital template preset (ships with Syntheon)
TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vendor", "syntheon", "syntheon", "inferencer", "vital", "init.vital"
)

# Load template once at import time
with open(TEMPLATE_PATH) as f:
    _TEMPLATE = json.load(f)


def hz_to_midi_note(hz):
    """Convert frequency in Hz to MIDI note number (continuous)."""
    if hz <= 0:
        return 0.0
    return 12.0 * math.log2(hz / 440.0) + 69.0


def _encode_wavetable(samples):
    """Encode a 2048-sample waveform as base64 float32 (Vital format)."""
    data = struct.pack(f'{len(samples)}f', *samples)
    return base64.b64encode(data).decode('ascii')


def _make_saw_wavetable():
    """Generate a sawtooth wavetable (2048 samples)."""
    t = np.linspace(0, 1, 2048, endpoint=False, dtype=np.float32)
    wave = 2.0 * t - 1.0  # -1 to +1 sawtooth
    return _encode_wavetable(wave)


def _make_square_wavetable(pulse_width=0.5):
    """Generate a square/pulse wavetable (2048 samples)."""
    t = np.linspace(0, 1, 2048, endpoint=False, dtype=np.float32)
    wave = np.where(t < pulse_width, 1.0, -1.0).astype(np.float32)
    return _encode_wavetable(wave)


def _make_sine_wavetable():
    """Generate a sine wavetable (2048 samples)."""
    t = np.linspace(0, 2 * np.pi, 2048, endpoint=False, dtype=np.float32)
    wave = np.sin(t).astype(np.float32)
    return _encode_wavetable(wave)


def _build_wavetable_entry(name, wave_data):
    """Build a single Vital wavetable entry with proper structure."""
    return {
        "author": "",
        "full_normalize": True,
        "name": name,
        "remove_all_dc": True,
        "version": "1.0.7",
        "groups": [
            {
                "components": [
                    {
                        "type": "Wave Source",
                        "interpolation": 1,
                        "interpolation_style": 1,
                        "keyframes": [
                            {"position": 0, "wave_data": wave_data}
                        ]
                    }
                ]
            }
        ]
    }


def params_to_vital(params_normalized, param_defs):
    """
    Convert INSTRUMENTAL's 28 normalized [0,1] params to a Vital preset dict.
    """
    # Denormalize params to physical ranges
    p = {}
    for i, pd in enumerate(param_defs):
        val = params_normalized[i]
        lo, hi = pd["lo"], pd["hi"]
        p[pd["name"]] = val * (hi - lo) + lo

    # Deep copy template
    preset = copy.deepcopy(_TEMPLATE)
    s = preset["settings"]

    # --- Oscillator levels ---
    s["osc_1_on"] = 1.0 if p["saw_mix"] > 0.01 else 0.0
    s["osc_1_level"] = p["saw_mix"]
    s["osc_2_on"] = 1.0 if p["square_mix"] > 0.01 else 0.0
    s["osc_2_level"] = p["square_mix"]
    s["osc_3_on"] = 1.0 if p["sine_mix"] > 0.01 else 0.0
    s["osc_3_level"] = p["sine_mix"]

    # --- Route oscillators ---
    if p["filter_cutoff"] > 10000:
        # Filter off — route direct out
        s["osc_1_destination"] = 2.0  # Direct Out
        s["osc_2_destination"] = 2.0
        s["osc_3_destination"] = 2.0
    else:
        # Route through Filter 1
        s["osc_1_destination"] = 0.0
        s["osc_2_destination"] = 0.0
        s["osc_3_destination"] = 0.0

    # --- Noise ---
    noise_total = p["noise_mix"] + p["noise_floor"]
    s["sample_on"] = 1.0 if noise_total > 0.01 else 0.0
    s["sample_level"] = min(1.0, noise_total * 2.0)

    # --- Detune ---
    s["osc_1_transpose"] = p["detune"]
    s["osc_2_transpose"] = p["detune"]
    s["osc_3_transpose"] = p["detune"]

    # --- Unison ---
    n_voices = max(1, round(p["unison_voices"]))
    for osc in ["osc_1", "osc_2", "osc_3"]:
        s[f"{osc}_unison_voices"] = float(n_voices)
        s[f"{osc}_unison_detune"] = p["unison_spread"] * 20.0

    # --- Pulse width (osc 2 wave frame) ---
    s["osc_2_wave_frame"] = p["pulse_width"]

    # --- Filter ---
    filter_midi = min(128.0, hz_to_midi_note(p["filter_cutoff"]))
    # If filter is wide open (cutoff > 10kHz), turn it off to avoid artifacts
    if p["filter_cutoff"] > 10000:
        s["filter_1_on"] = 0.0
    else:
        s["filter_1_on"] = 1.0
    s["filter_1_cutoff"] = filter_midi
    s["filter_1_resonance"] = p["filter_resonance"]
    s["filter_1_mix"] = 1.0

    slope = p["filter_slope"]
    if slope < 18:
        s["filter_1_model"] = 0.0
    elif slope < 36:
        s["filter_1_model"] = 1.0
    else:
        s["filter_1_model"] = 6.0

    # --- Amplitude envelope (env 1) — sustain fixed at 1.0 for full hold ---
    s["env_1_attack"] = p["attack"]
    s["env_1_decay"] = p["decay"]
    s["env_1_sustain"] = 1.0
    s["env_1_release"] = p["release"]

    # --- Filter envelope (env 2) — sustain fixed at 1.0 ---
    s["env_2_attack"] = p["filter_attack"]
    s["env_2_decay"] = p["filter_decay"]
    s["env_2_sustain"] = 1.0
    s["env_2_release"] = p["filter_release"]

    # --- Filter envelope amount ---
    filter_env_amount = p["filter_env"]
    if abs(filter_env_amount) > 0.01:
        mod_entry = {
            "source": "env_2",
            "destination": "filter_1_cutoff",
            "amount": filter_env_amount * 0.5
        }
        if "modulations" not in preset:
            preset["modulations"] = []
        preset["modulations"].append(mod_entry)

    # --- Master gain ---
    # Vital's volume is NOT 0-1. Keep template default (~5473).
    # Our gain is already expressed through osc levels.
    s["volume"] = 5473.0

    # --- Reverb ---
    s["reverb_on"] = 1.0 if p["reverb_mix"] > 0.01 else 0.0
    s["reverb_dry_wet"] = p["reverb_mix"]
    s["reverb_size"] = p["reverb_size"]

    # --- EQ ---
    s["eq_on"] = 1.0
    s["eq_low_cutoff"] = hz_to_midi_note(p["eq1_freq"])
    s["eq_low_gain"] = p["eq1_gain"]
    s["eq_high_cutoff"] = hz_to_midi_note(p["eq2_freq"])
    s["eq_high_gain"] = p["eq2_gain"]

    # --- Generate proper wavetables (saw, square, sine) ---
    s["wavetables"] = [
        _build_wavetable_entry("Saw", _make_saw_wavetable()),
        _build_wavetable_entry("Square", _make_square_wavetable(p.get("pulse_width", 0.5))),
        _build_wavetable_entry("Sine", _make_sine_wavetable()),
    ]

    # --- Metadata ---
    preset["preset_name"] = "INSTRUMENTAL Match"
    preset["author"] = "INSTRUMENTAL"
    preset["comments"] = "Auto-generated by INSTRUMENTAL (inverse synthesis)"

    return preset


def write_vital_preset(params_normalized, param_defs, output_path):
    """Generate and write a .vital preset file."""
    preset = params_to_vital(params_normalized, param_defs)
    with open(output_path, "w") as f:
        json.dump(preset, f, indent=2)
    return output_path
