"""
Test all Tier 1 + Tier 2 hypotheses in order. Each = modify synth/loss inline, 10K evals, play.
"""

import torch, numpy as np, soundfile as sf, subprocess, time, math, cma, librosa
import torchaudio

# ─── Load targets ───
TARGET_NOTES = []
for f, freq in [('notes/note_03_A3_221Hz.wav', 221.0), ('notes/note_02_Cs4_278Hz.wav', 278.0), ('notes/note_01_D4_295Hz.wav', 295.0)]:
    audio, sr = sf.read(f)
    TARGET_NOTES.append((torch.tensor(audio, dtype=torch.float32), freq, len(audio)/44100))

from src.synth import SynthPatch, PARAM_DEFS
from src.losses import get_loss

# v24 best params as baseline
V24 = [0.8996, 1.0, 0.0042, 0.0004, 0.4997, 0.9955, 0.0017, 0.0001,
       0.3991, 0.5441, 0.1885, 0.1455, 0.6262, 0.9598, 0.0306,
       0.0428, 0.0053, 0.2459, 0.2989, 0.5355, 0.2216, 0.7211, 0.4975, 0.4897]

orig_full, _ = librosa.load('target.wav', sr=44100, mono=True)
NOTE_DATA = [
    (0.023,0.148,294.5),(0.171,0.154,278.0),(0.325,0.145,220.6),
    (0.470,0.151,294.5),(0.621,0.145,278.0),(0.778,0.136,220.6),
    (0.914,0.154,292.8),(1.068,0.148,278.0),(1.216,0.148,292.8),
    (1.364,0.145,278.0),(1.509,0.145,292.8),(1.654,0.148,278.0),
    (1.802,0.157,220.6),(1.959,0.145,294.5),(2.104,0.148,278.0),
    (2.252,0.151,220.6),(2.403,0.145,292.8),(2.548,0.151,278.0),
    (2.699,0.145,220.6),(2.844,0.151,294.5),(2.995,0.470,292.8),
]
s, e = int(1.312*44100), int(4.5*44100)
rms_orig = np.sqrt(np.mean(orig_full[s:e]**2))


def render_sequence(synth, params, note_data=NOTE_DATA):
    """Render full note sequence."""
    full = np.zeros(len(orig_full))
    for onset, dur, freq in note_data:
        audio = synth.render(params, f0_hz=freq, duration=dur+0.3, note_duration=dur)
        audio_np = audio.detach().numpy().squeeze()
        pos = int((1.312+onset)*44100)
        end = pos+len(audio_np)
        if end <= len(full): full[pos:end] += audio_np
        else:
            trim = len(full)-pos
            if trim > 0: full[pos:pos+trim] += audio_np[:trim]
    rms_m = np.sqrt(np.mean(full[s:e]**2))
    full *= rms_orig/(rms_m+1e-8)
    if np.max(np.abs(full)) > 0.99: full = full / np.max(np.abs(full)) * 0.99
    return full.astype(np.float32)


def run_cmaes(eval_fn, n_params, x0, budget=10000, popsize=20, sigma=0.15):
    """Quick CMA-ES run."""
    es = cma.CMAEvolutionStrategy(x0, sigma, {
        'bounds': [[0]*n_params, [1]*n_params],
        'maxfevals': budget, 'popsize': popsize, 'verbose': -9,
    })
    best_loss = float('inf')
    while not es.stop():
        sols = es.ask()
        fits = [eval_fn(s) for s in sols]
        es.tell(sols, fits)
        if min(fits) < best_loss: best_loss = min(fits)
    return es.result.xbest, best_loss


def standard_eval(params_np, synth=None, loss_fn=None):
    """Standard evaluation: render 3 notes, compute matching loss."""
    if synth is None: synth = SynthPatch()
    if loss_fn is None: loss_fn = get_loss("matching")
    params = torch.tensor(params_np, dtype=torch.float32)
    total = 0.0
    for tgt, freq, dur in TARGET_NOTES:
        gen = synth.render(params, f0_hz=freq, duration=dur, note_duration=dur*0.9)
        target = tgt.unsqueeze(0)
        ml = min(target.shape[1], gen.shape[1])
        loss = loss_fn(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
        total += loss.item()
    return total / len(TARGET_NOTES)


# ═══════════════════════════════════════════════════════════
# TEST A: Chebyshev Waveshaping
# ═══════════════════════════════════════════════════════════
def test_A():
    """Add Chebyshev waveshaping: sine → polynomial → controlled harmonics."""
    print("\n" + "="*60, flush=True)
    print("TEST A: Chebyshev Waveshaping (5 extra params)", flush=True)
    print("="*60, flush=True)

    class ChebyshevSynth(SynthPatch):
        """Extends SynthPatch with Chebyshev waveshaping on the sine oscillator."""
        def __init__(self):
            super().__init__()
            self._n_params = len(PARAM_DEFS) + 5  # a1-a5 coefficients

        def get_param_count(self):
            return self._n_params

        def render(self, params_tensor, f0_hz=440.0, duration=1.0, note_duration=None):
            base_params = params_tensor[:len(PARAM_DEFS)]
            cheb_coeffs = params_tensor[len(PARAM_DEFS):] * 2.0  # scale [0,1] → [0,2]

            # Get base audio
            audio = super().render(base_params, f0_hz, duration, note_duration)
            signal = audio.squeeze()

            # Apply Chebyshev waveshaping to add harmonic content
            # T1(x)=x, T2(x)=2x²-1, T3(x)=4x³-3x, T4(x)=8x⁴-8x²+1, T5(x)=16x⁵-20x³+5x
            x = signal.clamp(-1, 1)
            shaped = (cheb_coeffs[0] * x +
                      cheb_coeffs[1] * (2*x**2 - 1) +
                      cheb_coeffs[2] * (4*x**3 - 3*x) +
                      cheb_coeffs[3] * (8*x**4 - 8*x**2 + 1) +
                      cheb_coeffs[4] * (16*x**5 - 20*x**3 + 5*x))

            # Mix original + shaped
            result = 0.5 * signal + 0.5 * shaped
            return result.unsqueeze(0)

    synth = ChebyshevSynth()
    x0 = V24 + [0.33, 0.33, 0.33, 0.33, 0.33]  # neutral coefficients

    def eval_fn(p):
        return standard_eval(p, synth=synth)

    t0 = time.time()
    best_p, best_loss = run_cmaes(eval_fn, len(x0), x0, budget=10000)
    print(f"  Loss: {best_loss:.4f} (baseline v24: ~2.13) in {time.time()-t0:.0f}s", flush=True)
    print(f"  Cheb coeffs: {[f'{best_p[i]:.3f}' for i in range(24, 29)]}", flush=True)

    audio = render_sequence(synth, torch.tensor(best_p, dtype=torch.float32))
    sf.write('output/test_A_chebyshev.wav', audio, 44100)
    return best_loss


# ═══════════════════════════════════════════════════════════
# TEST B: Parametric EQ (2 peak bands)
# ═══════════════════════════════════════════════════════════
def test_B():
    """Add 2-band parametric EQ that can BOOST frequencies."""
    print("\n" + "="*60, flush=True)
    print("TEST B: Parametric EQ — 2 peak bands (4 extra params)", flush=True)
    print("="*60, flush=True)

    class EQSynth(SynthPatch):
        def __init__(self):
            super().__init__()
            self._n_params = len(PARAM_DEFS) + 4

        def get_param_count(self):
            return self._n_params

        def render(self, params_tensor, f0_hz=440.0, duration=1.0, note_duration=None):
            base_params = params_tensor[:len(PARAM_DEFS)]
            eq_params = params_tensor[len(PARAM_DEFS):]

            audio = super().render(base_params, f0_hz, duration, note_duration)
            signal = audio.squeeze()
            N = signal.shape[-1]

            # 2 peak bands: freq1, gain1, freq2, gain2
            freq1 = eq_params[0] * 4000 + 500   # 500-4500 Hz
            gain1 = eq_params[1] * 12 - 6       # -6 to +6 dB
            freq2 = eq_params[2] * 8000 + 2000  # 2000-10000 Hz
            gain2 = eq_params[3] * 12 - 6       # -6 to +6 dB

            freqs = torch.fft.rfftfreq(N, d=1.0/44100)
            X = torch.fft.rfft(signal)

            # Apply peak EQ (Gaussian bell in frequency domain)
            for freq, gain in [(freq1, gain1), (freq2, gain2)]:
                Q = 2.0
                bell = gain / 6.0 * torch.exp(-0.5 * ((freqs - freq) / (freq / Q)) ** 2)
                X = X * (1.0 + bell)

            signal = torch.fft.irfft(X, n=N)
            return signal.unsqueeze(0)

    synth = EQSynth()
    x0 = V24 + [0.5, 0.5, 0.5, 0.5]  # center freqs, neutral gain

    def eval_fn(p):
        return standard_eval(p, synth=synth)

    t0 = time.time()
    best_p, best_loss = run_cmaes(eval_fn, len(x0), x0, budget=10000)
    print(f"  Loss: {best_loss:.4f} in {time.time()-t0:.0f}s", flush=True)
    eq = best_p[24:]
    print(f"  Band1: {eq[0]*4000+500:.0f}Hz @ {eq[1]*12-6:+.1f}dB", flush=True)
    print(f"  Band2: {eq[2]*8000+2000:.0f}Hz @ {eq[3]*12-6:+.1f}dB", flush=True)

    audio = render_sequence(synth, torch.tensor(best_p, dtype=torch.float32))
    sf.write('output/test_B_eq.wav', audio, 44100)
    return best_loss


# ═══════════════════════════════════════════════════════════
# TEST C: TFS/Envelope Loss
# ═══════════════════════════════════════════════════════════
def test_C():
    """Add temporal fine structure loss via Hilbert transform envelope."""
    print("\n" + "="*60, flush=True)
    print("TEST C: TFS/Envelope Loss (same synth, different loss)", flush=True)
    print("="*60, flush=True)

    base_loss = get_loss("matching")

    def tfs_loss(gen, tgt):
        """Matching loss + envelope trajectory loss."""
        base = base_loss(gen, tgt)
        # Envelope via absolute value smoothing (cheaper than Hilbert)
        g = gen.squeeze()
        t = tgt.squeeze()
        ml = min(len(g), len(t))
        g, t = g[:ml], t[:ml]
        # RMS envelope in short windows
        win = 256
        g_env = g.unfold(0, win, win//2).pow(2).mean(dim=1).sqrt()
        t_env = t.unfold(0, win, win//2).pow(2).mean(dim=1).sqrt()
        ml2 = min(len(g_env), len(t_env))
        env_loss = torch.mean(torch.abs(g_env[:ml2] - t_env[:ml2]))
        return base + 2.0 * env_loss

    synth = SynthPatch()
    x0 = V24[:]

    def eval_fn(p):
        params = torch.tensor(p, dtype=torch.float32)
        total = 0.0
        for tgt, freq, dur in TARGET_NOTES:
            gen = synth.render(params, f0_hz=freq, duration=dur, note_duration=dur*0.9)
            target = tgt.unsqueeze(0)
            ml = min(target.shape[1], gen.shape[1])
            loss = tfs_loss(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
            total += loss.item()
        return total / len(TARGET_NOTES)

    t0 = time.time()
    best_p, best_loss = run_cmaes(eval_fn, len(x0), x0, budget=10000)
    # Compute standard loss for comparison
    std_loss = standard_eval(best_p)
    print(f"  TFS Loss: {best_loss:.4f}, Standard Loss: {std_loss:.4f} in {time.time()-t0:.0f}s", flush=True)

    audio = render_sequence(synth, torch.tensor(best_p, dtype=torch.float32))
    sf.write('output/test_C_tfs.wav', audio, 44100)
    return std_loss


# ═══════════════════════════════════════════════════════════
# TEST D: CDPAM Loss
# ═══════════════════════════════════════════════════════════
def test_D():
    """Use CDPAM perceptual metric as loss."""
    print("\n" + "="*60, flush=True)
    print("TEST D: CDPAM Perceptual Loss", flush=True)
    print("="*60, flush=True)

    try:
        import cdpam
        cdpam_loss = cdpam.CDPAM()

        synth = SynthPatch()
        x0 = V24[:]

        def eval_fn(p):
            params = torch.tensor(p, dtype=torch.float32)
            total = 0.0
            for tgt, freq, dur in TARGET_NOTES:
                gen = synth.render(params, f0_hz=freq, duration=dur, note_duration=dur*0.9)
                ml = min(tgt.shape[0], gen.shape[1])
                # CDPAM expects (batch, samples) at 22050 Hz
                g = gen.squeeze()[:ml].unsqueeze(0)
                t = tgt[:ml].unsqueeze(0)
                loss = cdpam_loss.forward(t, g)
                total += loss.item()
            return total / len(TARGET_NOTES)

        t0 = time.time()
        best_p, best_loss = run_cmaes(eval_fn, len(x0), x0, budget=10000)
        std_loss = standard_eval(best_p)
        print(f"  CDPAM Loss: {best_loss:.4f}, Standard Loss: {std_loss:.4f} in {time.time()-t0:.0f}s", flush=True)

        audio = render_sequence(synth, torch.tensor(best_p, dtype=torch.float32))
        sf.write('output/test_D_cdpam.wav', audio, 44100)
        return std_loss
    except ImportError:
        print("  CDPAM not installed. pip install cdpam. SKIPPING.", flush=True)
        return None


# ═══════════════════════════════════════════════════════════
# TEST E: SPSA Fine-tuning after CMA-ES
# ═══════════════════════════════════════════════════════════
def test_E():
    """CMA-ES 5K + SPSA 5K fine-tuning."""
    print("\n" + "="*60, flush=True)
    print("TEST E: CMA-ES 5K → SPSA 5K fine-tuning", flush=True)
    print("="*60, flush=True)

    synth = SynthPatch()
    x0 = V24[:]

    # Phase 1: CMA-ES 5K
    t0 = time.time()
    best_p, best_loss = run_cmaes(standard_eval, len(x0), x0, budget=5000)
    print(f"  CMA-ES phase: {best_loss:.4f}", flush=True)

    # Phase 2: SPSA 5K
    params = np.array(best_p)
    lr = 0.01
    for step in range(2500):  # 2 evals per step = 5000 evals
        c = 0.01 / (1 + step) ** 0.1
        delta = np.random.choice([-1, 1], size=len(params))
        loss_plus = standard_eval(np.clip(params + c * delta, 0, 1))
        loss_minus = standard_eval(np.clip(params - c * delta, 0, 1))
        grad_est = (loss_plus - loss_minus) / (2 * c) * delta
        a = lr / (1 + step) ** 0.602
        params = np.clip(params - a * grad_est, 0, 1)
        if step % 500 == 0:
            current = standard_eval(params)
            if current < best_loss:
                best_loss = current
                best_p = params.copy()
            print(f"    SPSA step {step}: loss={current:.4f}", flush=True)

    final_loss = standard_eval(best_p)
    print(f"  Final: {final_loss:.4f} in {time.time()-t0:.0f}s", flush=True)

    audio = render_sequence(synth, torch.tensor(best_p, dtype=torch.float32))
    sf.write('output/test_E_spsa.wav', audio, 44100)
    return final_loss


# ═══════════════════════════════════════════════════════════
# TEST F: CLAP Embedding Loss
# ═══════════════════════════════════════════════════════════
def test_F():
    """Use CLAP audio embeddings as loss."""
    print("\n" + "="*60, flush=True)
    print("TEST F: CLAP Embedding Loss", flush=True)
    print("="*60, flush=True)

    try:
        from transformers import ClapModel, ClapProcessor
        model = ClapModel.from_pretrained("laion/clap-htsat-fused")
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
        model.eval()

        # Pre-compute target embeddings
        target_embeds = []
        for tgt, freq, dur in TARGET_NOTES:
            inputs = processor(audios=[tgt.numpy()], return_tensors="pt", sampling_rate=44100)
            with torch.no_grad():
                emb = model.get_audio_features(**inputs)
            target_embeds.append(emb)

        synth = SynthPatch()
        x0 = V24[:]

        def eval_fn(p):
            params = torch.tensor(p, dtype=torch.float32)
            total = 0.0
            for i, (tgt, freq, dur) in enumerate(TARGET_NOTES):
                gen = synth.render(params, f0_hz=freq, duration=dur, note_duration=dur*0.9)
                gen_np = gen.squeeze().detach().numpy()
                inputs = processor(audios=[gen_np], return_tensors="pt", sampling_rate=44100)
                with torch.no_grad():
                    gen_emb = model.get_audio_features(**inputs)
                dist = 1 - torch.nn.functional.cosine_similarity(gen_emb, target_embeds[i])
                total += dist.item()
            return total / len(TARGET_NOTES)

        t0 = time.time()
        best_p, best_loss = run_cmaes(eval_fn, len(x0), x0, budget=5000, popsize=16)
        std_loss = standard_eval(best_p)
        print(f"  CLAP Loss: {best_loss:.4f}, Standard Loss: {std_loss:.4f} in {time.time()-t0:.0f}s", flush=True)

        audio = render_sequence(synth, torch.tensor(best_p, dtype=torch.float32))
        sf.write('output/test_F_clap.wav', audio, 44100)
        return std_loss
    except Exception as e:
        print(f"  CLAP failed: {e}. SKIPPING.", flush=True)
        return None


# ═══════════════════════════════════════════════════════════
# TEST G: Multi-start CMA-ES (8 runs)
# ═══════════════════════════════════════════════════════════
def test_G():
    """8 independent CMA-ES runs, keep best."""
    print("\n" + "="*60, flush=True)
    print("TEST G: Multi-start CMA-ES (8 runs × 1.25K evals)", flush=True)
    print("="*60, flush=True)

    best_global = float('inf')
    best_params = None
    t0 = time.time()

    for run in range(8):
        # Mix v24 with random perturbation for diversity
        x0 = [v + np.random.uniform(-0.2, 0.2) for v in V24]
        x0 = [max(0, min(1, v)) for v in x0]

        p, loss = run_cmaes(standard_eval, len(x0), x0, budget=1250, popsize=16, sigma=0.25)
        print(f"    Run {run+1}: loss={loss:.4f}", flush=True)
        if loss < best_global:
            best_global = loss
            best_params = p

    print(f"  Best: {best_global:.4f} in {time.time()-t0:.0f}s", flush=True)

    synth = SynthPatch()
    audio = render_sequence(synth, torch.tensor(best_params, dtype=torch.float32))
    sf.write('output/test_G_multistart.wav', audio, 44100)
    return best_global


# ═══════════════════════════════════════════════════════════
# TEST H: FM Synthesis (2 extra params)
# ═══════════════════════════════════════════════════════════
def test_H():
    """Add FM synthesis: carrier + modulator."""
    print("\n" + "="*60, flush=True)
    print("TEST H: FM Synthesis (2 extra params: ratio + index)", flush=True)
    print("="*60, flush=True)

    class FMSynth(SynthPatch):
        def __init__(self):
            super().__init__()
            self._n_params = len(PARAM_DEFS) + 2

        def get_param_count(self):
            return self._n_params

        def render(self, params_tensor, f0_hz=440.0, duration=1.0, note_duration=None):
            base_params = params_tensor[:len(PARAM_DEFS)]
            fm_ratio = params_tensor[len(PARAM_DEFS)] * 7 + 1     # ratio 1-8
            fm_index = params_tensor[len(PARAM_DEFS)+1] * 10      # index 0-10

            # Get base audio
            audio = super().render(base_params, f0_hz, duration, note_duration)
            signal = audio.squeeze()

            # Add FM component
            n = len(signal)
            t = torch.linspace(0, duration, n)
            fm_signal = torch.sin(2 * math.pi * f0_hz * t +
                                  fm_index * torch.sin(2 * math.pi * f0_hz * fm_ratio * t))

            # Mix: 60% original + 40% FM
            result = 0.6 * signal + 0.4 * fm_signal * signal.abs().max()
            return result.unsqueeze(0)

    synth = FMSynth()
    x0 = V24 + [0.3, 0.2]  # ratio~3, low index

    def eval_fn(p):
        return standard_eval(p, synth=synth)

    t0 = time.time()
    best_p, best_loss = run_cmaes(eval_fn, len(x0), x0, budget=10000)
    print(f"  Loss: {best_loss:.4f} in {time.time()-t0:.0f}s", flush=True)
    print(f"  FM ratio: {best_p[24]*7+1:.2f}, index: {best_p[25]*10:.2f}", flush=True)

    audio = render_sequence(synth, torch.tensor(best_p, dtype=torch.float32))
    sf.write('output/test_H_fm.wav', audio, 44100)
    return best_loss


# ═══════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = {}

    # Baseline
    baseline = standard_eval(V24)
    print(f"\nBASELINE (v24): {baseline:.4f}\n", flush=True)

    # Tier 1
    results['A_chebyshev'] = test_A()
    results['B_eq'] = test_B()
    results['C_tfs'] = test_C()
    results['D_cdpam'] = test_D()

    # Tier 2
    results['E_spsa'] = test_E()
    results['F_clap'] = test_F()
    results['G_multistart'] = test_G()
    results['H_fm'] = test_H()

    # Summary
    print("\n" + "="*60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"  Baseline v24:  {baseline:.4f}", flush=True)
    for name, loss in sorted(results.items(), key=lambda x: x[1] if x[1] else 999):
        if loss is not None:
            delta = ((loss - baseline) / baseline) * 100
            print(f"  {name:>15}: {loss:.4f} ({delta:+.1f}%)", flush=True)

    best_name = min(results, key=lambda k: results[k] if results[k] else 999)
    print(f"\nBest: {best_name} ({results[best_name]:.4f})", flush=True)
