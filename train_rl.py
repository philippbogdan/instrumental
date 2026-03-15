"""
RL Agent for Sound Matching — trains a PPO agent to learn knob-turning strategy.
"""

import gymnasium as gym
import numpy as np
import torch
import librosa
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from src.synth import SynthPatch
from src.losses import get_loss


def extract_features(audio_np, sr=44100):
    """Extract a compact feature vector from audio."""
    if len(audio_np) < 512:
        audio_np = np.pad(audio_np, (0, 512 - len(audio_np)))
    mfcc = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=audio_np, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio_np, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=audio_np)
    rms = librosa.feature.rms(y=audio_np)
    return np.array([
        *np.mean(mfcc, axis=1),      # 13 values
        np.mean(centroid) / 22050,    # 1 value, normalized
        np.mean(rolloff) / 22050,     # 1 value, normalized
        np.mean(flatness),            # 1 value
        np.mean(rms),                 # 1 value
    ], dtype=np.float32)  # total: 17 features


N_PARAMS = 18
N_FEATURES = 17
MAX_STEPS = 50


class SynthMatchEnv(gym.Env):
    """Gymnasium environment: match a target sound by adjusting synth knobs."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.synth = SynthPatch()
        self.loss_fn = get_loss("stft")  # faster than matching loss for training
        self.n_params = N_PARAMS
        self.n_features = N_FEATURES

        # Observation: current params + target features + current features + diff
        obs_dim = self.n_params + self.n_features * 3  # params + target + current + diff
        self.observation_space = spaces.Box(-1.0, 2.0, shape=(obs_dim,), dtype=np.float32)

        # Action: adjust each param by [-0.1, +0.1]
        self.action_space = spaces.Box(-0.1, 0.1, shape=(self.n_params,), dtype=np.float32)

        self.target_features = None
        self.target_audio = None
        self.current_params = None
        self.prev_loss = None
        self.step_count = 0
        self.f0 = 440.0
        self.dur = 0.15

    def _compute_loss(self, params):
        audio = self.synth.render(
            torch.tensor(params, dtype=torch.float32),
            f0_hz=self.f0, duration=self.dur, note_duration=self.dur * 0.9
        ).detach()
        target = self.target_audio.unsqueeze(0)
        ml = min(target.shape[1], audio.shape[1])
        loss = self.loss_fn(
            audio[:, :ml].unsqueeze(0),
            target[:, :ml].unsqueeze(0)
        ).item()
        return loss, audio.squeeze().numpy()

    def _get_obs(self, current_features):
        diff = self.target_features - current_features
        return np.concatenate([
            self.current_params,
            self.target_features,
            current_features,
            diff
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random pitch
        self.f0 = float(np.random.choice([220.6, 278.0, 294.5, 330.0, 440.0]))

        # Random target
        gt_params = np.random.rand(self.n_params).astype(np.float32)
        target_audio = self.synth.render(
            torch.tensor(gt_params, dtype=torch.float32),
            f0_hz=self.f0, duration=self.dur, note_duration=self.dur * 0.9
        ).detach()
        self.target_audio = target_audio.squeeze(0)
        self.target_features = extract_features(self.target_audio.numpy(), sr=44100)

        # Random starting params
        self.current_params = np.random.rand(self.n_params).astype(np.float32)
        self.prev_loss, current_audio = self._compute_loss(self.current_params)
        current_features = extract_features(current_audio, sr=44100)

        self.step_count = 0
        return self._get_obs(current_features), {}

    def step(self, action):
        # Apply action (clamp to [0, 1])
        self.current_params = np.clip(self.current_params + action, 0.0, 1.0)

        # Compute new loss
        new_loss, current_audio = self._compute_loss(self.current_params)
        current_features = extract_features(current_audio, sr=44100)

        # Reward = improvement
        reward = (self.prev_loss - new_loss) * 10.0  # scale up for better signal
        self.prev_loss = new_loss

        self.step_count += 1
        terminated = new_loss < 0.1  # win condition
        truncated = self.step_count >= MAX_STEPS

        # Bonus for getting very close
        if terminated:
            reward += 50.0

        return self._get_obs(current_features), reward, terminated, truncated, {"loss": new_loss}


class ProgressCallback(BaseCallback):
    """Print progress during training."""

    def __init__(self, print_freq=5000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_losses = []

    def _on_step(self):
        # Collect episode info
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
            if "loss" in info:
                self.episode_losses.append(info["loss"])

        if self.num_timesteps % self.print_freq == 0:
            recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
            recent_losses = self.episode_losses[-100:] if self.episode_losses else [0]
            mean_r = np.mean(recent_rewards)
            mean_l = np.mean(recent_losses)
            print(f"  step {self.num_timesteps:>7}: mean_reward={mean_r:>+8.2f}  mean_loss={mean_l:.4f}")
        return True


def make_env():
    def _init():
        return SynthMatchEnv()
    return _init


if __name__ == "__main__":
    import time

    N_ENVS = 8
    TOTAL_STEPS = 1_000_000

    print(f"=== RL Training: PPO, {N_ENVS} parallel envs, {TOTAL_STEPS} steps ===")
    print(f"Observation: {N_PARAMS + N_FEATURES * 3}-dim")
    print(f"Action: {N_PARAMS}-dim continuous [-0.1, +0.1]")
    print(f"Episode: {MAX_STEPS} steps, random target each reset")
    print()

    # Parallel environments
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,         # steps per env before update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    callback = ProgressCallback(print_freq=10000)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)
    elapsed = time.time() - t0

    print(f"\nTraining done: {TOTAL_STEPS} steps in {elapsed:.1f}s ({TOTAL_STEPS/elapsed:.0f} steps/s)")

    # Save model
    model.save("output/rl_agent_v1")
    print("Model saved to output/rl_agent_v1.zip")

    # Quick eval: match a known target
    print("\n=== Evaluation ===")
    eval_env = SynthMatchEnv()
    obs, _ = eval_env.reset()

    print(f"Target f0: {eval_env.f0} Hz")
    print(f"Initial loss: {eval_env.prev_loss:.4f}")

    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if step % 10 == 0:
            print(f"  step {step}: loss={info['loss']:.4f} reward={reward:.4f}")
        if terminated:
            print(f"  MATCHED at step {step}! loss={info['loss']:.4f}")
            break

    print(f"Final loss: {info['loss']:.4f}")
    env.close()
