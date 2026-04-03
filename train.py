"""
train.py
========
Train the PPO Agent on our EV Charging Environment

What happens here:
  1. We create the environment
  2. We wrap it so multiple copies run in parallel (faster training)
  3. We initialize PPO — the RL algorithm
  4. We train for 500,000 timesteps (~5 min on CPU)
  5. We save the trained model and print results

Run with:
    python train.py
"""

import os
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from ev_env.charging_env import EVChargingEnv

console = Console()

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
TOTAL_TIMESTEPS  = int(os.getenv("TOTAL_TIMESTEPS", "2000000"))
N_ENVS           = 8          # Run 8 environments in parallel (faster)
MODEL_SAVE_PATH  = "models/ev_ppo_agent"
LOG_DIR          = "logs/"
GRID_LIMIT_MIN   = float(os.getenv("GRID_LIMIT_MIN", "85"))
GRID_LIMIT_MAX   = float(os.getenv("GRID_LIMIT_MAX", "100"))
EVAL_GRID_LIMIT  = float(os.getenv("EVAL_GRID_LIMIT", "100"))

FAST_RATE_MIN    = float(os.getenv("FAST_RATE_MIN", "8"))
FAST_RATE_MAX    = float(os.getenv("FAST_RATE_MAX", "12"))
SLOW_RATE_MIN    = float(os.getenv("SLOW_RATE_MIN", "3"))
SLOW_RATE_MAX    = float(os.getenv("SLOW_RATE_MAX", "7"))
FAST_POWER_MIN   = float(os.getenv("FAST_POWER_MIN", "2.5"))
FAST_POWER_MAX   = float(os.getenv("FAST_POWER_MAX", "4.5"))
SLOW_POWER_MIN   = float(os.getenv("SLOW_POWER_MIN", "0.8"))
SLOW_POWER_MAX   = float(os.getenv("SLOW_POWER_MAX", "1.8"))
AGGR_MIN         = float(os.getenv("AGGR_MIN", "1.0"))
AGGR_MAX         = float(os.getenv("AGGR_MAX", "1.15"))
SOFT_MARGIN      = float(os.getenv("SOFT_MARGIN", "5.0"))

EVAL_FAST_RATE   = float(os.getenv("EVAL_FAST_RATE", "10"))
EVAL_SLOW_RATE   = float(os.getenv("EVAL_SLOW_RATE", "5"))
EVAL_FAST_POWER  = float(os.getenv("EVAL_FAST_POWER", "3"))
EVAL_SLOW_POWER  = float(os.getenv("EVAL_SLOW_POWER", "1"))
EVAL_AGGR        = float(os.getenv("EVAL_AGGR", "1.05"))

GRID_LIMIT_MIN, GRID_LIMIT_MAX = sorted((GRID_LIMIT_MIN, GRID_LIMIT_MAX))
FAST_RATE_MIN, FAST_RATE_MAX = sorted((FAST_RATE_MIN, FAST_RATE_MAX))
SLOW_RATE_MIN, SLOW_RATE_MAX = sorted((SLOW_RATE_MIN, SLOW_RATE_MAX))
FAST_POWER_MIN, FAST_POWER_MAX = sorted((FAST_POWER_MIN, FAST_POWER_MAX))
SLOW_POWER_MIN, SLOW_POWER_MAX = sorted((SLOW_POWER_MIN, SLOW_POWER_MAX))
AGGR_MIN, AGGR_MAX = sorted((AGGR_MIN, AGGR_MAX))

os.makedirs("models", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def resolve_training_device():
    """Prefer DirectML on Windows; safely fall back to CUDA/CPU."""
    requested = os.getenv("TRAIN_DEVICE", "auto").strip()
    if requested and requested.lower() != "auto":
        return requested, f"forced by TRAIN_DEVICE={requested}"

    if os.name == "nt":
        try:
            import importlib

            torch_directml = importlib.import_module("torch_directml")
            dml_device = torch_directml.device()
            _ = torch.zeros(1, device=dml_device)
            return dml_device, "DirectML"
        except Exception as exc:
            console.print(f"[yellow]DirectML unavailable ({exc}). Falling back.[/yellow]")

    if torch.cuda.is_available():
        return "cuda", "CUDA"

    return "cpu", "CPU"


# ─────────────────────────────────────────────
#  CUSTOM CALLBACK: Print episode stats
# ─────────────────────────────────────────────
class TrainingCallback(BaseCallback):
    """
    A callback is a function called automatically by SB3 during training.
    We use it to print pretty progress updates every 5000 steps.

    BaseCallback is provided by SB3 — we just override `_on_step`.
    """
    def __init__(self, print_every=5000, verbose=0):
        super().__init__(verbose)
        self.print_every   = print_every
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called after every step across all parallel envs. Must return True."""

        # SB3 stores episode info in self.locals["infos"]
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_info = info["episode"]
                self.episode_rewards.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])

        # Print summary every N steps
        if self.num_timesteps % self.print_every == 0 and self.episode_rewards:
            recent_rewards = self.episode_rewards[-20:]  # last 20 episodes
            avg_reward     = np.mean(recent_rewards)
            console.print(
                f"  [cyan]Step {self.num_timesteps:>7,}[/cyan] | "
                f"Avg Reward (last 20 eps): [{'green' if avg_reward > 0 else 'red'}]{avg_reward:+.1f}[/]"
            )

        return True  # Returning False would stop training


# ─────────────────────────────────────────────
#  MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────
def train():
    console.print(Panel.fit(
        "[bold cyan]🚗 Grid-Aware EV Charging Orchestrator[/bold cyan]\n"
        "[white]Training PPO Agent...[/white]",
        border_style="cyan"
    ))
    console.print(
        f"[dim]Grid limit curriculum: {GRID_LIMIT_MIN:.0f}% → {GRID_LIMIT_MAX:.0f}% | "
        f"Eval limit: {EVAL_GRID_LIMIT:.0f}%[/dim]"
    )
    console.print(
        f"[dim]Rate curriculum F/S: {FAST_RATE_MIN:.1f}-{FAST_RATE_MAX:.1f} / "
        f"{SLOW_RATE_MIN:.1f}-{SLOW_RATE_MAX:.1f} %/min[/dim]"
    )
    console.print(
        f"[dim]Power curriculum F/S: {FAST_POWER_MIN:.1f}-{FAST_POWER_MAX:.1f} / "
        f"{SLOW_POWER_MIN:.1f}-{SLOW_POWER_MAX:.1f} units | "
        f"Aggressiveness: {AGGR_MIN:.2f}-{AGGR_MAX:.2f}[/dim]"
    )

    training_device, device_note = resolve_training_device()
    console.print(f"[dim]Training device: {training_device} ({device_note})[/dim]")

    # ── Step 1: Create vectorized environment ────────────────────────
    # make_vec_env runs N_ENVS copies of the environment in parallel.
    # This means we collect 4x more experience per second → faster training.
    console.print(f"\n[yellow]⚙ Creating {N_ENVS} parallel environments...[/yellow]")

    def make_training_env():
        return EVChargingEnv(
            grid_limit_pct=EVAL_GRID_LIMIT,
            randomize_grid_limit=True,
            grid_limit_range=(GRID_LIMIT_MIN, GRID_LIMIT_MAX),
            fast_charge_rate=EVAL_FAST_RATE,
            slow_charge_rate=EVAL_SLOW_RATE,
            fast_power=EVAL_FAST_POWER,
            slow_power=EVAL_SLOW_POWER,
            aggressiveness=EVAL_AGGR,
            soft_margin=SOFT_MARGIN,
            randomize_control_params=True,
            fast_rate_range=(FAST_RATE_MIN, FAST_RATE_MAX),
            slow_rate_range=(SLOW_RATE_MIN, SLOW_RATE_MAX),
            fast_power_range=(FAST_POWER_MIN, FAST_POWER_MAX),
            slow_power_range=(SLOW_POWER_MIN, SLOW_POWER_MAX),
            aggressiveness_range=(AGGR_MIN, AGGR_MAX),
            allowed_modes=(True, True, True),
        )

    vec_env = make_vec_env(make_training_env, n_envs=N_ENVS)

    # ── Step 2: Create evaluation environment ────────────────────────
    # A separate env just for evaluating policy during training
    eval_env = Monitor(EVChargingEnv(
        grid_limit_pct=EVAL_GRID_LIMIT,
        fast_charge_rate=EVAL_FAST_RATE,
        slow_charge_rate=EVAL_SLOW_RATE,
        fast_power=EVAL_FAST_POWER,
        slow_power=EVAL_SLOW_POWER,
        aggressiveness=EVAL_AGGR,
        soft_margin=SOFT_MARGIN,
        allowed_modes=(True, True, True),
    ))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best/",
        log_path=LOG_DIR,
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )

    # ── Step 3: Initialize PPO ────────────────────────────────────────
    # PPO = Proximal Policy Optimization
    # It uses a neural network (policy) that takes state → outputs action probabilities
    #
    # policy="MlpPolicy" → Multi-layer Perceptron (simple feedforward neural net)
    #   Input: 102 numbers (our observation)
    #   Hidden: 2 layers of 256 neurons each
    #   Output: 13 probabilities (one per charging profile)
    console.print("[yellow]⚙ Initializing PPO algorithm...[/yellow]")

    # 3-layer network for richer feature extraction on high-dimensional observation
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
    )

    # Linear learning rate schedule: starts at 3e-4, decays to 0
    def lr_schedule(progress_remaining):
        """Linear decay: starts at 3e-4, decays to 3e-5 (not zero — keeps learning)."""
        return 3e-5 + (3e-4 - 3e-5) * progress_remaining

    model = PPO(
        policy          = "MlpPolicy",    # Feedforward neural network
        env             = vec_env,
        learning_rate   = lr_schedule,     # Linear decay with floor for fine convergence
        n_steps         = 2048,            # Steps per update (2048 * 8 envs = 16k/update)
        batch_size      = 128,             # Smaller batches → more gradient updates per rollout
        n_epochs        = 15,              # More passes over each batch → better sample efficiency
        clip_range      = 0.2,             # Standard PPO clip range
        gamma           = 0.99,            # Discount factor (0.99 fits 180-step episodes well)
        gae_lambda      = 0.95,            # Standard GAE lambda for stable advantages
        ent_coef        = 0.01,            # Mild exploration (13 actions don't need much)
        vf_coef         = 0.5,             # Value function loss weight
        max_grad_norm   = 0.5,             # Gradient clipping for stability
        policy_kwargs   = policy_kwargs,
        tensorboard_log = LOG_DIR,         # Log for TensorBoard visualization
        device          = training_device,
        verbose         = 0,
    )

    console.print(f"  [dim]Policy network: {sum(p.numel() for p in model.policy.parameters()):,} parameters[/dim]")

    # ── Step 4: TRAIN ─────────────────────────────────────────────────
    console.print(f"\n[bold green]🎓 Training for {TOTAL_TIMESTEPS:,} timesteps...[/bold green]")
    console.print("[dim]  (reward should steadily increase from negative → positive)[/dim]\n")

    training_cb = TrainingCallback(print_every=25_000)

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [training_cb, eval_callback],
        progress_bar    = True,
    )

    # ── Step 5: Save the trained model ───────────────────────────────
    model.save(MODEL_SAVE_PATH)
    console.print(f"\n[bold green]✅ Model saved to: {MODEL_SAVE_PATH}.zip[/bold green]")

    # ── Step 6: Quick sanity check ────────────────────────────────────
    console.print("\n[yellow]🔍 Running quick sanity check (10 episodes)...[/yellow]")
    test_env = EVChargingEnv(
        grid_limit_pct=EVAL_GRID_LIMIT,
        fast_charge_rate=EVAL_FAST_RATE,
        slow_charge_rate=EVAL_SLOW_RATE,
        fast_power=EVAL_FAST_POWER,
        slow_power=EVAL_SLOW_POWER,
        aggressiveness=EVAL_AGGR,
        soft_margin=SOFT_MARGIN,
        allowed_modes=(True, True, True),
    )
    total_success = 0

    for ep in range(10):
        obs, _ = test_env.reset()
        done   = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = test_env.step(action)
            ep_rew += rew
            done = terminated or truncated
        total_success += info["success_count"]
        console.print(f"  Ep {ep+1:2d}: reward={ep_rew:+.1f} | cars_charged={info['success_count']}/50")

    avg_success = total_success / 10
    console.print(f"\n[bold]Average cars charged: {avg_success:.1f}/50 ({avg_success/50*100:.1f}%)[/bold]")
    console.print("\n[dim]Run [cyan]python evaluate.py[/cyan] for full analysis[/dim]")
    console.print("[dim]Run [cyan]streamlit run dashboard.py[/cyan] for live demo[/dim]")


if __name__ == "__main__":
    train()