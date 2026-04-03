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
TOTAL_TIMESTEPS  = 2_000_000  # Longer training for better convergence
N_ENVS           = 8          # Run 8 environments in parallel (faster)
MODEL_SAVE_PATH  = "models/ev_ppo_agent"
LOG_DIR          = "logs/"

os.makedirs("models", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


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

    # ── Step 1: Create vectorized environment ────────────────────────
    # make_vec_env runs N_ENVS copies of the environment in parallel.
    # This means we collect 4x more experience per second → faster training.
    console.print(f"\n[yellow]⚙ Creating {N_ENVS} parallel environments...[/yellow]")
    vec_env = make_vec_env(EVChargingEnv, n_envs=N_ENVS)

    # ── Step 2: Create evaluation environment ────────────────────────
    # A separate env just for evaluating policy during training
    eval_env = Monitor(EVChargingEnv())
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

    # Wider network for the 102-dim observation space with 13-action output
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # Wider layers for better representation
    )

    # Linear learning rate schedule: starts at 3e-4, decays to 0
    def lr_schedule(progress_remaining):
        """Linear decay: 1.0 at start → 0.0 at end."""
        return 3e-4 * progress_remaining

    model = PPO(
        policy          = "MlpPolicy",    # Feedforward neural network
        env             = vec_env,
        learning_rate   = lr_schedule,     # Linear decay for fine convergence
        n_steps         = 4096,            # Steps collected before each policy update
        batch_size      = 256,             # Larger batch → smoother gradient estimates
        n_epochs        = 10,              # Times to reuse each batch of data
        clip_range      = 0.2,             # Standard PPO clip range
        gamma           = 0.995,           # High discount: strongly value future rewards
        gae_lambda      = 0.98,            # Higher λ for less-biased advantage estimates
        ent_coef        = 0.05,            # More exploration early (13 actions need it)
        vf_coef         = 0.5,             # Value function loss weight
        max_grad_norm   = 0.5,             # Gradient clipping for stability
        policy_kwargs   = policy_kwargs,
        tensorboard_log = LOG_DIR,         # Log for TensorBoard visualization
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
    test_env  = EVChargingEnv()
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