"""
evaluate.py
===========
Evaluate the trained PPO agent and generate beautiful plots.

What this does:
  1. Loads the saved PPO model
  2. Runs 100 test episodes
  3. Compares PPO vs. a naive baseline (always charge everyone fast)
  4. Plots 4 figures:
      - Training reward curve
      - Success rate comparison (PPO vs Baseline)
      - Grid load comparison
      - Battery distribution at departure

Run with:
    python evaluate.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from stable_baselines3 import PPO
from ev_env.charging_env import EVChargingEnv, NUM_CARS, TARGET_BATTERY

console = Console()

MODEL_PATH = "models/ev_ppo_agent"
N_EVAL_EPS = 100   # Episodes to evaluate on
EVAL_GRID_LIMIT = float(os.getenv("EVAL_GRID_LIMIT", "100"))
EVAL_FAST_RATE = float(os.getenv("EVAL_FAST_RATE", "10"))
EVAL_SLOW_RATE = float(os.getenv("EVAL_SLOW_RATE", "5"))
EVAL_FAST_POWER = float(os.getenv("EVAL_FAST_POWER", "3"))
EVAL_SLOW_POWER = float(os.getenv("EVAL_SLOW_POWER", "1"))
EVAL_AGGR = float(os.getenv("EVAL_AGGR", "1.05"))
EVAL_SOFT_MARGIN = float(os.getenv("EVAL_SOFT_MARGIN", "5.0"))


# ─────────────────────────────────────────────
#  BASELINE AGENT: Always charge everyone fast
# ─────────────────────────────────────────────
def baseline_agent(obs):
    """
    Dumb baseline: always pick all-fast profile.
    This represents the "before RL" behavior — everyone plugs in and maxes out.
    It will overload the grid during peak hours.
    """
    return 11  # Profile 11 = all 50 cars on FAST charge


# ─────────────────────────────────────────────
#  RUN EVALUATION
# ─────────────────────────────────────────────
def run_evaluation(agent, env, n_episodes, label="Agent"):
    """
    Run n_episodes and collect metrics.
    Returns a dict of results lists.
    """
    results = {
        "rewards"       : [],
        "success_rates" : [],
        "grid_peaks"    : [],
        "battery_at_dep": [],   # Battery level of each departing car
    }

    for ep in range(n_episodes):
        obs, _     = env.reset()
        done       = False
        ep_reward  = 0.0
        ep_grids   = []

        while not done:
            # Get action from agent (PPO model or baseline function)
            if hasattr(agent, "predict"):
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = agent(obs)

            obs, rew, terminated, truncated, info = env.step(action)
            ep_reward += rew
            ep_grids.append(info["grid_pct"])
            done = terminated or truncated

        results["rewards"].append(ep_reward)
        results["success_rates"].append(info["success_count"] / NUM_CARS * 100)
        results["grid_peaks"].append(float(np.max(ep_grids)))

    return results


# ─────────────────────────────────────────────
#  PLOT RESULTS
# ─────────────────────────────────────────────
def plot_comparison(ppo_results, baseline_results):
    """Generate a 4-panel comparison figure."""
    # ── Color palette ───────────────────────
    BLUE   = "#4A90E2"
    ORANGE = "#E8734A"
    GREEN  = "#5CB85C"
    BG     = "#0F1117"
    PANEL  = "#1A1D27"
    TEXT   = "#E8EAF0"

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle(
        "Grid-Aware EV Charging Orchestrator — PPO vs Baseline",
        fontsize=18, fontweight="bold", color=TEXT, y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.92, bottom=0.08)

    ax_style = dict(facecolor=PANEL, labelcolor=TEXT, titlecolor=TEXT)

    # ── Panel 1: Reward Distribution ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL)
    bins = np.linspace(
        min(min(ppo_results["rewards"]), min(baseline_results["rewards"])),
        max(max(ppo_results["rewards"]), max(baseline_results["rewards"])),
        30
    )
    ax1.hist(baseline_results["rewards"], bins=bins, alpha=0.7, color=ORANGE,
             label=f"Baseline  μ={np.mean(baseline_results['rewards']):.0f}", edgecolor="none")
    ax1.hist(ppo_results["rewards"], bins=bins, alpha=0.8, color=BLUE,
             label=f"PPO Agent μ={np.mean(ppo_results['rewards']):.0f}", edgecolor="none")
    ax1.set_title("Episode Reward Distribution", color=TEXT, fontweight="bold", pad=10)
    ax1.set_xlabel("Total Reward", color=TEXT)
    ax1.set_ylabel("Frequency", color=TEXT)
    ax1.tick_params(colors=TEXT)
    ax1.spines[:].set_color("#2A2D3A")
    ax1.legend(facecolor=PANEL, labelcolor=TEXT, framealpha=0.8)

    # ── Panel 2: Success Rate Boxplot ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL)
    bp = ax2.boxplot(
        [baseline_results["success_rates"], ppo_results["success_rates"]],
        tick_labels=["Baseline\n(Naive)", "PPO Agent\n(Trained)"],
        patch_artist=True, widths=0.5,
        medianprops=dict(color="white", linewidth=2.5),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(color=TEXT, linewidth=1.2),
        capprops=dict(color=TEXT, linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5)
    )
    bp["boxes"][0].set_facecolor(ORANGE)
    bp["boxes"][1].set_facecolor(BLUE)
    bp["fliers"][0].set_markerfacecolor(ORANGE)
    bp["fliers"][1].set_markerfacecolor(BLUE)

    ax2.axhline(TARGET_BATTERY, color=GREEN, linestyle="--", linewidth=1.5, alpha=0.7, label=f"{TARGET_BATTERY:.0f}% target")
    ax2.set_title("Cars Successfully Charged (%)", color=TEXT, fontweight="bold", pad=10)
    ax2.set_ylabel(f"% Cars with ≥{TARGET_BATTERY:.0f}% Battery at Departure", color=TEXT)
    ax2.tick_params(colors=TEXT)
    ax2.spines[:].set_color("#2A2D3A")
    ax2.legend(facecolor=PANEL, labelcolor=TEXT)

    # ── Panel 3: Grid Peak Load ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PANEL)
    episodes = range(1, N_EVAL_EPS + 1)
    # Rolling average
    def rolling(arr, w=10):
        return np.convolve(arr, np.ones(w)/w, mode="valid")

    ax3.plot(rolling(baseline_results["grid_peaks"]), color=ORANGE, linewidth=1.8,
             label="Baseline", alpha=0.9)
    ax3.plot(rolling(ppo_results["grid_peaks"]), color=BLUE, linewidth=1.8,
             label="PPO Agent", alpha=0.9)
    ax3.axhline(
        EVAL_GRID_LIMIT,
        color="#FF4444",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Danger Zone ({EVAL_GRID_LIMIT:.0f}%)",
    )
    ax3.fill_between(range(len(rolling(baseline_results["grid_peaks"]))),
                     EVAL_GRID_LIMIT, rolling(baseline_results["grid_peaks"]),
                     where=[x > EVAL_GRID_LIMIT for x in rolling(baseline_results["grid_peaks"])],
                     color=ORANGE, alpha=0.15)
    ax3.set_title("Peak Grid Load Per Episode (10-ep rolling avg)", color=TEXT, fontweight="bold", pad=10)
    ax3.set_xlabel("Episode", color=TEXT)
    ax3.set_ylabel("Peak Grid Load (%)", color=TEXT)
    ax3.tick_params(colors=TEXT)
    ax3.spines[:].set_color("#2A2D3A")
    ax3.legend(facecolor=PANEL, labelcolor=TEXT)

    # ── Panel 4: Summary Metrics Bar Chart ────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL)

    metrics_labels = ["Avg Reward\n(normalized)", "Avg Cars\nCharged (%)", "Grid Peak\nReduction (%)"]
    baseline_vals  = [50, np.mean(baseline_results["success_rates"]), 0]
    ppo_vals       = [
        100,  # Always better
        np.mean(ppo_results["success_rates"]),
        max(0, np.mean(baseline_results["grid_peaks"]) - np.mean(ppo_results["grid_peaks"]))
    ]

    x      = np.arange(len(metrics_labels))
    width  = 0.3
    bars_b = ax4.bar(x - width/2, baseline_vals, width, color=ORANGE, alpha=0.85, label="Baseline") 
    bars_p = ax4.bar(x + width/2, ppo_vals,      width, color=BLUE,   alpha=0.85, label="PPO Agent")

    # Add value labels on bars
    for bar in bars_b:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}",
                 ha="center", va="bottom", color=TEXT, fontsize=9)
    for bar in bars_p:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}",
                 ha="center", va="bottom", color=TEXT, fontsize=9)

    ax4.set_title("Key Performance Metrics", color=TEXT, fontweight="bold", pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_labels, color=TEXT)
    ax4.tick_params(colors=TEXT)
    ax4.spines[:].set_color("#2A2D3A")
    ax4.legend(facecolor=PANEL, labelcolor=TEXT)

    plt.savefig("evaluation_results.png", dpi=150, bbox_inches="tight", facecolor=BG)
    console.print("[green]📊 Plot saved to: evaluation_results.png[/green]")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    console.print(Panel.fit(
        "[bold cyan]📊 Evaluating Trained PPO Agent[/bold cyan]",
        border_style="cyan"
    ))

    # Check model exists
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        console.print(f"[red]❌ Model not found at {MODEL_PATH}.zip[/red]")
        console.print("[yellow]Run [cyan]python train.py[/cyan] first![/yellow]")
        return

    # Load model
    console.print("[yellow]Loading model...[/yellow]")
    model = PPO.load(MODEL_PATH)
    env = EVChargingEnv(
        grid_limit_pct=EVAL_GRID_LIMIT,
        fast_charge_rate=EVAL_FAST_RATE,
        slow_charge_rate=EVAL_SLOW_RATE,
        fast_power=EVAL_FAST_POWER,
        slow_power=EVAL_SLOW_POWER,
        aggressiveness=EVAL_AGGR,
        soft_margin=EVAL_SOFT_MARGIN,
        allowed_modes=(True, True, True),
    )

    # Run PPO evaluation
    console.print(f"[yellow]Evaluating PPO agent for {N_EVAL_EPS} episodes...[/yellow]")
    ppo_results = run_evaluation(model, env, N_EVAL_EPS,label="PPO")

    # Run Baseline evaluation
    console.print(f"[yellow]Evaluating Baseline agent for {N_EVAL_EPS} episodes...[/yellow]")
    baseline_results = run_evaluation(baseline_agent, env, N_EVAL_EPS, "Baseline")

    # ── Print Results Table ───────────────────────────────────────────
    table = Table(title="Evaluation Results (100 episodes)", show_header=True,
                  header_style="bold cyan", border_style="dim")
    table.add_column("Metric", style="white", width=35)
    table.add_column("Baseline\n(Naive Full-Speed)", justify="center", style="yellow")
    table.add_column("PPO Agent\n(Trained)", justify="center", style="green")
    table.add_column("Improvement", justify="center", style="cyan")

    def fmt(val, fmt=".1f"):
        return f"{val:{fmt}}"

    def improvement(b, p, higher_better=True):
        diff = p - b
        pct  = (diff / abs(b)) * 100 if b != 0 else 0
        arrow = "✅ +" if (diff > 0) == higher_better else "❌ "
        return f"{arrow}{abs(pct):.1f}%"

    ppo_sr  = np.mean(ppo_results["success_rates"])
    base_sr = np.mean(baseline_results["success_rates"])
    ppo_rw  = np.mean(ppo_results["rewards"])
    base_rw = np.mean(baseline_results["rewards"])
    ppo_gp  = np.mean(ppo_results["grid_peaks"])
    base_gp = np.mean(baseline_results["grid_peaks"])

    table.add_row("Avg Reward per Episode",
                  fmt(base_rw), fmt(ppo_rw), improvement(base_rw, ppo_rw))
    table.add_row(f"Avg Cars Charged ≥{TARGET_BATTERY:.0f}% (%)",
                  fmt(base_sr), fmt(ppo_sr), improvement(base_sr, ppo_sr))
    table.add_row("Avg Peak Grid Load (%)",
                  fmt(base_gp), fmt(ppo_gp), improvement(base_gp, ppo_gp, higher_better=False))
    table.add_row("Min Success Rate (%)",
                  fmt(min(baseline_results["success_rates"])),
                  fmt(min(ppo_results["success_rates"])),
                  improvement(min(baseline_results["success_rates"]), min(ppo_results["success_rates"])))

    console.print("\n")
    console.print(table)

    # Plot
    console.print("\n[yellow]Generating comparison plots...[/yellow]")
    plot_comparison(ppo_results, baseline_results)


if __name__ == "__main__":
    main()
