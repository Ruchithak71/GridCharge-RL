"""
dashboard.py
============
Streamlit Live Demo Dashboard — The Hackathon Showstopper 🏆

This creates an interactive browser UI showing the RL agent making
real-time decisions every second.

Run with:
    streamlit run dashboard.py

What it shows:
  - 50 car cards with live battery meters
  - Real-time grid load gauge
    - Agent/controller action decided each step
  - Metrics: success rate, avg battery, grid stress
"""

import os
import sys
import time
import subprocess
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from ev_env.charging_env import (
    EVChargingEnv,
    NUM_CARS,
    TARGET_BATTERY,
    MAX_STEPS,
    FAST_CHARGE_RATE,
    SLOW_CHARGE_RATE,
        FAST_POWER,
        SLOW_POWER,
        ACTION_WAIT,
        ACTION_SLOW,
        ACTION_FAST,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ EV Charging Orchestrator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS (dark glassmorphism theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1525 50%, #0a1020 100%); }

    /* Main title */
    .main-title {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #4A90E2, #00D4AA);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center; color: #8892A4; font-size: 1rem; margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1.2rem 1.5rem;
        backdrop-filter: blur(10px);
        text-align: center;
    }
    .metric-val  { font-size: 2rem; font-weight: 700; color: #4A90E2; }
    .metric-lbl  { font-size: 0.85rem; color: #8892A4; margin-top: 0.2rem; }

    /* Car grid */
    .car-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; padding: 0.5rem;
        margin: 2px; font-size: 0.7rem;
        transition: all 0.3s ease;
    }
    .car-fast  { border-color: #4A90E2 !important; background: rgba(74,144,226,0.1) !important; }
    .car-slow  { border-color: #00D4AA !important; background: rgba(0,212,170,0.06) !important; }
    .car-wait  { border-color: #555 !important; }
    .car-done  { opacity: 0.35; }

    /* Action badge */
    .badge-fast { background:#4A90E2; color:white; border-radius:4px; padding:1px 6px; font-size:0.65rem; }
    .badge-slow { background:#00D4AA; color:#000; border-radius:4px; padding:1px 6px; font-size:0.65rem; }
    .badge-wait { background:#555; color:white; border-radius:4px; padding:1px 6px; font-size:0.65rem; }

    /* Battery bar */
    .batt-bar { height: 6px; border-radius: 3px; margin-top: 4px; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: rgba(10,14,26,0.95); }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4A90E2, #00D4AA);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.5rem 1.5rem;
        width: 100%; transition: opacity 0.2s;
    }
    .stButton>button:hover { opacity: 0.85; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL (cached so it only loads once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "models/ev_ppo_agent"
    model_zip = f"{model_path}.zip"

    if not os.path.exists(model_zip):
        return None, f"Model file not found at {model_zip}"

    # Probe model loading in a subprocess so native crashes don't kill Streamlit.
    probe_code = (
        "from stable_baselines3 import PPO; "
        f"PPO.load(r'{model_path}'); "
        "print('OK')"
    )

    try:
        probe = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as e:
        return None, f"Model probe failed: {e}"

    if probe.returncode != 0:
        stderr = (probe.stderr or "").strip()
        if probe.returncode < 0:
            return None, f"Model probe crashed with return code {probe.returncode}."
        if stderr:
            return None, f"Model probe failed: {stderr.splitlines()[-1]}"
        return None, f"Model probe failed with return code {probe.returncode}."

    try:
        model = PPO.load(model_path)

        # New environment includes dynamic grid-limit in observation.
        expected_shape = EVChargingEnv(render_mode=None).observation_space.shape
        model_shape = getattr(model.observation_space, "shape", None)
        if model_shape != expected_shape:
            return None, (
                f"Model observation shape {model_shape} does not match env shape {expected_shape}. "
                "Retrain with `python train.py`."
            )

        return model, None
    except Exception as e:
        return None, str(e)


CONTROL_PRESETS = {
    "Safe": {
        "fast_rate": 8.0,
        "slow_rate": 3.0,
        "fast_power": 2.0,
        "slow_power": 0.6,
        "aggressiveness": 1.00,
        "soft_margin": 9.0,
    },
    "Balanced": {
        "fast_rate": float(FAST_CHARGE_RATE),
        "slow_rate": float(SLOW_CHARGE_RATE),
        "fast_power": float(FAST_POWER),
        "slow_power": float(SLOW_POWER),
        "aggressiveness": 1.05,
        "soft_margin": 5.0,
    },
    "Aggressive": {
        "fast_rate": 12.0,
        "slow_rate": 6.0,
        "fast_power": 4.8,
        "slow_power": 1.8,
        "aggressiveness": 1.12,
        "soft_margin": 2.5,
    },
}


def apply_control_preset(preset_name):
    preset = CONTROL_PRESETS[preset_name]
    for key, value in preset.items():
        st.session_state[key] = float(value)
    st.session_state["control_profile"] = preset_name


def infer_control_profile(fast_rate, slow_rate, fast_power, slow_power, aggressiveness, soft_margin):
    values = {
        "fast_rate": float(fast_rate),
        "slow_rate": float(slow_rate),
        "fast_power": float(fast_power),
        "slow_power": float(slow_power),
        "aggressiveness": float(aggressiveness),
        "soft_margin": float(soft_margin),
    }
    for name, preset in CONTROL_PRESETS.items():
        if all(abs(values[key] - float(preset[key])) <= 1e-6 for key in preset):
            return name
    return "Random"


# ─────────────────────────────────────────────
#  HELPER: Battery color
# ─────────────────────────────────────────────
def battery_color(pct):
    if pct >= TARGET_BATTERY:   return "#4CAF50"   # green
    elif pct >= 50: return "#FFC107"   # amber
    elif pct >= 30: return "#FF9800"   # orange
    else:           return "#F44336"   # red


def mode_to_label(mode):
    if mode == ACTION_FAST:
        return "FAST", "badge-fast"
    if mode == ACTION_SLOW:
        return "SLOW", "badge-slow"
    return "WAIT", "badge-wait"


def car_class(mode, done):
    if done:
        return "car-card car-done"
    if mode == ACTION_FAST:
        return "car-card car-fast"
    if mode == ACTION_SLOW:
        return "car-card car-slow"
    return "car-card car-wait"


def mode_to_rate(mode, env):
    if mode == ACTION_FAST:
        return env.fast_charge_rate
    if mode == ACTION_SLOW:
        return env.slow_charge_rate
    return 0.0


def mode_to_power(mode, env):
    if mode == ACTION_FAST:
        return env.fast_power
    if mode == ACTION_SLOW:
        return env.slow_power
    return 0.0


def resolve_allowed_mode(preferred_mode, allow_fast, allow_slow, allow_wait):
    if preferred_mode == ACTION_FAST:
        order = (ACTION_FAST, ACTION_SLOW, ACTION_WAIT)
    elif preferred_mode == ACTION_SLOW:
        order = (ACTION_SLOW, ACTION_FAST, ACTION_WAIT)
    else:
        order = (ACTION_WAIT, ACTION_SLOW, ACTION_FAST)

    for mode in order:
        if mode == ACTION_FAST and allow_fast:
            return ACTION_FAST
        if mode == ACTION_SLOW and allow_slow:
            return ACTION_SLOW
        if mode == ACTION_WAIT and allow_wait:
            return ACTION_WAIT

    return ACTION_WAIT


def build_manual_modes(env, grid_limit_pct, soft_margin, allow_fast, allow_slow, allow_wait):
    """Deterministic no-training controller with fast>slow>wait priority bands."""
    modes = np.full(NUM_CARS, ACTION_WAIT, dtype=np.int8)
    active_idx = np.where(~env.cars_done)[0]
    if active_idx.size == 0:
        return modes, "idle"

    current_load = float(env.last_grid_pct)
    base_load = float(env._grid_base_load(env.sim_hour))
    limit = float(grid_limit_pct)
    soft_start = max(0.0, limit - soft_margin)
    target_load = min(limit * env.aggressiveness, 130.0)

    deficits = np.maximum(TARGET_BATTERY - env.battery, 0.0)
    urgency_desc = active_idx[np.argsort(-deficits[active_idx])]
    near_target = active_idx[np.argsort(deficits[active_idx])]

    def choose(preferred_mode):
        return resolve_allowed_mode(preferred_mode, allow_fast, allow_slow, allow_wait)

    if current_load < soft_start:
        band_state = "below-soft"
        projected = base_load

        for idx in urgency_desc:
            mode = choose(ACTION_FAST)
            add_power = mode_to_power(mode, env)

            if mode == ACTION_FAST and projected + add_power > soft_start and allow_slow:
                mode = choose(ACTION_SLOW)
                add_power = mode_to_power(mode, env)

            modes[idx] = mode
            projected += add_power

        desired_load = max(soft_start, target_load)
        if allow_fast and allow_slow:
            for idx in urgency_desc:
                if modes[idx] == ACTION_SLOW:
                    delta = env.fast_power - env.slow_power
                    if projected + delta <= desired_load:
                        modes[idx] = ACTION_FAST
                        projected += delta

    elif current_load <= limit:
        band_state = "near-limit"
        projected = base_load
        default_mode = choose(ACTION_SLOW)
        default_power = mode_to_power(default_mode, env)

        for idx in urgency_desc:
            modes[idx] = default_mode
            projected += default_power

        if projected > target_load and allow_wait:
            for idx in urgency_desc[::-1]:
                current_mode = modes[idx]
                replacement = choose(ACTION_WAIT)
                old_power = mode_to_power(current_mode, env)
                new_power = mode_to_power(replacement, env)
                modes[idx] = replacement
                projected += new_power - old_power
                if projected <= target_load:
                    break

        if allow_fast and allow_slow:
            for idx in urgency_desc:
                if modes[idx] == ACTION_SLOW:
                    delta = env.fast_power - env.slow_power
                    if projected + delta <= target_load:
                        modes[idx] = ACTION_FAST
                        projected += delta

    else:
        band_state = "over-limit"
        projected = base_load
        default_mode = choose(ACTION_WAIT)
        default_power = mode_to_power(default_mode, env)

        for idx in urgency_desc:
            modes[idx] = default_mode
            projected += default_power

        # Keep near-target vehicles in SLOW so they can depart early.
        if allow_slow:
            for idx in near_target:
                if modes[idx] != ACTION_WAIT:
                    continue
                replacement = choose(ACTION_SLOW)
                old_power = mode_to_power(modes[idx], env)
                new_power = mode_to_power(replacement, env)
                if projected + (new_power - old_power) <= limit:
                    modes[idx] = replacement
                    projected += new_power - old_power

        # If load cools down inside this branch, restore SLOW before FAST.
        if projected < soft_start and allow_slow:
            for idx in urgency_desc:
                if modes[idx] == ACTION_WAIT:
                    replacement = choose(ACTION_SLOW)
                    delta = mode_to_power(replacement, env) - mode_to_power(modes[idx], env)
                    if projected + delta <= soft_start:
                        modes[idx] = replacement
                        projected += delta

    modes[env.cars_done] = ACTION_WAIT
    return modes, band_state


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────────────────────────────
    st.markdown('<div class="main-title">⚡ Grid-Aware EV Charging Orchestrator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Reinforcement Learning Agent Managing 50 Electric Vehicles in Real-Time</div>', unsafe_allow_html=True)

    # ── Load Model (safe mode) ─────────────────────────────────────
    model, err = load_model()
    rl_available = model is not None

    # ── Sidebar Controls ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Controls")

        if "control_profile" not in st.session_state:
            st.session_state.control_profile = "Balanced"
        if "fast_rate" not in st.session_state:
            apply_control_preset(st.session_state.control_profile)

        st.markdown("### Preset Profiles")
        preset_cols = st.columns(3)
        if preset_cols[0].button("Safe", use_container_width=True):
            apply_control_preset("Safe")
        if preset_cols[1].button("Balanced", use_container_width=True):
            apply_control_preset("Balanced")
        if preset_cols[2].button("Aggressive", use_container_width=True):
            apply_control_preset("Aggressive")

        speed = st.slider("Simulation Speed", 0.05, 1.0, 0.15, 0.05,
                          help="Seconds per step (lower = faster)")
        grid_limit_pct = st.slider(
            "Allowed Grid Load (%)",
            min_value=70,
            max_value=110,
            value=100,
            step=1,
            help="Penalty increases above this threshold. Lower value = stricter grid safety.",
        )

        fast_rate = st.slider(
            "Fast Charge Rate (%/min)",
            min_value=1.0,
            max_value=15.0,
            step=0.5,
            key="fast_rate",
        )
        slow_rate = st.slider(
            "Slow Charge Rate (%/min)",
            min_value=1.0,
            max_value=10.0,
            step=0.5,
            key="slow_rate",
        )
        if slow_rate > fast_rate:
            slow_rate = fast_rate
            st.session_state["slow_rate"] = slow_rate
            st.caption("Slow rate clipped to fast rate to preserve fast > slow priority.")

        fast_power = st.slider(
            "Fast Power Draw (grid units)",
            min_value=0.5,
            max_value=6.0,
            step=0.1,
            key="fast_power",
        )
        slow_power = st.slider(
            "Slow Power Draw (grid units)",
            min_value=0.1,
            max_value=3.0,
            step=0.1,
            key="slow_power",
        )
        if slow_power > fast_power:
            slow_power = fast_power
            st.session_state["slow_power"] = slow_power
            st.caption("Slow power clipped to fast power to preserve fast > slow priority.")

        aggressiveness = st.slider(
            "Aggressiveness (Limit Multiplier)",
            min_value=1.00,
            max_value=1.15,
            step=0.01,
            help="Target load = grid limit x aggressiveness",
            key="aggressiveness",
        )
        soft_margin = st.slider(
            "Soft Band Margin (units below limit)",
            min_value=1.0,
            max_value=15.0,
            step=0.5,
            help="Below this margin controller prefers FAST; near limit it shifts to SLOW.",
            key="soft_margin",
        )

        control_profile = infer_control_profile(
            fast_rate,
            slow_rate,
            fast_power,
            slow_power,
            aggressiveness,
            soft_margin,
        )
        st.session_state.control_profile = control_profile
        if control_profile == "Random":
            st.caption("Control Profile: Random (custom slider mix)")
        else:
            st.caption(f"Control Profile: {control_profile}")

        st.markdown("### Allowed Charging Modes")
        allow_fast = st.checkbox("FAST", value=True)
        allow_slow = st.checkbox("SLOW", value=True)
        allow_wait = st.checkbox("WAIT", value=True)

        mode_valid = allow_fast or allow_slow or allow_wait
        if not mode_valid:
            st.error("Select at least one charging mode.")

        controller_options = ["Manual Controller"]
        if rl_available:
            controller_options.append("RL Agent")
        controller_mode = st.radio("Controller Mode", controller_options, index=0)

        st.markdown("---")
        st.markdown("### 🎨 Legend")
        st.markdown("""
        <span class="badge-fast">FAST</span> High priority mode<br><br>
        <span class="badge-slow">SLOW</span> Moderate mode<br><br>
        <span class="badge-wait">WAIT</span> Grid relief — 0%/min
        """, unsafe_allow_html=True)
        st.caption(f"Grid danger threshold: {grid_limit_pct}%")
        st.caption(f"Rates F/S: {fast_rate:.1f}/{slow_rate:.1f} %/min")
        st.caption(f"Power F/S: {fast_power:.1f}/{slow_power:.1f} units")
        st.caption(f"Target load multiplier: {aggressiveness:.2f}")
        st.markdown("---")
        st.markdown("### 📖 How It Works")
        st.markdown("""
        The controller observes:
        - 🔋 Battery % of each car
        - 🧮 Remaining charge deficit to 85%
        - ⚡ Current grid load and selected grid limit

        Cars only depart after reaching the target battery level.
        Manual mode uses your FAST/SLOW/WAIT rules directly.
        RL mode uses the trained policy but still respects selected mode checkboxes.
        """)

        run_btn = st.button("▶ Run New Episode", disabled=not mode_valid)

    if not rl_available:
        st.warning(
            f"RL model unavailable ({err}). Manual Controller mode is active."
        )

    if not (allow_fast or allow_slow or allow_wait):
        st.info("Choose at least one allowed mode to start charging.")
        return

    use_rl = controller_mode == "RL Agent" and rl_available

    # ── Session State ────────────────────────────────────────────────
    if "env" not in st.session_state or run_btn:
        st.session_state.grid_limit_pct = float(grid_limit_pct)
        st.session_state.env       = EVChargingEnv(
            render_mode=None,
            grid_limit_pct=st.session_state.grid_limit_pct,
            fast_charge_rate=fast_rate,
            slow_charge_rate=slow_rate,
            fast_power=fast_power,
            slow_power=slow_power,
            aggressiveness=aggressiveness,
            soft_margin=soft_margin,
            include_mode_assignments=True,
            allowed_modes=(allow_fast, allow_slow, allow_wait),
        )
        obs, _                     = st.session_state.env.reset()
        st.session_state.obs       = obs
        st.session_state.done      = False
        st.session_state.step      = 0
        st.session_state.rewards   = []
        st.session_state.grids     = []
        st.session_state.running   = True
        st.session_state.last_info = None  # Store final episode info
        st.session_state.last_band = "idle"
        # Rerun to show fresh state
        st.rerun()

    env = st.session_state.env
    env.set_grid_limit(float(grid_limit_pct))
    env.set_charge_rates(float(fast_rate), float(slow_rate))
    env.set_power_draw(float(fast_power), float(slow_power))
    env.set_aggressiveness(float(aggressiveness))
    env.set_soft_margin(float(soft_margin))
    env.set_allowed_modes(allow_fast, allow_slow, allow_wait)
    st.session_state.grid_limit_pct = float(grid_limit_pct)

    # ── Metric Row ───────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    metrics_placeholder = {
        "step"   : m1.empty(),
        "success": m2.empty(),
        "grid"   : m3.empty(),
        "battery": m4.empty(),
    }

    # ── Grid Load Chart ──────────────────────────────────────────────
    chart_col, cars_col = st.columns([1, 2])
    with chart_col:
        st.markdown("#### ⚡ Grid Load Over Time")
        grid_chart = st.empty()

    with cars_col:
        st.markdown("#### 🚗 Car Fleet Status")
        car_grid = st.empty()

    # ── Status Bar ───────────────────────────────────────────────────
    status_bar = st.empty()

    # ── SIMULATION LOOP ─────────────────────────────────────────────
    while not st.session_state.done:
        obs = st.session_state.obs

        # Agent picks action
        if use_rl:
            action, _ = model.predict(obs, deterministic=True)
            step_action = int(action)
            band_state = "rl-policy"
            manual_modes = None
        else:
            manual_modes, band_state = build_manual_modes(
                env,
                float(grid_limit_pct),
                float(soft_margin),
                allow_fast,
                allow_slow,
                allow_wait,
            )
            step_action = {"manual_modes": manual_modes.tolist()}
            action = -1

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(step_action)
        st.session_state.obs  = next_obs
        st.session_state.done = terminated or truncated
        st.session_state.step += 1
        st.session_state.rewards.append(reward)
        st.session_state.grids.append(info["grid_pct"])
        st.session_state.last_info = info  # Save for post-loop access
        st.session_state.last_band = band_state

        if "mode_assignments" in info:
            display_modes = np.asarray(info["mode_assignments"], dtype=np.int8)
        elif manual_modes is not None:
            display_modes = manual_modes
        else:
            display_modes = np.full(NUM_CARS, ACTION_WAIT, dtype=np.int8)

        step = st.session_state.step

        # ── Update Metrics ───────────────────────────────────────────
        metrics_placeholder["step"].markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{step} / {MAX_STEPS}</div>
            <div class="metric-lbl">⏱ Step (Minutes)</div>
        </div>""", unsafe_allow_html=True)

        metrics_placeholder["success"].markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:#4CAF50">{info['success_count']} / {NUM_CARS}</div>
            <div class="metric-lbl">✅ Cars Charged ≥{TARGET_BATTERY:.0f}%</div>
        </div>""", unsafe_allow_html=True)

        limit_now = float(info.get("grid_limit_pct", grid_limit_pct))
        soft_now = float(info.get("soft_band_start_pct", max(0.0, limit_now - soft_margin)))
        target_now = float(info.get("target_load_pct", limit_now * aggressiveness))
        grid_color = "#F44336" if info["grid_pct"] > limit_now else "#4A90E2"
        metrics_placeholder["grid"].markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:{grid_color}">{info['grid_pct']:.1f}%</div>
            <div class="metric-lbl">⚡ Grid (Soft {soft_now:.0f}% | Limit {limit_now:.0f}%)</div>
        </div>""", unsafe_allow_html=True)

        metrics_placeholder["battery"].markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:#FFC107">{info['avg_battery']:.1f}%</div>
            <div class="metric-lbl">🔋 Avg Active Battery</div>
        </div>""", unsafe_allow_html=True)

        # ── Update Grid Chart ─────────────────────────────────────────
        with grid_chart:
            fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0F1117")
            ax.set_facecolor("#1A1D27")
            grids = st.session_state.grids
            x = range(len(grids))
            ax.plot(x, grids, color="#4A90E2", linewidth=2, zorder=3)
            ax.fill_between(x, grids, alpha=0.2, color="#4A90E2")
            ax.axhline(soft_now, color="#FFC107", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axhline(limit_now, color="#FF4444", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.axhline(target_now, color="#9C27B0", linestyle=":", linewidth=1.0, alpha=0.8)
            ax.axhline(100, color="#FF4444", linestyle="-", linewidth=0.5, alpha=0.4)
            ax.fill_between(x, limit_now, [min(g, 110) for g in grids],
                            where=[g > limit_now for g in grids],
                            color="#FF4444", alpha=0.25, label="Danger Zone")
            ax.set_xlim(0, MAX_STEPS)
            y_max = max(110, int(target_now) + 10, int(max(grids)) + 5)
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Step (minutes)", color="#8892A4", fontsize=8)
            ax.set_ylabel("Grid Load %", color="#8892A4", fontsize=8)
            ax.tick_params(colors="#8892A4", labelsize=7)
            ax.spines[:].set_color("#2A2D3A")
            ax.text(5, min(limit_now + 2, 102), f"⚠ Danger Zone ({limit_now:.0f}%)", color="#FF4444", fontsize=7, alpha=0.9)
            plt.tight_layout(pad=0.5)
            grid_chart.pyplot(fig, clear_figure=True)
            plt.close(fig)

        # ── Update Car Grid ───────────────────────────────────────────
        # Build HTML for 50 car cards
        cars_html = '<div style="display:flex; flex-wrap:wrap; gap:3px;">'
        for i in range(NUM_CARS):
            batt     = env.battery[i]
            need_pct = max(TARGET_BATTERY - env.battery[i], 0)
            done_car = env.cars_done[i]
            mode     = int(display_modes[i])
            rate     = mode_to_rate(mode, env)
            label, badge_cls = mode_to_label(mode)
            cls      = car_class(mode, done_car)
            bc       = battery_color(batt)

            cars_html += f"""
            <div class="{cls}" style="width:88px;">
                <div style="color:#aaa; font-size:0.65rem;">🚗 Car {i+1:02d}</div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2px;">
                    <span style="color:white; font-weight:600; font-size:0.8rem;">{batt:.0f}%</span>
                    {"<span class='" + badge_cls + "'>" + label + "</span>" if not done_car else "<span style='color:#555; font-size:0.65rem'>GONE</span>"}
                </div>
                <div class="batt-bar" style="background:#222; width:100%;">
                    <div class="batt-bar" style="width:{min(batt,100):.0f}%; background:{bc};"></div>
                </div>
                <div style="color:#666; font-size:0.6rem; margin-top:2px;">🎯 Need {need_pct:.0f}%</div>
            </div>"""
        cars_html += "</div>"
        car_grid.markdown(cars_html, unsafe_allow_html=True)

        # ── Status ───────────────────────────────────────────────────
        selected_modes = []
        if allow_fast:
            selected_modes.append("FAST")
        if allow_slow:
            selected_modes.append("SLOW")
        if allow_wait:
            selected_modes.append("WAIT")

        if use_rl:
            agent_lbl = (
                f"RL Agent (Profile #{int(action)} | Fast {info.get('fast_cars', 0)} | "
                f"Slow {info.get('slow_cars', 0)} | Wait {info.get('wait_cars', 0)})"
            )
        else:
            agent_lbl = (
                f"Manual Controller ({band_state} | Fast {info.get('fast_cars', 0)} | "
                f"Slow {info.get('slow_cars', 0)} | Wait {info.get('wait_cars', 0)})"
            )
        departed  = int(np.sum(env.cars_done))
        status_bar.markdown(
            f"**Agent:** {agent_lbl} &nbsp;|&nbsp; "
            f"**Control Profile:** {control_profile} &nbsp;|&nbsp; "
            f"**Grid Limit:** {limit_now:.0f}% &nbsp;|&nbsp; "
            f"**Target:** {target_now:.0f}% &nbsp;|&nbsp; "
            f"**Allowed Modes:** {', '.join(selected_modes)} &nbsp;|&nbsp; "
            f"**Departed:** {departed}/{NUM_CARS} &nbsp;|&nbsp; "
            f"**Cumulative Reward:** {sum(st.session_state.rewards):.1f}"
        )

        time.sleep(speed)

    # ── Episode Complete ──────────────────────────────────────────────
    final_info = st.session_state.get("last_info")
    if final_info is not None:
        success = final_info["success_count"]
        rate    = success / NUM_CARS * 100
        total_r = sum(st.session_state.rewards)

        if rate >= 85:
            st.success(f"🏆 Episode Complete! **{success}/{NUM_CARS} cars ({rate:.1f}%)** successfully charged! Total Reward: {total_r:.1f}")
        elif rate >= 60:
            st.warning(f"⚠️ Episode Complete. **{success}/{NUM_CARS} cars ({rate:.1f}%)** charged. Room for improvement!")
        else:
            st.error(f"❌ Episode Complete. Only **{success}/{NUM_CARS} cars ({rate:.1f}%)** charged. Grid was stressed.")

    if st.button("🔄 Run Another Episode"):
        for key in ["env", "obs", "done", "step", "rewards", "grids", "running", "last_info"]:
            st.session_state.pop(key, None)
        st.rerun()


if __name__ == "__main__":
    main()
