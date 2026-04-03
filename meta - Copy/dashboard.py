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
  - Agent action decided each step
  - Metrics: success rate, avg battery, grid stress
"""

import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import PPO
from ev_env.charging_env import EVChargingEnv, NUM_CARS, TARGET_BATTERY, MAX_STEPS

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
    try:
        model = PPO.load("models/ev_ppo_agent")
        return model, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
#  HELPER: Battery color
# ─────────────────────────────────────────────
def battery_color(pct):
    if pct >= 80:   return "#4CAF50"   # green
    elif pct >= 50: return "#FFC107"   # amber
    elif pct >= 30: return "#FF9800"   # orange
    else:           return "#F44336"   # red


def action_to_label(charge_rate):
    if charge_rate >= 1.0:    return "FAST", "badge-fast"
    elif charge_rate > 0.01:  return "SLOW", "badge-slow"
    else:                     return "WAIT", "badge-wait"


def car_class(charge_rate, done):
    if done:               return "car-card car-done"
    if charge_rate >= 1.0:  return "car-card car-fast"
    if charge_rate > 0.01:  return "car-card car-slow"
    return "car-card car-wait"


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────────────────────────────
    st.markdown('<div class="main-title">⚡ Grid-Aware EV Charging Orchestrator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Reinforcement Learning Agent Managing 50 Electric Vehicles in Real-Time</div>', unsafe_allow_html=True)

    # ── Sidebar Controls ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        speed = st.slider("Simulation Speed", 0.05, 1.0, 0.15, 0.05,
                          help="Seconds per step (lower = faster)")
        use_rl = st.toggle("Use RL Agent", value=True,
                           help="Toggle to compare with naive baseline")
        st.markdown("---")
        st.markdown("### 🎨 Legend")
        st.markdown("""
        <span class="badge-fast">FAST</span> High priority — 15%/min<br><br>
        <span class="badge-slow">SLOW</span> Low priority — 5%/min<br><br>
        <span class="badge-wait">WAIT</span> Grid relief — 0%/min
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📖 How It Works")
        st.markdown("""
        The PPO agent observes:
        - 🔋 Battery % of each car
        - ⏰ Time until departure
        - ⚡ Current grid load

        It picks a **charging profile** that maximizes the number of cars charged before their owner returns.
        """)

        run_btn = st.button("▶ Run New Episode")

    # ── Load Model ───────────────────────────────────────────────────
    model, err = load_model()
    if model is None:
        st.error(f"❌ Could not load model: {err}")
        st.info("👉 Run `python train.py` first, then restart this dashboard.")
        st.code("python train.py", language="bash")
        return

    # ── Session State ────────────────────────────────────────────────
    if "env" not in st.session_state or run_btn:
        st.session_state.env       = EVChargingEnv(render_mode=None)
        obs, _                     = st.session_state.env.reset()
        st.session_state.obs       = obs
        st.session_state.done      = False
        st.session_state.step      = 0
        st.session_state.rewards   = []
        st.session_state.grids     = []
        st.session_state.running   = True
        st.session_state.last_info = None  # Store final episode info
        # Rerun to show fresh state
        st.rerun()

    env = st.session_state.env

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
        else:
            action = 12  # Baseline: all fast

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(int(action))
        st.session_state.obs  = next_obs
        st.session_state.done = terminated or truncated
        st.session_state.step += 1
        st.session_state.rewards.append(reward)
        st.session_state.grids.append(info["grid_pct"])
        st.session_state.last_info = info  # Save for post-loop access

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
            <div class="metric-lbl">✅ Cars Charged ≥80%</div>
        </div>""", unsafe_allow_html=True)

        grid_color = "#F44336" if info["grid_pct"] > 85 else "#4A90E2"
        metrics_placeholder["grid"].markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:{grid_color}">{info['grid_pct']:.1f}%</div>
            <div class="metric-lbl">⚡ Grid Load</div>
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
            ax.axhline(85, color="#FF4444", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.axhline(100, color="#FF4444", linestyle="-", linewidth=0.5, alpha=0.4)
            ax.fill_between(x, 85, [min(g, 100) for g in grids],
                            where=[g > 85 for g in grids],
                            color="#FF4444", alpha=0.25, label="Danger Zone")
            ax.set_xlim(0, MAX_STEPS)
            ax.set_ylim(0, 105)
            ax.set_xlabel("Step (minutes)", color="#8892A4", fontsize=8)
            ax.set_ylabel("Grid Load %", color="#8892A4", fontsize=8)
            ax.tick_params(colors="#8892A4", labelsize=7)
            ax.spines[:].set_color("#2A2D3A")
            ax.text(5, 87, "⚠ Danger Zone (85%)", color="#FF4444", fontsize=7, alpha=0.9)
            plt.tight_layout(pad=0.5)
            grid_chart.pyplot(fig, clear_figure=True)
            plt.close(fig)

        # ── Update Car Grid ───────────────────────────────────────────
        # Recompute charge rates from action (mirroring env logic)
        charge_rates = np.zeros(NUM_CARS)
        if action > 0:
            profile_map = {
                1: (5, 0),   2: (5, 1),
                3: (10, 0),  4: (10, 1),
                5: (15, 0),  6: (15, 1),
                7: (20, 0),  8: (20, 1),
                9: (25, 0),  10: (25, 1),
                11: (30, 1), 12: (50, 1),
            }
            n_fast, rest_mode = profile_map.get(int(action), (5, 1))
            rest_rate = 0.4 if rest_mode == 1 else 0.0
            time_remaining = np.maximum(env.departure - env.current_step, 1.0)
            deficit        = np.maximum(TARGET_BATTERY - env.battery, 0.0)
            urgency        = np.where(~env.cars_done, deficit / time_remaining, -999.0)
            sorted_idx     = np.argsort(-urgency)
            fast_set       = set(sorted_idx[:n_fast])
            for i in range(NUM_CARS):
                if not env.cars_done[i]:
                    charge_rates[i] = 1.0 if i in fast_set else rest_rate

        # Build HTML for 50 car cards
        cars_html = '<div style="display:flex; flex-wrap:wrap; gap:3px;">'
        for i in range(NUM_CARS):
            batt     = env.battery[i]
            dep_min  = max(env.departure[i] - env.current_step, 0)
            done_car = env.cars_done[i]
            rate     = charge_rates[i]
            label, badge_cls = action_to_label(rate)
            cls      = car_class(rate, done_car)
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
                <div style="color:#666; font-size:0.6rem; margin-top:2px;">⏰ {dep_min:.0f}min</div>
            </div>"""
        cars_html += "</div>"
        car_grid.markdown(cars_html, unsafe_allow_html=True)

        # ── Status ───────────────────────────────────────────────────
        agent_lbl = f"RL Agent (Profile #{action}: {int(action)*5} fast)" if use_rl else "Baseline (Full Speed)"
        departed  = int(np.sum(env.cars_done))
        status_bar.markdown(
            f"**Agent:** {agent_lbl} &nbsp;|&nbsp; "
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
