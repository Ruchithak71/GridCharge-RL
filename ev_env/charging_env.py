"""
ev_env/charging_env.py
======================
The Heart of the Project — Our Custom Gymnasium Environment

What is a Gymnasium Environment?
---------------------------------
Think of it like a video game:
  - The "game world" is our charging station simulation
  - The "player" is the RL agent (PPO algorithm)
  - Each "frame" is one timestep (1 simulated minute)
  - The agent looks at the screen (state), presses a button (action),
    gets a score (reward), and learns what button combos win the game

Key Methods Every Gymnasium Env Must Have:
  - reset()  → Start a new episode (new set of 50 cars arrives)
  - step()   → Agent takes an action, world updates, we return reward
  - render() → (Optional) Print/visualize current state
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# ─────────────────────────────────────────────
#  CONSTANTS  (tweak these to tune difficulty)
# ─────────────────────────────────────────────
NUM_CARS          = 50       # Number of EVs plugged in simultaneously
MAX_STEPS         = 1200     # Safety cap for training runtime (cars no longer leave due time)
FAST_CHARGE_RATE  = 10.0     # % battery added per minute on fast charge (dashboard default)
SLOW_CHARGE_RATE  = 5.0      # % battery added per minute on slow charge (dashboard default)
WAIT_CHARGE_RATE  = 0.0      # No charging
TARGET_BATTERY    = 85.0     # Cars only depart once they reach at least 85%
GRID_CAPACITY     = 120.0    # Normalized max grid load (%) — slightly more headroom

# Power drawn from the grid per car (in arbitrary "grid units")
FAST_POWER  = 3.0
SLOW_POWER  = 1.0
WAIT_POWER  = 0.0

# Normalization/clip bounds for runtime controls.
MAX_FAST_RATE = 15.0
MAX_SLOW_RATE = 10.0
MAX_FAST_POWER = 6.0
MAX_SLOW_POWER = 3.0
MIN_CHARGE_RATE = 0.1
MIN_POWER = 0.1
MIN_SOFT_MARGIN = 1.0
MAX_SOFT_MARGIN = 15.0
MIN_AGGRESSIVENESS = 1.0
MAX_AGGRESSIVENESS = 1.15
OBS_GRID_MAX = 140.0

# Action constants — makes code readable
ACTION_WAIT  = 0
ACTION_SLOW  = 1
ACTION_FAST  = 2


class EVChargingEnv(gym.Env):
    """
    Custom Gymnasium Environment: Smart EV Charging Station

    Episode Flow:
    1. reset() is called → 50 cars arrive with random battery
    2. Agent observes state (112 numbers describing cars + grid + runtime controls)
    3. Agent picks an action (a charging profile for the station)
    4. step() applies the action, simulates 1 minute passing
    5. Reward is computed based on charging success and grid stress
    6. Repeat steps 2-5 for 180 minutes
    7. Episode ends when all cars are at or above target battery (or safety cap)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode=None,
        grid_limit_pct=100.0,
        randomize_grid_limit=False,
        grid_limit_range=(85.0, 100.0),
        fast_charge_rate=FAST_CHARGE_RATE,
        slow_charge_rate=SLOW_CHARGE_RATE,
        fast_power=FAST_POWER,
        slow_power=SLOW_POWER,
        aggressiveness=1.03,
        soft_margin=5.0,
        randomize_control_params=False,
        fast_rate_range=(FAST_CHARGE_RATE, FAST_CHARGE_RATE),
        slow_rate_range=(SLOW_CHARGE_RATE, SLOW_CHARGE_RATE),
        fast_power_range=(FAST_POWER, FAST_POWER),
        slow_power_range=(SLOW_POWER, SLOW_POWER),
        aggressiveness_range=(1.0, 1.0),
        include_mode_assignments=False,
        allowed_modes=(True, True, True),
    ):
        super().__init__()
        self.render_mode = render_mode

        # User-configurable target for "safe" grid usage.
        self.grid_limit_pct = float(np.clip(grid_limit_pct, 60.0, 120.0))
        self.randomize_grid_limit = bool(randomize_grid_limit)
        lo, hi = grid_limit_range
        self.grid_limit_range = (float(min(lo, hi)), float(max(lo, hi)))

        # Runtime control randomization for variable-conditioned training.
        self.randomize_control_params = bool(randomize_control_params)
        self.fast_rate_range = self._sorted_range(fast_rate_range)
        self.slow_rate_range = self._sorted_range(slow_rate_range)
        self.fast_power_range = self._sorted_range(fast_power_range)
        self.slow_power_range = self._sorted_range(slow_power_range)
        self.aggressiveness_range = self._sorted_range(aggressiveness_range)
        self.include_mode_assignments = bool(include_mode_assignments)

        # Runtime control values.
        self.fast_charge_rate = FAST_CHARGE_RATE
        self.slow_charge_rate = SLOW_CHARGE_RATE
        self.fast_power = FAST_POWER
        self.slow_power = SLOW_POWER
        self.aggressiveness = 1.03
        self.soft_margin = 5.0
        self.allow_fast = True
        self.allow_slow = True
        self.allow_wait = True

        self.set_charge_rates(fast_charge_rate, slow_charge_rate)
        self.set_power_draw(fast_power, slow_power)
        self.set_aggressiveness(aggressiveness)
        self.set_soft_margin(soft_margin)
        self.set_allowed_modes(*allowed_modes)

        # ── OBSERVATION SPACE ─────────────────────────────────────────
        # What the agent SEES at each step. A flat numpy array of 112 numbers:
        #   [0:50]  → battery_pct for each car (normalized 0→1)
        #   [50:100]→ remaining deficit-to-target for each car (normalized 0→1)
        #   [100]   → current total grid load % (normalized 0→1)
        #   [101]   → current hour of day (normalized 0→1, i.e., 0h=0.0, 24h=1.0)
        #   [102]   → user-selected grid limit % (normalized 0→1)
        #   [103]   → fast charge rate
        #   [104]   → slow charge rate
        #   [105]   → fast power draw
        #   [106]   → slow power draw
        #   [107]   → aggressiveness
        #   [108]   → soft-margin
        #   [109]   → fast enabled flag
        #   [110]   → slow enabled flag
        #   [111]   → wait enabled flag
        # All values are clipped to [0, 1] (gymnasium standard for Box spaces)
        obs_size = NUM_CARS * 2 + 12
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # ── ACTION SPACE ──────────────────────────────────────────────
        # We use 13 profiles that prioritize SLOW charging over WAIT.
        # Profile 0   → Everyone waits (emergency brake only)
        # Profile 1   → Everyone slow charges
        # Profiles 2-11: Top N cars fast (N=5..50), others slow
        # Profile 12  → Relief mode: top 15 fast, others wait
        self.num_profiles = 13  # 0 to 12
        self.action_space = spaces.Discrete(self.num_profiles)

        # Internal state (filled in reset())
        self.battery       = None   # shape: (NUM_CARS,)  battery % per car
        self.current_step  = 0
        self.sim_hour      = 18.0   # Simulation starts at 6 PM (peak demand)
        self.cars_done     = None   # Track which cars have already departed
        self.success_count = 0      # Cars that reached target battery
        self.last_grid_pct = 0.0    # Latest total grid load %, fed into next observation

    def set_grid_limit(self, grid_limit_pct):
        """Update the runtime grid safety threshold from UI or evaluation config."""
        self.grid_limit_pct = float(np.clip(grid_limit_pct, 60.0, 120.0))

    def set_charge_rates(self, fast_rate, slow_rate):
        """Set runtime fast/slow charge rates (percent battery per step)."""
        fast = float(np.clip(fast_rate, MIN_CHARGE_RATE, MAX_FAST_RATE))
        slow = float(np.clip(slow_rate, MIN_CHARGE_RATE, MAX_SLOW_RATE))
        if slow > fast:
            slow = fast
        self.fast_charge_rate = fast
        self.slow_charge_rate = slow

    def set_power_draw(self, fast_power, slow_power):
        """Set runtime fast/slow power draw used for grid-load computation."""
        fast = float(np.clip(fast_power, MIN_POWER, MAX_FAST_POWER))
        slow = float(np.clip(slow_power, MIN_POWER, MAX_SLOW_POWER))
        if slow > fast:
            slow = fast
        self.fast_power = fast
        self.slow_power = slow

    def set_aggressiveness(self, aggressiveness):
        """Set load target multiplier (1.0 means target equals slider limit)."""
        self.aggressiveness = float(np.clip(aggressiveness, MIN_AGGRESSIVENESS, MAX_AGGRESSIVENESS))

    def set_soft_margin(self, soft_margin):
        """Set the pre-limit band size where strategy shifts fast → slow."""
        self.soft_margin = float(np.clip(soft_margin, MIN_SOFT_MARGIN, MAX_SOFT_MARGIN))

    def set_allowed_modes(self, fast_enabled=True, slow_enabled=True, wait_enabled=True):
        """Enable/disable FAST/SLOW/WAIT. At least one mode must remain enabled."""
        allow_fast = bool(fast_enabled)
        allow_slow = bool(slow_enabled)
        allow_wait = bool(wait_enabled)
        if not (allow_fast or allow_slow or allow_wait):
            raise ValueError("At least one charging mode must be enabled.")
        self.allow_fast = allow_fast
        self.allow_slow = allow_slow
        self.allow_wait = allow_wait

    @staticmethod
    def _sorted_range(raw_range):
        """Return a numeric range as (min, max)."""
        lo, hi = raw_range
        return float(min(lo, hi)), float(max(lo, hi))

    def _resolve_mode(self, preferred_mode):
        """Resolve a preferred mode against enabled-mode constraints."""
        if preferred_mode == ACTION_FAST:
            order = (ACTION_FAST, ACTION_SLOW, ACTION_WAIT)
        elif preferred_mode == ACTION_SLOW:
            order = (ACTION_SLOW, ACTION_FAST, ACTION_WAIT)
        else:
            order = (ACTION_WAIT, ACTION_SLOW, ACTION_FAST)

        for mode in order:
            if mode == ACTION_FAST and self.allow_fast:
                return ACTION_FAST
            if mode == ACTION_SLOW and self.allow_slow:
                return ACTION_SLOW
            if mode == ACTION_WAIT and self.allow_wait:
                return ACTION_WAIT

        return ACTION_WAIT

    def _build_profile_modes(self, action, active_mask):
        """Translate discrete profile action into per-car mode assignments."""
        modes = np.full(NUM_CARS, ACTION_WAIT, dtype=np.int8)

        if action == 0:
            modes[active_mask] = self._resolve_mode(ACTION_WAIT)
            return modes

        profile_map = {
            1: (0, 1),
            2: (5, 1),   3: (10, 1),
            4: (15, 1),  5: (20, 1),
            6: (25, 1),  7: (30, 1),
            8: (35, 1),  9: (40, 1),
            10: (45, 1), 11: (50, 1),
            12: (15, 0),
        }
        n_fast_cars, rest_mode = profile_map.get(int(action), (0, 1))
        rest_preferred = ACTION_SLOW if rest_mode == 1 else ACTION_WAIT

        deficit = np.maximum(TARGET_BATTERY - self.battery, 0.0)
        urgency = np.where(active_mask, deficit, -999.0)
        sorted_idx = np.argsort(-urgency)
        fast_set = set(sorted_idx[:n_fast_cars])

        for i in np.where(active_mask)[0]:
            preferred = ACTION_FAST if i in fast_set else rest_preferred
            modes[i] = self._resolve_mode(preferred)

        return modes

    def _build_manual_modes(self, manual_modes, active_mask):
        """Convert dashboard-provided per-car mode payload into safe assignments."""
        modes = np.full(NUM_CARS, ACTION_WAIT, dtype=np.int8)
        default_mode = self._resolve_mode(ACTION_SLOW)
        modes[active_mask] = default_mode

        if manual_modes is None:
            return modes

        arr = np.asarray(manual_modes, dtype=np.int8).reshape(-1)
        if arr.size != NUM_CARS:
            return modes

        for i in np.where(active_mask)[0]:
            preferred = int(np.clip(arr[i], ACTION_WAIT, ACTION_FAST))
            modes[i] = self._resolve_mode(preferred)

        return modes

    def _modes_to_rates(self, modes):
        """Map per-car FAST/SLOW/WAIT modes to charge-rate values."""
        rates = np.zeros(NUM_CARS, dtype=np.float32)
        rates[modes == ACTION_FAST] = self.fast_charge_rate
        rates[modes == ACTION_SLOW] = self.slow_charge_rate
        return rates

    def _mode_power(self, mode):
        if mode == ACTION_FAST:
            return self.fast_power
        if mode == ACTION_SLOW:
            return self.slow_power
        return WAIT_POWER

    # ──────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        """
        Start a fresh episode.
        Called at the beginning of every training rollout.
        Returns: (observation, info_dict)
        """
        super().reset(seed=seed)

        # Randomize 50 car batteries: each car arrives with 20–60% charge
        self.battery = self.np_random.uniform(low=25.0, high=65.0, size=NUM_CARS).astype(np.float32)

        self.current_step  = 0
        self.sim_hour      = 18.0   # Always start at 6 PM
        self.cars_done     = np.zeros(NUM_CARS, dtype=bool)
        self.success_count = 0

        if self.randomize_grid_limit:
            lo, hi = self.grid_limit_range
            self.grid_limit_pct = float(self.np_random.uniform(low=lo, high=hi))
        else:
            self.grid_limit_pct = float(np.clip(self.grid_limit_pct, 60.0, 120.0))

        if self.randomize_control_params:
            self.set_charge_rates(
                self.np_random.uniform(*self.fast_rate_range),
                self.np_random.uniform(*self.slow_rate_range),
            )
            self.set_power_draw(
                self.np_random.uniform(*self.fast_power_range),
                self.np_random.uniform(*self.slow_power_range),
            )
            self.set_aggressiveness(
                self.np_random.uniform(*self.aggressiveness_range)
            )

        # Initial observation reflects base demand before any EV action is taken.
        self.last_grid_pct = (self._grid_base_load(self.sim_hour) / GRID_CAPACITY) * 100.0

        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────────
    def step(self, action):
        """
        Apply the agent's chosen action for 1 simulated minute.

        Args:
            action (int): Index 0-12 → charging profile

        Returns:
            obs        : New state after this step
            reward     : Scalar score for this action
            terminated : True if episode is naturally done (all cars departed)
            truncated  : True if we hit MAX_STEPS time limit
            info       : Dictionary with extra metrics
        """
        # ── 1. TRANSLATE ACTION → PER-CAR MODES ───────────────────────
        active_mask = ~self.cars_done

        if isinstance(action, dict):
            modes = self._build_manual_modes(action.get("manual_modes"), active_mask)
            action_kind = "manual"
        else:
            try:
                action_idx = int(action)
            except (TypeError, ValueError):
                action_idx = 0
            action_idx = int(np.clip(action_idx, 0, self.num_profiles - 1))
            modes = self._build_profile_modes(action_idx, active_mask)
            action_kind = action_idx

        modes[~active_mask] = ACTION_WAIT
        charge_rates = self._modes_to_rates(modes)

        # ── 2. GRID LOAD CALCULATION ─────────────────────────────────
        fast_count = int(np.sum((modes == ACTION_FAST) & active_mask))
        slow_count = int(np.sum((modes == ACTION_SLOW) & active_mask))
        wait_count = int(np.sum((modes == ACTION_WAIT) & active_mask))

        base_load    = self._grid_base_load(self.sim_hour)
        ev_load      = fast_count * self.fast_power + slow_count * self.slow_power
        
        total_load   = base_load + ev_load
        grid_pct     = total_load / GRID_CAPACITY * 100.0  # Can exceed 100!

        # ── 3. UPDATE CAR BATTERIES ───────────────────────────────────
        # KEY MECHANIC: Grid overload reduces charging efficiency!
        # Above 85% grid load, charging becomes progressively less effective
        # (simulating voltage drops, brownouts, circuit breaker limits)
        threshold = self.grid_limit_pct
        if grid_pct > threshold:
            # Above the user-selected threshold, charging efficiency drops quickly.
            overload_factor = max(0.15, 1.0 - 0.85 * ((grid_pct - threshold) / 30.0))
            charge_rates[modes != ACTION_WAIT] = charge_rates[modes != ACTION_WAIT] * overload_factor

        self.battery = np.clip(self.battery + charge_rates, 0.0, 100.0)

        # ── 4. ADVANCE TIME ───────────────────────────────────────────
        self.current_step += 1
        self.sim_hour      = (18.0 + self.current_step / 60.0) % 24.0

        # ── 5. CHECK DEPARTURES (only after reaching target) ──────────
        step_reward     = 0.0
        newly_departed  = 0

        # Count cars charged at target battery (live count, not just at departure)
        self.success_count = int(np.sum(self.battery >= TARGET_BATTERY))

        for i in range(NUM_CARS):
            if self.cars_done[i]:
                continue

            reached_target = self.battery[i] >= TARGET_BATTERY

            if reached_target:
                self.cars_done[i] = True
                newly_departed    += 1
                step_reward += 22.0  # ✅ Car leaves only when target is satisfied.
                overshoot = min((self.battery[i] - TARGET_BATTERY) / 15.0, 1.0)
                step_reward += 4.0 * overshoot
                if self.battery[i] >= 100.0:
                    step_reward += 6.0

        # ── 6. COMPUTE REWARD ─────────────────────────────────────────
        # 6a. Per-step shaping: reward battery PROGRESS on active cars
        #     This gives the agent immediate feedback for charging
        active_now = ~self.cars_done
        active_count = int(np.sum(active_now))
        if active_count > 0:
            # Reward progress towards target
            avg_progress = float(np.mean(
                np.minimum(self.battery[active_now], TARGET_BATTERY)
            )) / TARGET_BATTERY
            step_reward += 2.0 * avg_progress  # continuous shaping (stronger signal)

            # Bonus for fraction of active cars already above target
            above_target = np.sum(self.battery[active_now] >= TARGET_BATTERY)
            step_reward += 1.0 * (above_target / max(active_count, 1))

            # Deficit-aware bonus: prioritize cars with large remaining deficit.
            deficit_now = np.maximum(TARGET_BATTERY - self.battery[active_now], 0.0)
            high_deficit_mask = deficit_now > 20.0
            if np.any(high_deficit_mask):
                high_deficit_recovered = np.sum(
                    self.battery[active_now][high_deficit_mask] >= (TARGET_BATTERY - 5.0)
                )
                step_reward += 2.0 * (high_deficit_recovered / max(np.sum(high_deficit_mask), 1))

            # Discourage idle waiting: prefer SLOW over WAIT for active cars.
            waiting_active = int(np.sum((modes == ACTION_WAIT) & active_now & (self.battery < TARGET_BATTERY)))
            step_reward -= 0.25 * float(waiting_active)

        # 6b. Grid reward around user threshold with soft/target bands.
        soft_band_start = max(0.0, threshold - self.soft_margin)
        target_load = threshold * self.aggressiveness

        if grid_pct > target_load:
            overflow = grid_pct - target_load
            step_reward -= 1.0 * overflow
            step_reward -= 0.10 * (overflow ** 2)
        elif grid_pct > threshold:
            mild_over = grid_pct - threshold
            step_reward -= 0.25 * mild_over
            step_reward += 0.6 * (grid_pct / max(target_load, 1.0))
        else:
            utilization = grid_pct / max(threshold, 1.0)
            band_center = max(soft_band_start, 1.0)
            proximity = max(0.0, 1.0 - abs(grid_pct - band_center) / band_center)
            step_reward += 1.0 * utilization
            step_reward += 1.0 * proximity

        # Extra hard penalty once absolute grid load crosses 100%.
        if grid_pct > 100.0:
            critical_overflow = grid_pct - 100.0
            step_reward -= 0.25 * (critical_overflow ** 2)

        # 6c. Small power efficiency cost
        step_reward -= 0.0005 * (fast_count * self.fast_power + slow_count * self.slow_power)

        # ── 7. TERMINATION CONDITIONS ─────────────────────────────────
        terminated = bool(np.all(self.cars_done))
        truncated  = (self.current_step >= MAX_STEPS)

        if truncated and not terminated:
            remaining = int(np.sum(~self.cars_done))
            step_reward -= 75.0 * remaining

        # End-of-episode bonus — strongly incentivize high success rates
        if (terminated or truncated):
            success_rate = self.success_count / NUM_CARS
            # Smooth, continuous end-of-episode bonus (easier gradient)
            # Quadratic scaling: ramps up aggressively at high success rates
            step_reward += 400.0 * (success_rate ** 2)
            if success_rate >= 1.0:
                step_reward += 200.0   # 🏆 Perfect episode jackpot!
            elif success_rate < 0.5:
                step_reward -= 50.0 * (1.0 - success_rate)  # Penalize truly bad episodes

        # Store the latest total load so the next observation includes full live grid state.
        self.last_grid_pct = float(grid_pct)

        # ── 8. EXTRA INFO (for logging/analysis) ──────────────────────
        # Count cars currently at ≥ 80% (includes both active and departed)
        cars_charged = int(np.sum(self.battery >= TARGET_BATTERY))

        info = {
            "success_count" : cars_charged,
            "total_departed": int(np.sum(self.cars_done)),
            "grid_pct"      : round(float(grid_pct), 2),
            "grid_limit_pct": round(float(self.grid_limit_pct), 2),
            "grid_over_limit": round(max(0.0, float(grid_pct - self.grid_limit_pct)), 2),
            "target_load_pct": round(float(target_load), 2),
            "soft_band_start_pct": round(float(soft_band_start), 2),
            "fast_cars"     : fast_count,
            "slow_cars"     : slow_count,
            "wait_cars"     : wait_count,
            "cars_remaining": int(np.sum(~self.cars_done)),
            "avg_battery"   : round(float(np.mean(self.battery[~self.cars_done])) if np.any(~self.cars_done) else 100.0, 2),
            "action_kind"   : action_kind,
            "fast_rate"     : round(float(self.fast_charge_rate), 3),
            "slow_rate"     : round(float(self.slow_charge_rate), 3),
            "fast_power"    : round(float(self.fast_power), 3),
            "slow_power"    : round(float(self.slow_power), 3),
            "aggressiveness": round(float(self.aggressiveness), 3),
            "soft_margin"   : round(float(self.soft_margin), 3),
            "allow_fast"    : bool(self.allow_fast),
            "allow_slow"    : bool(self.allow_slow),
            "allow_wait"    : bool(self.allow_wait),
        }

        if self.include_mode_assignments:
            info["mode_assignments"] = modes.astype(np.int8).tolist()

        return self._get_obs(), float(step_reward), terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────
    def _get_obs(self):
        """
        Build the observation vector (what the agent sees).
        All values normalized to [0, 1] for stable neural network training.
        """
        battery_norm    = self.battery / 100.0
        deficit_norm    = np.maximum(TARGET_BATTERY - self.battery, 0.0) / TARGET_BATTERY
        grid_load_norm  = np.array([np.clip(self.last_grid_pct, 0.0, OBS_GRID_MAX) / OBS_GRID_MAX], dtype=np.float32)
        hour_norm       = np.array([self.sim_hour / 24.0], dtype=np.float32)
        grid_limit_norm = np.array([self.grid_limit_pct / GRID_CAPACITY], dtype=np.float32)
        fast_rate_norm  = np.array([self.fast_charge_rate / MAX_FAST_RATE], dtype=np.float32)
        slow_rate_norm  = np.array([self.slow_charge_rate / MAX_SLOW_RATE], dtype=np.float32)
        fast_power_norm = np.array([self.fast_power / MAX_FAST_POWER], dtype=np.float32)
        slow_power_norm = np.array([self.slow_power / MAX_SLOW_POWER], dtype=np.float32)
        aggr_norm       = np.array([
            np.clip((self.aggressiveness - MIN_AGGRESSIVENESS) / (MAX_AGGRESSIVENESS - MIN_AGGRESSIVENESS), 0.0, 1.0)
        ], dtype=np.float32)
        margin_norm     = np.array([
            np.clip((self.soft_margin - MIN_SOFT_MARGIN) / (MAX_SOFT_MARGIN - MIN_SOFT_MARGIN), 0.0, 1.0)
        ], dtype=np.float32)
        allow_fast_norm = np.array([1.0 if self.allow_fast else 0.0], dtype=np.float32)
        allow_slow_norm = np.array([1.0 if self.allow_slow else 0.0], dtype=np.float32)
        allow_wait_norm = np.array([1.0 if self.allow_wait else 0.0], dtype=np.float32)

        obs = np.concatenate([
            battery_norm,
            deficit_norm,
            grid_load_norm,
            hour_norm,
            grid_limit_norm,
            fast_rate_norm,
            slow_rate_norm,
            fast_power_norm,
            slow_power_norm,
            aggr_norm,
            margin_norm,
            allow_fast_norm,
            allow_slow_norm,
            allow_wait_norm,
        ]).astype(np.float32)
        return obs

    # ──────────────────────────────────────────────────────────────────
    def _grid_base_load(self, hour):
        """
        Simulate realistic electricity demand curve (no EVs).
        Peak at 6-8 PM (dinner + AC + TVs), low at 3 AM.

        This is based on real utility grid data patterns.
        Returns a value in [0, GRID_CAPACITY].
        """
        # Gaussian peaks model:
        # Evening peak around 19:00h
        evening_peak = 45.0 * np.exp(-0.5 * ((hour - 19.0) / 2.5) ** 2)
        # Morning peak around 8:00h
        morning_peak = 25.0 * np.exp(-0.5 * ((hour - 8.0) / 1.5) ** 2)
        # Overnight base load
        base         = 20.0

        return float(base + evening_peak + morning_peak)

    # ──────────────────────────────────────────────────────────────────
    def render(self):
        """Simple text rendering for debugging."""
        if self.render_mode == "human":
            active = ~self.cars_done
            print(
                f"Step {self.current_step:3d} | "
                f"Hour: {self.sim_hour:.1f}h | "
                f"Grid: {self.last_grid_pct:.1f}% (limit {self.grid_limit_pct:.1f}%) | "
                f"Active Cars: {int(np.sum(active))} | "
                f"Avg Battery: {np.mean(self.battery[active]):.1f}% | "
                f"Success: {self.success_count} | "
                f"Rates(F/S): {self.fast_charge_rate:.1f}/{self.slow_charge_rate:.1f}"
            )
