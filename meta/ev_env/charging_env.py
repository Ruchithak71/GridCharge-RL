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
MAX_STEPS         = 180      # Episode length = 180 simulated minutes (3 hours)
FAST_CHARGE_RATE  = 1.5      # % battery added per minute on fast charge (~90%/hr)
SLOW_CHARGE_RATE  = 0.5      # % battery added per minute on slow charge (~30%/hr)
WAIT_CHARGE_RATE  = 0.0      # No charging
TARGET_BATTERY    = 80.0     # We want every car at ≥ 80% before departure
GRID_CAPACITY     = 120.0    # Normalized max grid load (%) — slightly more headroom

# Power drawn from the grid per car (in arbitrary "grid units")
FAST_POWER  = 3.0
SLOW_POWER  = 1.0
WAIT_POWER  = 0.0

# Action constants — makes code readable
ACTION_WAIT  = 0
ACTION_SLOW  = 1
ACTION_FAST  = 2


class EVChargingEnv(gym.Env):
    """
    Custom Gymnasium Environment: Smart EV Charging Station

    Episode Flow:
    1. reset() is called → 50 cars arrive with random battery & departure time
    2. Agent observes state (102 numbers describing all cars + grid)
    3. Agent picks an action (a charging profile for the station)
    4. step() applies the action, simulates 1 minute passing
    5. Reward is computed based on charging success and grid stress
    6. Repeat steps 2-5 for 180 minutes
    7. Episode ends → count how many cars left with ≥ 80% battery
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ── OBSERVATION SPACE ─────────────────────────────────────────
        # What the agent SEES at each step. A flat numpy array of 102 numbers:
        #   [0:50]  → battery_pct for each car (normalized 0→1)
        #   [50:100]→ minutes_until_departure for each car (normalized 0→1)
        #   [100]   → current grid load (normalized 0→1)
        #   [101]   → current hour of day (normalized 0→1, i.e., 0h=0.0, 24h=1.0)
        # All values are clipped to [0, 1] (gymnasium standard for Box spaces)
        obs_size = NUM_CARS * 2 + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # ── ACTION SPACE ──────────────────────────────────────────────
        # Instead of 3^50 combinations (impossible!), we use a SMART ENCODING:
        # The agent picks ONE integer 0–8, representing a "charging policy profile"
        # Each profile sets a threshold: "charge fast if urgency_score > threshold"
        #
        # Urgency Score = (TARGET_BATTERY - battery_pct) / max(minutes_left, 1)
        # Higher score = car needs more charge per minute to make it in time
        #
        # Profile 0  → Everyone waits (grid emergency)
        # Profile 1  → Top 5 cars get fast charge, rest WAIT
        # Profile 2  → Top 5 fast, rest SLOW
        # Profile 3  → Top 10 fast, rest WAIT
        # Profile 4  → Top 10 fast, rest SLOW
        # ...
        # Profile 11 → Top 25 fast, rest SLOW (aggressive)
        # Profile 12 → ALL 50 fast charge (max aggression)
        self.num_profiles = 13  # 0 to 12
        self.action_space = spaces.Discrete(self.num_profiles)

        # Internal state (filled in reset())
        self.battery       = None   # shape: (NUM_CARS,)  battery % per car
        self.departure     = None   # shape: (NUM_CARS,)  minutes until departure
        self.current_step  = 0
        self.sim_hour      = 18.0   # Simulation starts at 6 PM (peak demand)
        self.cars_done     = None   # Track which cars have already departed
        self.success_count = 0      # Cars that left with ≥ 80%

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

        # Randomize departure times: cars leave 45–180 minutes from now
        # (min 45 min so every car is physically chargeable to 80%)
        self.departure = self.np_random.uniform(low=45.0, high=MAX_STEPS, size=NUM_CARS).astype(np.float32)

        self.current_step  = 0
        self.sim_hour      = 18.0   # Always start at 6 PM
        self.cars_done     = np.zeros(NUM_CARS, dtype=bool)
        self.success_count = 0

        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────────
    def step(self, action):
        """
        Apply the agent's chosen action for 1 simulated minute.

        Args:
            action (int): Index 0-8 → charging profile

        Returns:
            obs        : New state after this step
            reward     : Scalar score for this action
            terminated : True if episode is naturally done (all cars departed)
            truncated  : True if we hit MAX_STEPS time limit
            info       : Dictionary with extra metrics
        """
        # ── 1. TRANSLATE ACTION → CAR ASSIGNMENTS ────────────────────
        #    Compute urgency for each car that hasn't departed yet
        active_mask = ~self.cars_done
        charge_rates = np.zeros(NUM_CARS, dtype=np.float32)

        if action == 0:
            # Profile 0: All cars WAIT (grid emergency brake)
            pass
        else:
            # Decode action → (n_fast_cars, rest_mode)
            # Odd actions: rest=WAIT, Even actions: rest=SLOW
            # action 1→(5,WAIT) 2→(5,SLOW) 3→(10,WAIT) 4→(10,SLOW)
            # 5→(15,WAIT) 6→(15,SLOW) 7→(20,WAIT) 8→(20,SLOW)
            # 9→(25,WAIT) 10→(25,SLOW) 11→(30,SLOW) 12→(50,FAST=all)
            profile_map = {
                1: (5, 0),   2: (5, 1),
                3: (10, 0),  4: (10, 1),
                5: (15, 0),  6: (15, 1),
                7: (20, 0),  8: (20, 1),
                9: (25, 0),  10: (25, 1),
                11: (30, 1), 12: (50, 1),
            }
            n_fast_cars, rest_mode = profile_map[int(action)]
            rest_rate = SLOW_CHARGE_RATE if rest_mode == 1 else WAIT_CHARGE_RATE

            # Urgency = deficit per minute remaining
            time_remaining = np.maximum(self.departure - self.current_step, 1.0)
            deficit        = np.maximum(TARGET_BATTERY - self.battery, 0.0)
            urgency        = deficit / time_remaining

            urgency_masked = np.where(active_mask, urgency, -999.0)
            sorted_idx = np.argsort(-urgency_masked)

            fast_set = set(sorted_idx[:n_fast_cars])
            for i in range(NUM_CARS):
                if self.cars_done[i]:
                    continue
                if i in fast_set:
                    charge_rates[i] = FAST_CHARGE_RATE
                else:
                    charge_rates[i] = rest_rate

        # ── 2. GRID LOAD CALCULATION ─────────────────────────────────
        # Calculate counts BEFORE they get scaled down by overload_factor
        fast_count = int(np.sum(charge_rates == FAST_CHARGE_RATE))
        slow_count = int(np.sum(charge_rates == SLOW_CHARGE_RATE))

        base_load    = self._grid_base_load(self.sim_hour)
        ev_load      = fast_count * FAST_POWER + slow_count * SLOW_POWER
        
        total_load   = base_load + ev_load
        grid_pct     = total_load / GRID_CAPACITY * 100.0  # Can exceed 100!

        # ── 3. UPDATE CAR BATTERIES ───────────────────────────────────
        # KEY MECHANIC: Grid overload reduces charging efficiency!
        # Above 85% grid load, charging becomes progressively less effective
        # (simulating voltage drops, brownouts, circuit breaker limits)
        if grid_pct > 85.0:
            overload_factor = max(0.2, 1.0 - (grid_pct - 85.0) / 50.0)
            charge_rates = charge_rates * overload_factor

        self.battery = np.clip(self.battery + charge_rates, 0.0, 100.0)

        # ── 4. ADVANCE TIME ───────────────────────────────────────────
        self.current_step += 1
        self.sim_hour      = (18.0 + self.current_step / 60.0) % 24.0

        # ── 5. CHECK DEPARTURES ───────────────────────────────────────
        step_reward     = 0.0
        newly_departed  = 0

        for i in range(NUM_CARS):
            if self.cars_done[i]:
                continue
            if self.current_step >= self.departure[i]:
                self.cars_done[i] = True
                newly_departed    += 1
                if self.battery[i] >= TARGET_BATTERY:
                    self.success_count += 1
                    step_reward += 25.0    # ✅ Car leaves happy!
                    # Bonus for charging above target (extra satisfaction)
                    overshoot = min((self.battery[i] - TARGET_BATTERY) / 20.0, 1.0)
                    step_reward += 5.0 * overshoot
                else:
                    # Penalty proportional to how short we fell
                    shortfall = (TARGET_BATTERY - self.battery[i]) / TARGET_BATTERY
                    step_reward -= 30.0 * shortfall   # ❌ Car leaves unhappy (stronger penalty)

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

            # Urgency-aware bonus: extra reward for charging cars that are close to departing
            time_left = np.maximum(self.departure[active_now] - self.current_step, 1.0)
            urgent_mask = time_left < 30.0  # cars leaving within 30 min
            if np.any(urgent_mask):
                urgent_charged = np.sum(self.battery[active_now][urgent_mask] >= TARGET_BATTERY)
                step_reward += 2.0 * (urgent_charged / max(np.sum(urgent_mask), 1))

        # 6b. Penalize grid stress (exponential above 90%)
        # Keep penalty mild so agent isn't afraid to charge
        if grid_pct > 90.0:
            step_reward -= 0.2 * ((grid_pct - 90.0) / 20.0) ** 2

        # 6c. Small power efficiency cost
        step_reward -= 0.0005 * (fast_count * FAST_POWER + slow_count * SLOW_POWER)

        # ── 7. TERMINATION CONDITIONS ─────────────────────────────────
        terminated = bool(np.all(self.cars_done))
        truncated  = (self.current_step >= MAX_STEPS)

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

        # ── 8. EXTRA INFO (for logging/analysis) ──────────────────────
        info = {
            "success_count" : self.success_count,
            "total_departed": int(np.sum(self.cars_done)),
            "grid_pct"      : round(float(grid_pct), 2),
            "fast_cars"     : fast_count,
            "slow_cars"     : slow_count,
            "avg_battery"   : round(float(np.mean(self.battery[~self.cars_done])) if np.any(~self.cars_done) else 100.0, 2),
        }

        return self._get_obs(), float(step_reward), terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────
    def _get_obs(self):
        """
        Build the observation vector (what the agent sees).
        All values normalized to [0, 1] for stable neural network training.
        """
        battery_norm    = self.battery / 100.0
        departure_norm  = np.clip(self.departure - self.current_step, 0, MAX_STEPS) / MAX_STEPS
        grid_load_norm  = np.array([self._grid_base_load(self.sim_hour) / GRID_CAPACITY], dtype=np.float32)
        hour_norm       = np.array([self.sim_hour / 24.0], dtype=np.float32)

        obs = np.concatenate([battery_norm, departure_norm, grid_load_norm, hour_norm]).astype(np.float32)
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
                f"Grid: {self._grid_base_load(self.sim_hour):.1f}% | "
                f"Active Cars: {int(np.sum(active))} | "
                f"Avg Battery: {np.mean(self.battery[active]):.1f}% | "
                f"Success: {self.success_count}"
            )
