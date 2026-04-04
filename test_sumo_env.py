#!/usr/bin/env python3
"""Standalone test script for SumoEnv with optional blockage scenarios.

Runs one episode using a fixed-time controller that cycles through the four
green-phase actions (0 → 1 → 2 → 3 → 0 …), spending green_duration steps on
each phase.  Per-decision-step stats are printed to stdout.

Does NOT require LLM, TensorFlow, wandb, or CityFlow.

Usage
-----
    # No blockage, headless SUMO
    python test_sumo_env.py

    # With a blockage scenario
    python test_sumo_env.py --scenario data/sumo/scenarios/accident_single_lane.json

    # Open SUMO-GUI for visualisation
    python test_sumo_env.py --gui

    # Both flags
    python test_sumo_env.py --scenario data/sumo/scenarios/construction_zone.json --gui
"""

import argparse
import os
import sys

# Ensure the repository root is on sys.path so utils.* imports work.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.blockage_manager import BlockageManager
from utils.scenario_config import load_scenario
from utils.sumo_env import SumoEnv

# ── Paths ──────────────────────────────────────────────────────────────────────
_DEFAULT_SUMOCFG = os.path.join(
    _REPO_ROOT, "data", "sumo", "single_intersection", "run.sumocfg"
)

# ── Controller parameters ──────────────────────────────────────────────────────
_TLS_ID = "TLS"
_NUM_STEPS = 3600        # Total simulation steps per episode (== seconds)
_GREEN_DURATION = 30     # Steps held per green phase (== seconds with step-len 1)
_YELLOW_DURATION = 3     # Steps held per yellow transition
_CYCLE_LENGTH = 4        # Number of green-phase actions before cycling


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-time SUMO controller test with optional lane blockages."
    )
    parser.add_argument(
        "--scenario",
        metavar="PATH",
        default=None,
        help="Path to a scenario JSON file (default: no blockages).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch SUMO-GUI instead of headless sumo.",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: D401
    """Entry point."""
    args = _parse_args()

    # Load scenario (optional).
    blockage_manager = None
    if args.scenario is not None:
        name, desc, blockages = load_scenario(args.scenario)
        print(f"Scenario : {name}")
        print(f"           {desc}")
        print(f"Blockages: {len(blockages)}")
        for b in blockages:
            end_str = str(b.end_step) if b.end_step is not None else "episode end"
            print(
                f"  [{b.blockage_id}] {b.lane_id}  method={b.method}"
                f"  severity={b.severity:.2f}"
                f"  steps {b.start_step}–{end_str}"
                f"  pos={b.position:.1f} m"
            )
        if blockages:
            blockage_manager = BlockageManager(blockages)
    else:
        print("Scenario : baseline (no blockages)")

    print()

    # Build environment.
    config = {
        "sumocfg_path": _DEFAULT_SUMOCFG,
        "tls_id": _TLS_ID,
        "num_steps": _NUM_STEPS,
        "yellow_duration": _YELLOW_DURATION,
        "green_duration": _GREEN_DURATION,
        "use_gui": args.gui,
    }
    env = SumoEnv(config, blockage_manager=blockage_manager)

    # Run one episode.
    state = env.reset()
    action = 0
    done = False

    header = f"{'SimStep':>8}  {'Action':>6}  {'Vehicles':>8}  Blocked lanes"
    print(header)
    print("-" * 60)

    while not done:
        state, reward, done, info = env.step(action)

        total_vehicles = sum(state["lane_vehicle_count"].values())
        blocked = state["blocked_lanes"]
        blocked_str = ", ".join(blocked) if blocked else "—"

        print(
            f"{info['step']:>8}  "
            f"{state['current_phase']:>6}  "
            f"{total_vehicles:>8}  "
            f"{blocked_str}"
        )

        # Advance fixed-time cycle.
        action = (action + 1) % _CYCLE_LENGTH

    # Final statistics.
    att = env.get_average_travel_time()
    arrived = info["arrived"]

    print("-" * 60)
    print(f"Average travel time (ATT) : {att:.2f} steps  ({att:.2f} s at 1 s/step)")
    print(f"Total throughput (arrived): {arrived} vehicles")

    env.close()


if __name__ == "__main__":
    main()
