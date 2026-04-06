"""SUMO heuristic baseline runner (FixedTime and MaxPressure).

Mirrors the structure of ``run_sumo_open_llm.py`` — same argument parsing,
episode loop, termination logic, and output format — but drives the controller
with a deterministic heuristic instead of an LLM.

MaxPressure adaptation
----------------------
The CityFlow ``MaxPressureAgent`` reads a ``traffic_movement_pressure_queue``
array produced by the CityFlow environment.  For SUMO we reconstruct the
equivalent pressure from ``SumoEnv.get_state()["lane_waiting_vehicle_count"]``:

    Phase 0 (N-S through): pressure = wait(N2TLS_0) + wait(S2TLS_0)
    Phase 1 (N-S left):    pressure = wait(N2TLS_1) + wait(S2TLS_1)
    Phase 2 (E-W through): pressure = wait(E2TLS_0) + wait(W2TLS_0)
    Phase 3 (E-W left):    pressure = wait(E2TLS_1) + wait(W2TLS_1)

The action with the highest pressure is chosen.  This is functionally
equivalent to the ``choose_action`` method in ``models/maxpressure_agent.py``
applied to a 4-phase intersection.

Output schema
-------------
Identical to ``run_sumo_open_llm.py``:
  step_log.csv, episode_summary.csv, state_action_ep{N}.json,
  agent.conf, run.conf
"""

import argparse
import csv
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("run_sumo_baseline")

# Matching CityFlow runner constants (from utils/config.py).
_RUN_COUNTS     = 3600
_MIN_ACTION_TIME = 30        # green_duration per decision step
_YELLOW_TIME    = 3          # yellow_duration
_NUM_PHASES     = 4

# MaxPressure: lanes serving each phase action.
# Matches the net.xml phase-to-lane mapping documented in SumoEnv.ACTION_TO_SUMO_PHASE.
_MP_PHASE_LANES: Dict[int, List[str]] = {
    0: ["N2TLS_0", "S2TLS_0"],   # N-S through
    1: ["N2TLS_1", "S2TLS_1"],   # N-S left
    2: ["E2TLS_0", "W2TLS_0"],   # E-W through
    3: ["E2TLS_1", "W2TLS_1"],   # E-W left
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a heuristic baseline traffic signal controller in SUMO."
    )
    parser.add_argument("--memo",        type=str,   default="SumoBaseline",
                        help="Run tag used in output paths and wandb group name.")
    parser.add_argument("--controller",  type=str,   default="fixedtime",
                        choices=["fixedtime", "maxpressure"],
                        help="Heuristic controller to use.")
    parser.add_argument("--scenario",    type=str,
                        default="data/sumo/scenarios/baseline.json",
                        help="Path to scenario JSON file.")
    parser.add_argument("--num_episodes", type=int,  default=3,
                        help="Number of independent simulation episodes.")
    parser.add_argument("--output_dir",  type=str,   default="./results/sumo",
                        help="Root output directory.")
    parser.add_argument("--use_gui",     action="store_true", default=False,
                        help="Launch sumo-gui instead of headless sumo.")
    parser.add_argument("--proj_name",   type=str,   default="LLM-TSCS-SUMO",
                        help="wandb project name.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Controller implementations
# ---------------------------------------------------------------------------

def fixedtime_action(step_num: int, _state: Dict[str, Any]) -> int:
    """Cycle through actions 0→1→2→3→0… one phase per decision step.

    Mirrors ``models/fixedtime_agent.py`` which advances phases sequentially,
    spending ``FIXED_TIME`` steps (= MIN_ACTION_TIME here) on each.

    Args:
        step_num: Zero-based decision step counter (within episode).
        _state:   Unused; present for a consistent action-function signature.

    Returns:
        Integer action index 0–3.
    """
    return step_num % _NUM_PHASES


def maxpressure_action(state: Dict[str, Any]) -> int:
    """Choose the phase with the highest cumulative waiting-vehicle pressure.

    Adapts ``models/maxpressure_agent.py::choose_action`` to consume
    ``SumoEnv.get_state()`` directly instead of CityFlow's state dict.

    For each phase, pressure = sum of waiting vehicles in the lanes that phase
    serves.  Ties are broken by lowest action index (``np.argmax`` semantics).

    Args:
        state: Dict from ``SumoEnv.get_state()``.

    Returns:
        Integer action index 0–3.
    """
    waiting = state["lane_waiting_vehicle_count"]
    pressures = [
        sum(waiting.get(ln, 0) for ln in _MP_PHASE_LANES[a])
        for a in range(_NUM_PHASES)
    ]
    return int(np.argmax(pressures))


# ---------------------------------------------------------------------------
# Output helpers  (identical to run_sumo_open_llm.py)
# ---------------------------------------------------------------------------

def _safe_dump_json(data: Any, path: str) -> None:
    """Write *data* as JSON to *path*, silently on error."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception as exc:
        logger.warning("Could not write JSON to %s: %s", path, exc)


def _setup_output_dir(output_dir: str, memo: str, scenario_name: str) -> str:
    """Create a timestamped run directory, return its path."""
    stamp = time.strftime("%m_%d_%H_%M_%S", time.localtime())
    run_dir = os.path.join(output_dir, memo, f"{scenario_name}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _init_step_csv(run_dir: str) -> str:
    """Create step_log.csv with header, return file path."""
    path = os.path.join(run_dir, "step_log.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([
            "episode", "step", "action", "total_vehicle_count",
            "total_waiting_count", "reward", "blocked_lanes",
        ])
    return path


def _append_step_row(
    step_csv: str,
    episode: int,
    step: int,
    action: int,
    state: Dict[str, Any],
    reward: float,
) -> None:
    """Append one row to step_log.csv."""
    total_veh  = sum(state["lane_vehicle_count"].values())
    total_wait = sum(state["lane_waiting_vehicle_count"].values())
    blocked    = "|".join(state["blocked_lanes"]) if state["blocked_lanes"] else ""
    with open(step_csv, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(
            [episode, step, action, total_veh, total_wait, f"{reward:.4f}", blocked]
        )


def _init_episode_csv(run_dir: str) -> str:
    """Create episode_summary.csv with header, return file path."""
    path = os.path.join(run_dir, "episode_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([
            "episode", "test_reward", "test_avg_queue_len",
            "test_queuing_vehicle_num", "test_avg_waiting_time",
            "test_avg_travel_time", "throughput",
            "scenario", "controller",
        ])
    return path


def _append_episode_row(
    ep_csv: str,
    episode: int,
    results: Dict[str, Any],
    scenario_name: str,
    controller: str,
) -> None:
    """Append one row to episode_summary.csv."""
    with open(ep_csv, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([
            episode,
            f"{results['test_reward']:.4f}",
            f"{results['test_avg_queue_len']:.4f}",
            int(results["test_queuing_vehicle_num"]),
            f"{results['test_avg_waiting_time']:.4f}",
            f"{results['test_avg_travel_time']:.4f}",
            int(results.get("throughput", 0)),
            scenario_name,
            controller,
        ])


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: Any,
    episode_idx: int,
    controller: str,
    step_csv: str,
) -> Dict[str, Any]:
    """Run one full simulation episode, return metrics dict.

    Args:
        env:          Initialised SumoEnv (reset called inside here).
        episode_idx:  Zero-based episode counter for CSV logging.
        controller:   ``"fixedtime"`` or ``"maxpressure"``.
        step_csv:     Path to the open step_log.csv file.

    Returns:
        Dict with CityFlow-compatible metric keys plus ``throughput`` and
        ``state_action_log``.
    """
    state = env.reset()
    done = False
    total_reward = 0.0
    queue_length_episode: List[float] = []
    waiting_time_episode: List[float] = []
    state_action_log: List[Dict[str, Any]] = []

    max_steps = int(_RUN_COUNTS / _MIN_ACTION_TIME)

    for step_num in range(max_steps):
        if done:
            break

        # Choose action based on controller.
        if controller == "fixedtime":
            action = fixedtime_action(step_num, state)
        else:  # maxpressure
            action = maxpressure_action(state)

        # Log state before the step.
        state_action_log.append({
            "step": step_num,
            "state": {
                "lane_vehicle_count":         dict(state["lane_vehicle_count"]),
                "lane_waiting_vehicle_count":  dict(state["lane_waiting_vehicle_count"]),
                "current_phase":              state["current_phase"],
                "blocked_lanes":              list(state["blocked_lanes"]),
            },
            "action": action,
        })

        next_state, reward, done, info = env.step(action)
        state_action_log[-1]["reward"] = float(reward)

        total_reward += reward
        total_waiting = sum(state["lane_waiting_vehicle_count"].values())
        total_veh     = sum(state["lane_vehicle_count"].values())
        queue_length_episode.append(float(total_waiting))
        waiting_time_episode.append(
            total_waiting / max(total_veh, 1) * _MIN_ACTION_TIME
        )

        _append_step_row(step_csv, episode_idx, step_num, action, state, reward)
        state = next_state

    avg_travel_time = env.get_average_travel_time()

    return {
        "test_reward":              total_reward,
        "test_avg_queue_len":       float(np.mean(queue_length_episode)) if queue_length_episode else 0.0,
        "test_queuing_vehicle_num": float(np.sum(queue_length_episode))  if queue_length_episode else 0.0,
        "test_avg_waiting_time":    float(np.mean(waiting_time_episode)) if waiting_time_episode else 0.0,
        "test_avg_travel_time":     avg_travel_time,
        "throughput":               info.get("arrived", 0),
        "state_action_log":         state_action_log,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(in_args: argparse.Namespace) -> None:
    """Entry point — mirrors ``main()`` in ``run_sumo_open_llm.py``."""
    # ── Lazy imports ──────────────────────────────────────────────────────────
    try:
        from utils.sumo_env import SumoEnv
    except ImportError as exc:
        raise ImportError(
            "traci / sumolib not found.  Install SUMO 1.18+ and ensure "
            "traci is on sys.path.  See requirements_sumo.txt for details."
        ) from exc

    from utils.blockage_manager import BlockageManager
    from utils.scenario_config import load_scenario

    # ── Scenario loading ──────────────────────────────────────────────────────
    scenario_name, scenario_desc, blockages = load_scenario(in_args.scenario)
    has_blockages = len(blockages) > 0
    blockage_manager = BlockageManager(blockages) if has_blockages else None

    logger.info("Scenario   : %s", scenario_name)
    logger.info("             %s", scenario_desc)
    logger.info("Blockages  : %d", len(blockages))
    logger.info("Controller : %s", in_args.controller)

    # ── Output directory ──────────────────────────────────────────────────────
    run_dir = _setup_output_dir(in_args.output_dir, in_args.memo, scenario_name)
    logger.info("Output     : %s", run_dir)

    step_csv    = _init_step_csv(run_dir)
    episode_csv = _init_episode_csv(run_dir)

    # ── Save configuration ────────────────────────────────────────────────────
    agent_conf = {
        "CONTROLLER":   in_args.controller,
        "FIXED_TIME":   [_MIN_ACTION_TIME] * _NUM_PHASES,
    }
    run_conf = {
        "SCENARIO":        in_args.scenario,
        "SCENARIO_NAME":   scenario_name,
        "NUM_EPISODES":    in_args.num_episodes,
        "RUN_COUNTS":      _RUN_COUNTS,
        "MIN_ACTION_TIME": _MIN_ACTION_TIME,
        "YELLOW_TIME":     _YELLOW_TIME,
        "USE_GUI":         in_args.use_gui,
        "MODEL_NAME":      f"{in_args.memo}-{in_args.controller}",
        "PROJECT_NAME":    in_args.proj_name,
    }
    _safe_dump_json(agent_conf, os.path.join(run_dir, "agent.conf"))
    _safe_dump_json(run_conf,   os.path.join(run_dir, "run.conf"))

    # ── wandb initialisation ──────────────────────────────────────────────────
    wandb_logger = None
    try:
        import wandb
        all_config = {**agent_conf, **run_conf}
        wandb_logger = wandb.init(
            project=in_args.proj_name,
            group=f"{in_args.memo}-{in_args.controller}-{scenario_name}-{_NUM_PHASES}_Phases",
            name=f"{scenario_name}",
            config=all_config,
        )
        logger.info("wandb initialised.")
    except Exception as exc:
        logger.warning("wandb not available, skipping: %s", exc)

    # ── SumoEnv ───────────────────────────────────────────────────────────────
    sumo_config = {
        "sumocfg_path":    "data/sumo/single_intersection/run.sumocfg",
        "tls_id":          "TLS",
        "num_steps":       _RUN_COUNTS,
        "yellow_duration": _YELLOW_TIME,
        "green_duration":  _MIN_ACTION_TIME,
        "use_gui":         in_args.use_gui,
    }
    env = SumoEnv(sumo_config, blockage_manager=blockage_manager)

    # ── Episode loop ──────────────────────────────────────────────────────────
    all_results: List[Dict[str, Any]] = []
    last_10: Dict[str, List[float]] = {
        "test_reward_over": [],
        "test_avg_queue_len_over": [],
        "test_queuing_vehicle_num_over": [],
        "test_avg_waiting_time_over": [],
        "test_avg_travel_time_over": [],
    }

    for ep in range(in_args.num_episodes):
        logger.info("===== Episode %d / %d =====", ep + 1, in_args.num_episodes)
        ep_start = time.time()

        results = run_episode(env, ep, in_args.controller, step_csv)

        sa_path = os.path.join(run_dir, f"state_action_ep{ep}.json")
        _safe_dump_json(results["state_action_log"], sa_path)

        ep_metrics = {k: v for k, v in results.items() if k != "state_action_log"}
        all_results.append(ep_metrics)

        _append_episode_row(episode_csv, ep, ep_metrics, scenario_name, in_args.controller)

        if ep >= max(0, in_args.num_episodes - 10):
            for key in last_10:
                base = key[:-5]
                last_10[key].append(ep_metrics.get(base, 0.0))

        # Per-episode wandb log (mirrors model_test.py).
        if wandb_logger is not None:
            try:
                wandb_logger.log({
                    "test_reward":             ep_metrics["test_reward"],
                    "test_avg_queue_len":      ep_metrics["test_avg_queue_len"],
                    "test_queuing_vehicle_num": ep_metrics["test_queuing_vehicle_num"],
                    "test_avg_waiting_time":   ep_metrics["test_avg_waiting_time"],
                    "test_avg_travel_time":    ep_metrics["test_avg_travel_time"],
                    "throughput":              ep_metrics["throughput"],
                })
            except Exception as exc:
                logger.warning("wandb log failed: %s", exc)

        logger.info(
            "Episode %d done in %.1f s  |  ATT=%.2f  throughput=%d  reward=%.2f",
            ep + 1, time.time() - ep_start,
            ep_metrics["test_avg_travel_time"],
            ep_metrics["throughput"],
            ep_metrics["test_reward"],
        )

    env.close()

    # ── Final wandb summary ───────────────────────────────────────────────────
    if wandb_logger is not None:
        try:
            final_summary = {
                k: float(np.mean(v)) for k, v in last_10.items() if v
            }
            wandb_logger.log(final_summary)
            import wandb as _wandb
            _wandb.finish()
        except Exception as exc:
            logger.warning("wandb finish failed: %s", exc)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Run complete  |  {in_args.num_episodes} episode(s)")
    print(f"Scenario     : {scenario_name}")
    print(f"Controller   : {in_args.controller}")
    if all_results:
        print(f"Mean ATT     : {np.mean([r['test_avg_travel_time'] for r in all_results]):.2f} s")
        print(f"Mean through : {np.mean([r['throughput']           for r in all_results]):.1f} veh")
        print(f"Mean queue   : {np.mean([r['test_avg_queue_len']   for r in all_results]):.2f}")
        print(f"Mean wait    : {np.mean([r['test_avg_waiting_time'] for r in all_results]):.2f} s")
    print(f"Results      : {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
