"""SUMO + LLM traffic signal control runner.

Mirrors the structure of ``run_open_LLM.py`` — same LLM loading, querying,
wandb, and output patterns — but drives a SUMO simulation via SumoEnv instead
of CityFlow.

Episode termination matches CityFlow: each episode runs for RUN_COUNTS=3600
simulation steps, with decision steps of green_duration=30 s (= MIN_ACTION_TIME
from utils/config.py), giving 120 decision steps per episode.

Output schema
-------------
Per run directory ``{output_dir}/{memo}/{scenario_name}_{timestamp}/``:
  step_log.csv          — one row per decision step
  episode_summary.csv   — one row per episode (appended)
  state_action_ep{N}.json — state-action trace for episode N (matching
                            the state_action.json format in CityFlow records/)
  agent.conf            — JSON dump of agent configuration
  run.conf              — JSON dump of run configuration

Metric keys match ``utils/model_test.py`` exactly for cross-backend
comparability:
  test_reward, test_avg_queue_len, test_queuing_vehicle_num,
  test_avg_waiting_time, test_avg_travel_time
  (and ``_over`` suffixed equivalents logged to wandb at episode end)
"""

import argparse
import csv
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("run_sumo_open_llm")

# CityFlow equivalent: RUN_COUNTS / MIN_ACTION_TIME = 3600 / 30 = 120
_RUN_COUNTS = 3600
_MIN_ACTION_TIME = 30        # = green_duration inside SumoEnv
_YELLOW_TIME = 3             # = yellow_duration inside SumoEnv
_NUM_PHASES = 4              # green-phase action indices 0–3

# Fallback action when the LLM response cannot be parsed.
_FALLBACK_ACTION = 0

# Response tag used by all SUMO prompts (analogous to <signal> in CityFlow).
_PHASE_ANSWER_PATTERN = re.compile(r"<phase>(.*?)</phase>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an LLM traffic signal controller in SUMO."
    )
    parser.add_argument("--memo",        type=str,   default="SumoLLMRun",
                        help="Run tag used in output paths and wandb group name.")
    parser.add_argument("--llm_path",    type=str,   default="",
                        help="Path to the local LLM checkpoint directory.")
    parser.add_argument("--llm_model",   type=str,   default="local_llm",
                        help="Model name (for logging and wandb only).")
    parser.add_argument("--new_max_tokens", type=int, default=1024,
                        help="Maximum new tokens for LLM generation.")
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
    parser.add_argument("--prompt_type", type=str,   default="commonsense",
                        choices=["commonsense", "waittime", "blockage_aware"],
                        help="Prompt style to use for the LLM.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# LLM loading (lazy – matching initialize_llm in llm_aft_trainer.py)
# ---------------------------------------------------------------------------

def load_llm(llm_path: str, new_max_tokens: int) -> Any:
    """Load a local HuggingFace LLM and tokenizer.

    Raises:
        ImportError: If ``transformers`` or ``torch`` is not installed.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required to run the LLM agent. "
            "Install with: pip install transformers torch"
        ) from exc

    logger.info("Loading LLM from %s", llm_path)

    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        padding_side="left",
        padding=True,
    )
    tokenizer.pad_token_id = 0

    generation_kwargs = {
        "min_length": -1,
        "top_k": 50,
        "top_p": 1.0,
        "temperature": 0.1,
        "do_sample": True,
        "max_new_tokens": new_max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    logger.info("LLM loaded successfully.")
    return model, tokenizer, generation_kwargs


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt_text(
    prompt_type: str,
    state: Dict[str, Any],
    active_blockages: List[Any],
    has_blockages_in_scenario: bool,
) -> str:
    """Build a flat prompt string from the current state.

    Matches the format used in ``utils/my_utils.py::getPrompt``:
      ``system_content + "\\n\\n### Instruction:\\n" + user_content + "\\n\\n### Response:\\n"``

    When ``prompt_type == "blockage_aware"`` but no blockages are currently
    active (or the scenario has none), falls back to ``waittime`` and emits a
    WARNING (spec requirement: not silent).

    Args:
        prompt_type:              "commonsense" | "waittime" | "blockage_aware".
        state:                    Dict from ``SumoEnv.get_state()``.
        active_blockages:         Currently active ``Blockage`` objects.
        has_blockages_in_scenario: True if the scenario JSON contains any
                                   blockage definitions at all.

    Returns:
        Single flat string ready for tokenisation.
    """
    from prompts.sumo_blockage_prompt import (
        get_blockage_prompt,
        get_commonsense_prompt,
        get_waittime_prompt,
    )

    lane_vc = state["lane_vehicle_count"]
    lane_wc = state["lane_waiting_vehicle_count"]
    cur_phase = state["current_phase"]

    effective_type = prompt_type
    if prompt_type == "blockage_aware" and not active_blockages:
        if not has_blockages_in_scenario:
            logger.warning(
                "prompt_type='blockage_aware' requested but scenario has no "
                "blockages defined. Falling back to 'waittime' prompt."
            )
        else:
            logger.warning(
                "prompt_type='blockage_aware' requested but no blockages are "
                "currently active at this step. Falling back to 'waittime' prompt."
            )
        effective_type = "waittime"

    if effective_type == "blockage_aware":
        messages = get_blockage_prompt(lane_vc, lane_wc, active_blockages, cur_phase)
    elif effective_type == "waittime":
        messages = get_waittime_prompt(lane_vc, lane_wc, cur_phase)
    else:  # commonsense
        messages = get_commonsense_prompt(lane_vc, lane_wc, cur_phase)

    # Flat format matching llm_aft_trainer.py (system + instruction + response)
    system_content = messages[0]["content"]
    user_content   = messages[1]["content"]
    return (
        system_content
        + "\n\n### Instruction:\n"
        + user_content
        + "\n\n### Response:\n"
    )


# ---------------------------------------------------------------------------
# LLM query
# ---------------------------------------------------------------------------

def query_llm(
    model: Any,
    tokenizer: Any,
    generation_kwargs: Dict[str, Any],
    prompt: str,
    fail_logs: List[Dict[str, Any]],
    fail_log_file: str,
) -> tuple:
    """Tokenise *prompt*, generate a response, and parse the phase action.

    Mirrors the inference loop in ``LLM_Inference.test()`` in
    ``utils/llm_aft_trainer.py``.

    Returns:
        (action: int, response_text: str)
        *action* is the parsed phase index (0–3), or ``_FALLBACK_ACTION``
        on parse failure.
    """
    try:
        import torch
    except ImportError:
        return _FALLBACK_ACTION, ""

    inputs = tokenizer(
        [prompt],
        truncation=True,
        max_length=2048,
        padding=True,
        return_tensors="pt",
    ).to("cuda" if hasattr(model, "device") and str(getattr(model, "device", "cpu")) != "cpu" else "cpu")

    with torch.no_grad():
        response_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            **generation_kwargs,
        )

    response_text: str = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    # Strip the prompt prefix (same as `res[len(prompts[i]):]` in the trainer).
    response_body = response_text[len(prompt):]

    phases = _PHASE_ANSWER_PATTERN.findall(response_body)
    phase_str = phases[-1].strip() if phases else ""
    action: int
    try:
        action = int(phase_str)
        if action not in range(_NUM_PHASES):
            raise ValueError(f"Phase index {action} out of range 0–3.")
    except (ValueError, TypeError):
        action = _FALLBACK_ACTION
        fail_logs.append({"prompt": prompt, "response": response_body})
        _safe_dump_json(fail_logs, fail_log_file)
        logger.warning(
            "LLM parse failure (got %r); using fallback action %d.",
            phase_str, _FALLBACK_ACTION,
        )

    return action, response_body


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _safe_dump_json(data: Any, path: str) -> None:
    """Write *data* to *path* as JSON, silently on error."""
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
        w = csv.writer(fh)
        w.writerow([
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
        w = csv.writer(fh)
        w.writerow([
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
    model: Optional[Any],
    tokenizer: Optional[Any],
    generation_kwargs: Optional[Dict[str, Any]],
    prompt_type: str,
    has_blockages_in_scenario: bool,
    step_csv: str,
    fail_logs: List[Dict[str, Any]],
    fail_log_file: str,
) -> Dict[str, Any]:
    """Run one full simulation episode and collect metrics.

    Args:
        env:                      Initialised SumoEnv (already reset externally).
        episode_idx:              Zero-based episode counter (for CSV logging).
        model / tokenizer / generation_kwargs: LLM components.
        prompt_type:              Prompt style string.
        has_blockages_in_scenario: Whether the loaded scenario has any
                                   blockage definitions.
        step_csv:                 Path to the step_log.csv file.
        fail_logs:                Mutable list of failed LLM parses (shared
                                  across episodes for accumulation).
        fail_log_file:            Path to save fail_logs JSON.

    Returns:
        Dict with keys matching ``model_test.py`` metric names, plus
        ``throughput`` and ``state_action_log``.
    """
    state = env.reset()
    done = False
    total_reward = 0.0
    queue_length_episode: List[float] = []
    waiting_time_episode: List[float] = []
    state_action_log: List[Dict[str, Any]] = []

    # Termination: RUN_COUNTS / MIN_ACTION_TIME decision steps, matching
    # generator.py: range(int(RUN_COUNTS / MIN_ACTION_TIME))
    max_steps = int(_RUN_COUNTS / _MIN_ACTION_TIME)

    for step_num in range(max_steps):
        if done:
            break

        # Collect active blockages for prompt and logging.
        active_blockages = []
        if env.blockage_manager is not None:
            active_blockages = env.blockage_manager.get_active_blockages()

        # Build prompt and query LLM.
        if model is not None:
            prompt = build_prompt_text(
                prompt_type, state, active_blockages, has_blockages_in_scenario
            )
            action, llm_response = query_llm(
                model, tokenizer, generation_kwargs,
                prompt, fail_logs, fail_log_file,
            )
        else:
            # No model provided — should not happen in this runner; fallback.
            action = _FALLBACK_ACTION
            llm_response = ""
            prompt = ""

        # Log state before the step (mirrors state_action_log in model_test.py).
        state_action_log.append({
            "step": step_num,
            "state": {
                "lane_vehicle_count": dict(state["lane_vehicle_count"]),
                "lane_waiting_vehicle_count": dict(state["lane_waiting_vehicle_count"]),
                "current_phase": state["current_phase"],
                "blocked_lanes": list(state["blocked_lanes"]),
            },
            "action": action,
            "llm_response": llm_response,
        })

        next_state, reward, done, info = env.step(action)
        state_action_log[-1]["reward"] = float(reward)

        # Accumulate per-step metrics (same as oneline.py / model_test.py).
        total_reward += reward
        total_waiting = sum(state["lane_waiting_vehicle_count"].values())
        queue_length_episode.append(float(total_waiting))

        # Approximate per-step waiting time as mean waiting per vehicle.
        total_veh = sum(state["lane_vehicle_count"].values())
        waiting_time_episode.append(
            total_waiting / max(total_veh, 1) * _MIN_ACTION_TIME
        )

        _append_step_row(step_csv, episode_idx, step_num, action, state, reward)
        state = next_state

    # Travel time via SumoEnv's internal tracking (mirrors vehicle_travel_times
    # calculation in model_test.py / oneline.py).
    avg_travel_time = env.get_average_travel_time()

    results: Dict[str, Any] = {
        "test_reward":             total_reward,
        "test_avg_queue_len":      float(np.mean(queue_length_episode)) if queue_length_episode else 0.0,
        "test_queuing_vehicle_num": float(np.sum(queue_length_episode)) if queue_length_episode else 0.0,
        "test_avg_waiting_time":   float(np.mean(waiting_time_episode)) if waiting_time_episode else 0.0,
        "test_avg_travel_time":    avg_travel_time,
        "throughput":              info.get("arrived", 0),
        "state_action_log":        state_action_log,
    }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(in_args: argparse.Namespace) -> None:
    """Entry point — mirrors the ``main()`` structure of ``run_open_LLM.py``."""
    # ── Lazy imports (SUMO not required at import time) ──────────────────────
    try:
        from utils.sumo_env import SumoEnv
    except ImportError as exc:
        raise ImportError(
            "traci / sumolib not found.  Install SUMO (SUMO 1.18+) and ensure "
            "traci is on sys.path.  See requirements_sumo.txt for details."
        ) from exc

    from utils.blockage_manager import BlockageManager
    from utils.scenario_config import load_scenario

    # ── Scenario loading ──────────────────────────────────────────────────────
    scenario_name, scenario_desc, blockages = load_scenario(in_args.scenario)
    has_blockages = len(blockages) > 0
    blockage_manager = BlockageManager(blockages) if has_blockages else None

    logger.info("Scenario : %s", scenario_name)
    logger.info("           %s", scenario_desc)
    logger.info("Blockages: %d", len(blockages))

    # ── Output directory (mirrors path construction in run_open_LLM.py) ─────
    run_dir = _setup_output_dir(in_args.output_dir, in_args.memo, scenario_name)
    logger.info("Output   : %s", run_dir)

    step_csv    = _init_step_csv(run_dir)
    episode_csv = _init_episode_csv(run_dir)
    fail_log_file = os.path.join(run_dir, "llm_fail_logs.json")
    fail_logs: List[Dict[str, Any]] = []

    # ── Save run configuration (mirrors copy_conf_file in pipeline.py) ───────
    agent_conf = {
        "LLM_PATH":       in_args.llm_path,
        "LLM_MODEL":      in_args.llm_model,
        "NEW_MAX_TOKENS": in_args.new_max_tokens,
        "PROMPT_TYPE":    in_args.prompt_type,
    }
    run_conf = {
        "SCENARIO":       in_args.scenario,
        "SCENARIO_NAME":  scenario_name,
        "NUM_EPISODES":   in_args.num_episodes,
        "RUN_COUNTS":     _RUN_COUNTS,
        "MIN_ACTION_TIME": _MIN_ACTION_TIME,
        "YELLOW_TIME":    _YELLOW_TIME,
        "USE_GUI":        in_args.use_gui,
        "MODEL_NAME":     f"{in_args.memo}-{in_args.llm_model}",
        "PROJECT_NAME":   in_args.proj_name,
    }
    _safe_dump_json(agent_conf, os.path.join(run_dir, "agent.conf"))
    _safe_dump_json(run_conf,   os.path.join(run_dir, "run.conf"))

    # ── LLM loading (matches LLM_Inference.initialize_llm) ───────────────────
    model, tokenizer, generation_kwargs = load_llm(
        in_args.llm_path, in_args.new_max_tokens
    )

    # ── wandb initialisation (mirrors pipeline.py / oneline.py) ─────────────
    wandb_logger = None
    try:
        import wandb
        all_config = {**agent_conf, **run_conf}
        wandb_logger = wandb.init(
            project=in_args.proj_name,
            group=f"{in_args.memo}-{in_args.llm_model}-{scenario_name}-{_NUM_PHASES}_Phases",
            name=f"{scenario_name}",
            config=all_config,
        )
        logger.info("wandb initialised.")
    except Exception as exc:
        logger.warning("wandb not available, skipping: %s", exc)

    # ── SumoEnv initialisation ────────────────────────────────────────────────
    sumo_config = {
        "sumocfg_path":    "data/sumo/single_intersection/run.sumocfg",
        "tls_id":          "TLS",
        "num_steps":       _RUN_COUNTS,
        "yellow_duration": _YELLOW_TIME,
        "green_duration":  _MIN_ACTION_TIME,
        "use_gui":         in_args.use_gui,
    }
    env = SumoEnv(sumo_config, blockage_manager=blockage_manager)

    # ── Episode loop ─────────────────────────────────────────────────────────
    all_results: List[Dict[str, Any]] = []
    last_10_results: Dict[str, List[float]] = {
        "test_reward_over": [],
        "test_avg_queue_len_over": [],
        "test_queuing_vehicle_num_over": [],
        "test_avg_waiting_time_over": [],
        "test_avg_travel_time_over": [],
    }

    for ep in range(in_args.num_episodes):
        logger.info("===== Episode %d / %d =====", ep + 1, in_args.num_episodes)
        ep_start = time.time()

        results = run_episode(
            env=env,
            episode_idx=ep,
            model=model,
            tokenizer=tokenizer,
            generation_kwargs=generation_kwargs,
            prompt_type=in_args.prompt_type,
            has_blockages_in_scenario=has_blockages,
            step_csv=step_csv,
            fail_logs=fail_logs,
            fail_log_file=fail_log_file,
        )

        # Save per-episode state_action log (mirrors dump_json(state_action_log, ...)).
        sa_path = os.path.join(run_dir, f"state_action_ep{ep}.json")
        _safe_dump_json(results["state_action_log"], sa_path)

        ep_metrics = {k: v for k, v in results.items() if k != "state_action_log"}
        all_results.append(ep_metrics)

        _append_episode_row(
            episode_csv, ep, ep_metrics, scenario_name,
            f"llm_{in_args.llm_model}"
        )

        # Accumulate for last-10 wandb log (mirrors pipeline.py).
        if ep >= max(0, in_args.num_episodes - 10):
            for key in last_10_results:
                base = key[:-5]  # strip "_over"
                last_10_results[key].append(ep_metrics.get(base, 0.0))

        # Per-episode wandb log (mirrors ``logger.log(results)`` in model_test.py).
        if wandb_logger is not None:
            try:
                log_dict = {
                    "test_reward":             ep_metrics["test_reward"],
                    "test_avg_queue_len":      ep_metrics["test_avg_queue_len"],
                    "test_queuing_vehicle_num": ep_metrics["test_queuing_vehicle_num"],
                    "test_avg_waiting_time":   ep_metrics["test_avg_waiting_time"],
                    "test_avg_travel_time":    ep_metrics["test_avg_travel_time"],
                    "throughput":              ep_metrics["throughput"],
                }
                wandb_logger.log(log_dict)
            except Exception as exc:
                logger.warning("wandb log failed: %s", exc)

        elapsed = time.time() - ep_start
        logger.info(
            "Episode %d done in %.1f s  |  ATT=%.2f  throughput=%d  reward=%.2f",
            ep + 1, elapsed,
            ep_metrics["test_avg_travel_time"],
            ep_metrics["throughput"],
            ep_metrics["test_reward"],
        )

    env.close()

    # ── Final wandb summary (mirrors last_10_results in pipeline.py) ─────────
    if wandb_logger is not None:
        try:
            final_summary = {
                k: float(np.mean(v)) for k, v in last_10_results.items() if v
            }
            wandb_logger.log(final_summary)
            import wandb as _wandb
            _wandb.finish()
            logger.info("wandb finished.")
        except Exception as exc:
            logger.warning("wandb finish failed: %s", exc)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Run complete  |  {in_args.num_episodes} episode(s)")
    print(f"Scenario     : {scenario_name}")
    print(f"Controller   : llm_{in_args.llm_model}")
    if all_results:
        mean_att       = np.mean([r["test_avg_travel_time"] for r in all_results])
        mean_tp        = np.mean([r["throughput"]           for r in all_results])
        mean_queue     = np.mean([r["test_avg_queue_len"]   for r in all_results])
        mean_wait      = np.mean([r["test_avg_waiting_time"] for r in all_results])
        print(f"Mean ATT     : {mean_att:.2f} s")
        print(f"Mean through : {mean_tp:.1f} veh")
        print(f"Mean queue   : {mean_queue:.2f}")
        print(f"Mean wait    : {mean_wait:.2f} s")
    print(f"Results      : {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
