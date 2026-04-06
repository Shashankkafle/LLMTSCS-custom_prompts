"""Evaluation script for SUMO simulation results.

Reads ``episode_summary.csv`` files written by ``run_sumo_open_llm.py`` and
``run_sumo_baseline.py`` and computes the same metrics that
``utils/model_test.py`` reports for CityFlow, enabling direct cross-backend
comparison:

  * Mean ATT (average travel time)
  * Mean throughput (arrived vehicles per episode)
  * Mean max queue length
  * Mean per-step queue length
  * Mean per-step waiting time
  * ATT broken down by episode

When two output directories are provided, results are displayed side by side
and a comparison CSV is saved.

Usage
-----
    # Single directory
    python evaluate_sumo.py results/sumo/SumoBaseline/baseline_05_01_12_00_00

    # Side-by-side comparison
    python evaluate_sumo.py \\
        results/sumo/SumoBaseline/baseline_05_01_12_00_00 \\
        results/sumo/SumoLLMRun/accident_single_lane_05_01_12_30_00 \\
        --output_csv comparison.csv
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Column contract for episode_summary.csv
# (must stay in sync with _init_episode_csv / _append_episode_row in runners)
# ---------------------------------------------------------------------------
_EP_COLUMNS = [
    "episode",
    "test_reward",
    "test_avg_queue_len",
    "test_queuing_vehicle_num",
    "test_avg_waiting_time",
    "test_avg_travel_time",
    "throughput",
    "scenario",
    "controller",
]

_FLOAT_COLS = {
    "test_reward",
    "test_avg_queue_len",
    "test_queuing_vehicle_num",
    "test_avg_waiting_time",
    "test_avg_travel_time",
    "throughput",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _find_episode_csv(directory: str) -> Optional[str]:
    """Return the path to ``episode_summary.csv`` in *directory*, or None."""
    candidate = os.path.join(directory, "episode_summary.csv")
    return candidate if os.path.isfile(candidate) else None


def _find_step_csv(directory: str) -> Optional[str]:
    """Return the path to ``step_log.csv`` in *directory*, or None."""
    candidate = os.path.join(directory, "step_log.csv")
    return candidate if os.path.isfile(candidate) else None


def _read_episode_csv(path: str) -> List[Dict[str, Any]]:
    """Read *path* and return a list of row dicts with numeric columns cast."""
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cast: Dict[str, Any] = {}
            for col, val in row.items():
                if col in _FLOAT_COLS:
                    try:
                        cast[col] = float(val)
                    except (ValueError, TypeError):
                        cast[col] = float("nan")
                else:
                    cast[col] = val
            rows.append(cast)
    return rows


def _read_step_csv(path: str) -> List[Dict[str, Any]]:
    """Read *path* and return a list of row dicts."""
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cast: Dict[str, Any] = {}
            for col, val in row.items():
                try:
                    cast[col] = float(val)
                except (ValueError, TypeError):
                    cast[col] = val
            rows.append(cast)
    return rows


def _read_run_conf(directory: str) -> Dict[str, Any]:
    """Load run.conf (or agent.conf) if present, else return empty dict."""
    for fname in ("run.conf", "agent.conf"):
        p = os.path.join(directory, fname)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                try:
                    return json.load(fh)
                except Exception:
                    pass
    return {}


# ---------------------------------------------------------------------------
# Per-directory statistics
# ---------------------------------------------------------------------------

def compute_stats(directory: str) -> Dict[str, Any]:
    """Compute summary statistics for one run directory.

    Returns a dict with the following keys (matching ``model_test.py`` metrics
    plus extras for the evaluation script):

    From ``episode_summary.csv``
      mean_att          — mean of test_avg_travel_time across episodes
      std_att           — std of test_avg_travel_time
      att_per_episode   — list of per-episode ATT values
      mean_throughput   — mean arrived vehicles per episode
      std_throughput    — std of throughput
      mean_avg_queue    — mean of test_avg_queue_len
      std_avg_queue     — std
      mean_avg_waiting  — mean of test_avg_waiting_time
      std_avg_waiting   — std
      mean_total_reward — mean reward per episode
      num_episodes      — number of rows in episode_summary.csv

    From ``step_log.csv`` (if present)
      mean_max_queue    — mean per-episode maximum total_waiting_count

    Metadata
      directory         — input directory path
      scenario          — scenario name (from CSV or run.conf)
      controller        — controller name (from CSV or run.conf)
    """
    ep_csv_path = _find_episode_csv(directory)
    if ep_csv_path is None:
        raise FileNotFoundError(
            f"No episode_summary.csv found in '{directory}'. "
            "Ensure the directory was produced by run_sumo_baseline.py or "
            "run_sumo_open_llm.py."
        )

    episodes = _read_episode_csv(ep_csv_path)
    if not episodes:
        raise ValueError(f"episode_summary.csv in '{directory}' is empty.")

    conf = _read_run_conf(directory)

    # Pull scalar lists for core metrics.
    atts       = [r["test_avg_travel_time"]  for r in episodes]
    throughputs = [r["throughput"]            for r in episodes]
    avg_queues  = [r["test_avg_queue_len"]    for r in episodes]
    avg_waits   = [r["test_avg_waiting_time"] for r in episodes]
    rewards     = [r["test_reward"]           for r in episodes]

    # Max queue per episode from step_log.csv (if available).
    mean_max_queue: Optional[float] = None
    step_csv_path = _find_step_csv(directory)
    if step_csv_path is not None:
        step_rows = _read_step_csv(step_csv_path)
        if step_rows:
            # Group waiting counts by episode.
            ep_waits: Dict[int, List[float]] = {}
            for row in step_rows:
                ep_idx = int(row.get("episode", 0))
                w = float(row.get("total_waiting_count", 0.0))
                ep_waits.setdefault(ep_idx, []).append(w)
            max_per_ep = [max(waits) for waits in ep_waits.values() if waits]
            if max_per_ep:
                mean_max_queue = float(np.mean(max_per_ep))

    # Scenario / controller from CSV rows (fallback to conf).
    scenario   = episodes[0].get("scenario",   conf.get("SCENARIO_NAME",   "unknown"))
    controller = episodes[0].get("controller", conf.get("CONTROLLER",      "unknown"))

    return {
        "directory":        directory,
        "scenario":         scenario,
        "controller":       controller,
        "num_episodes":     len(episodes),
        "mean_att":         float(np.mean(atts)),
        "std_att":          float(np.std(atts)),
        "att_per_episode":  atts,
        "mean_throughput":  float(np.mean(throughputs)),
        "std_throughput":   float(np.std(throughputs)),
        "mean_avg_queue":   float(np.mean(avg_queues)),
        "std_avg_queue":    float(np.std(avg_queues)),
        "mean_avg_waiting": float(np.mean(avg_waits)),
        "std_avg_waiting":  float(np.std(avg_waits)),
        "mean_total_reward": float(np.mean(rewards)),
        "mean_max_queue":   mean_max_queue,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_COL_W = 28   # label column width
_VAL_W = 20   # value column width


def _fmt(val: Any, decimals: int = 2) -> str:
    """Format a value for tabular display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _print_single(stats: Dict[str, Any]) -> None:
    """Print statistics for a single run directory."""
    hdr = f"{'Metric':<{_COL_W}}  {'Value':>{_VAL_W}}"
    sep = "─" * (len(hdr) + 2)
    print(sep)
    print(f"Directory : {stats['directory']}")
    print(f"Scenario  : {stats['scenario']}")
    print(f"Controller: {stats['controller']}")
    print(f"Episodes  : {stats['num_episodes']}")
    print(sep)
    print(hdr)
    print(sep)

    rows = [
        ("Mean ATT (s)",              _fmt(stats["mean_att"])),
        ("  Std ATT",                 _fmt(stats["std_att"])),
        ("Mean throughput (veh/ep)",  _fmt(stats["mean_throughput"], 1)),
        ("  Std throughput",          _fmt(stats["std_throughput"], 1)),
        ("Mean avg queue (veh)",      _fmt(stats["mean_avg_queue"])),
        ("  Std avg queue",           _fmt(stats["std_avg_queue"])),
        ("Mean max queue (veh)",      _fmt(stats["mean_max_queue"])),
        ("Mean avg wait (s)",         _fmt(stats["mean_avg_waiting"])),
        ("Mean reward",               _fmt(stats["mean_total_reward"])),
    ]
    for label, val in rows:
        print(f"{label:<{_COL_W}}  {val:>{_VAL_W}}")

    print(sep)
    print("ATT per episode:")
    for i, att in enumerate(stats["att_per_episode"]):
        print(f"  Episode {i:>3}: {att:.2f} s")
    print()


def _print_comparison(stats_a: Dict[str, Any], stats_b: Dict[str, Any]) -> None:
    """Print a side-by-side comparison of two run directories."""
    hdr = (
        f"{'Metric':<{_COL_W}}  "
        f"{'Dir A':>{_VAL_W}}  "
        f"{'Dir B':>{_VAL_W}}  "
        f"{'Δ (B−A)':>{_VAL_W}}"
    )
    sep = "─" * (len(hdr) + 2)
    print(sep)
    print(f"Dir A  : {stats_a['directory']}")
    print(f"  Scenario={stats_a['scenario']}  Controller={stats_a['controller']}")
    print(f"Dir B  : {stats_b['directory']}")
    print(f"  Scenario={stats_b['scenario']}  Controller={stats_b['controller']}")
    print(sep)
    print(hdr)
    print(sep)

    def delta(a: Any, b: Any, decimals: int = 2) -> str:
        if a is None or b is None:
            return "N/A"
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.{decimals}f}"

    rows: List[Tuple[str, str, str, str]] = [
        ("Mean ATT (s)",
         _fmt(stats_a["mean_att"]),
         _fmt(stats_b["mean_att"]),
         delta(stats_a["mean_att"], stats_b["mean_att"])),
        ("  Std ATT",
         _fmt(stats_a["std_att"]),
         _fmt(stats_b["std_att"]),
         ""),
        ("Mean throughput (veh/ep)",
         _fmt(stats_a["mean_throughput"], 1),
         _fmt(stats_b["mean_throughput"], 1),
         delta(stats_a["mean_throughput"], stats_b["mean_throughput"], 1)),
        ("Mean avg queue (veh)",
         _fmt(stats_a["mean_avg_queue"]),
         _fmt(stats_b["mean_avg_queue"]),
         delta(stats_a["mean_avg_queue"], stats_b["mean_avg_queue"])),
        ("Mean max queue (veh)",
         _fmt(stats_a["mean_max_queue"]),
         _fmt(stats_b["mean_max_queue"]),
         delta(
             stats_a["mean_max_queue"] if stats_a["mean_max_queue"] is not None else float("nan"),
             stats_b["mean_max_queue"] if stats_b["mean_max_queue"] is not None else float("nan"),
         )),
        ("Mean avg wait (s)",
         _fmt(stats_a["mean_avg_waiting"]),
         _fmt(stats_b["mean_avg_waiting"]),
         delta(stats_a["mean_avg_waiting"], stats_b["mean_avg_waiting"])),
        ("Mean reward",
         _fmt(stats_a["mean_total_reward"]),
         _fmt(stats_b["mean_total_reward"]),
         delta(stats_a["mean_total_reward"], stats_b["mean_total_reward"])),
    ]
    for label, va, vb, d in rows:
        print(f"{label:<{_COL_W}}  {va:>{_VAL_W}}  {vb:>{_VAL_W}}  {d:>{_VAL_W}}")

    print(sep)

    # ATT per episode side-by-side.
    print("ATT per episode (s):")
    max_ep = max(len(stats_a["att_per_episode"]), len(stats_b["att_per_episode"]))
    for i in range(max_ep):
        a_val = stats_a["att_per_episode"][i] if i < len(stats_a["att_per_episode"]) else None
        b_val = stats_b["att_per_episode"][i] if i < len(stats_b["att_per_episode"]) else None
        print(
            f"  Episode {i:>3}: "
            f"{'N/A':>10}" if a_val is None else f"  Episode {i:>3}: {a_val:>10.2f}",
            f"  {'N/A':>10}" if b_val is None else f"  {b_val:>10.2f}",
        )
    print()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _save_comparison_csv(
    stats_list: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """Save a comparison CSV where each row is one run directory.

    Columns match the summary metrics so the file can be read by the same
    ``episode_summary.csv`` pipeline for further analysis.
    """
    fields = [
        "directory", "scenario", "controller", "num_episodes",
        "mean_att", "std_att",
        "mean_throughput", "std_throughput",
        "mean_avg_queue", "std_avg_queue",
        "mean_max_queue",
        "mean_avg_waiting", "std_avg_waiting",
        "mean_total_reward",
        "att_per_episode",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for stats in stats_list:
            row = dict(stats)
            # Serialise the list as a pipe-separated string for CSV compat.
            row["att_per_episode"] = "|".join(f"{v:.4f}" for v in stats["att_per_episode"])
            w.writerow(row)
    print(f"Comparison CSV saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SUMO runner outputs.  Pass one directory for a single-run "
            "summary, or two directories for a side-by-side comparison."
        )
    )
    parser.add_argument(
        "directories",
        nargs="+",
        metavar="DIR",
        help="One or two run output directories to evaluate.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional path to save a comparison CSV.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    if len(args.directories) > 2:
        print(
            "ERROR: At most two directories can be compared at once.",
            file=sys.stderr,
        )
        sys.exit(1)

    stats_list: List[Dict[str, Any]] = []
    for directory in args.directories:
        try:
            stats = compute_stats(directory)
            stats_list.append(stats)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR reading '{directory}': {exc}", file=sys.stderr)
            sys.exit(1)

    if len(stats_list) == 1:
        _print_single(stats_list[0])
    else:
        _print_comparison(stats_list[0], stats_list[1])

    if args.output_csv is not None:
        _save_comparison_csv(stats_list, args.output_csv)
    elif len(stats_list) == 2:
        # Auto-save when two directories are compared and no path was given.
        auto_path = "sumo_comparison.csv"
        _save_comparison_csv(stats_list, auto_path)


if __name__ == "__main__":
    main()
