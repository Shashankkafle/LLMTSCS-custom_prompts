"""Blockage-aware LLM prompt for SUMO single-intersection control.

Follows the same structure and ReAct-style reasoning format as
``utils/my_utils.py::getPrompt``:

  * A system message establishing the expert role.
  * A user message that describes the intersection layout, presents the
    current per-lane vehicle and wait counts, lists any active lane
    blockages, and asks the LLM to choose the best green-phase action.

Response tag: ``<phase>N</phase>`` where N is the integer action index (0–3).
This mirrors the ``<signal>ETWT</signal>`` tag used in the CityFlow prompts so
the runner can apply the same ``re.findall`` extraction pattern.

Phase reference (must stay in sync with net.xml / sumo_env.py):
    Action 0 → Phase 0  N-S through  lanes N2TLS_0, S2TLS_0
    Action 1 → Phase 2  N-S left     lanes N2TLS_1, S2TLS_1
    Action 2 → Phase 4  E-W through  lanes E2TLS_0, W2TLS_0
    Action 3 → Phase 6  E-W left     lanes E2TLS_1, W2TLS_1

Do NOT import this module in code paths that run without active blockages.
The runner handles the fallback (logs a WARNING and uses the standard waittime
prompt) before ever reaching this module.
"""

import json
import os
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Phase metadata – keep in sync with net.xml / SumoEnv.ACTION_TO_SUMO_PHASE
# ---------------------------------------------------------------------------

_PHASE_LABELS = {
    0: "N-S through  (lanes: N2TLS_0, S2TLS_0)",
    1: "N-S left     (lanes: N2TLS_1, S2TLS_1)",
    2: "E-W through  (lanes: E2TLS_0, W2TLS_0)",
    3: "E-W left     (lanes: E2TLS_1, W2TLS_1)",
}

# Lanes served by each green action (inbound lanes only; outbound not shown).
_PHASE_LANES = {
    0: ["N2TLS_0", "S2TLS_0"],
    1: ["N2TLS_1", "S2TLS_1"],
    2: ["E2TLS_0", "W2TLS_0"],
    3: ["E2TLS_1", "W2TLS_1"],
}

# Lane names grouped by approach direction for the intersection description.
_APPROACH_LANES = {
    "N": ["N2TLS_0 (through)", "N2TLS_1 (left-turn)"],
    "S": ["S2TLS_0 (through)", "S2TLS_1 (left-turn)"],
    "E": ["E2TLS_0 (through)", "E2TLS_1 (left-turn)"],
    "W": ["W2TLS_0 (through)", "W2TLS_1 (left-turn)"],
}

# Absolute path to the system-prompt JSON (same directory as this file).
_SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt_commonsense.json")


def _load_system_prompt() -> str:
    """Load the system prompt string from ``prompt_commonsense.json``."""
    with open(_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["system_prompt"]


# ---------------------------------------------------------------------------
# State-text helpers (mirror state2text in my_utils.py)
# ---------------------------------------------------------------------------

def _state_to_text(
    lane_vehicle_count: Dict[str, int],
    lane_waiting_count: Dict[str, int],
) -> str:
    """Render per-phase vehicle and queue counts as a human-readable block.

    Mirrors the ``state2text`` function in ``utils/my_utils.py``.  For each
    action (0–3) we show the total vehicle count and waiting count across the
    two lanes served by that phase.

    Args:
        lane_vehicle_count:  {lane_id: int} total vehicles from get_state().
        lane_waiting_count:  {lane_id: int} waiting vehicles from get_state().

    Returns:
        Multi-line string ready to embed in the user prompt.
    """
    lines: List[str] = []
    for action, label in _PHASE_LABELS.items():
        lanes = _PHASE_LANES[action]
        total_veh = sum(lane_vehicle_count.get(ln, 0) for ln in lanes)
        total_wait = sum(lane_waiting_count.get(ln, 0) for ln in lanes)
        per_lane_parts = []
        for ln in lanes:
            per_lane_parts.append(
                f"{ln}: {lane_waiting_count.get(ln, 0)} waiting / "
                f"{lane_vehicle_count.get(ln, 0)} total"
            )
        lines.append(
            f"Phase {action} — {label}\n"
            f"  Early queued (waiting): {total_wait} (Total)\n"
            f"  Approaching (moving):   {total_veh - total_wait} (Total)\n"
            f"  Per-lane: {'; '.join(per_lane_parts)}\n"
        )
    return "\n".join(lines)


def _blockage_to_text(blockages: List[Any]) -> str:
    """Render active blockage objects as a human-readable block.

    Each *blockage* object is expected to expose ``lane_id``, ``method``,
    ``position``, and ``severity`` attributes (matching the ``Blockage``
    dataclass from ``utils/blockage_manager.py``).

    Returns:
        Multi-line string, or ``"None"`` when *blockages* is empty.
    """
    if not blockages:
        return "None"
    lines: List[str] = []
    for b in blockages:
        if b.method == "obstacle_vehicle":
            method_note = "stopped vehicle — full blockage"
        else:
            pct = int(b.severity * 100)
            method_note = f"speed restriction — {pct}% reduction"
        lines.append(
            f"  • Lane {b.lane_id}  @ {b.position:.1f} m  [{method_note}]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_blockage_prompt(
    lane_vehicle_count: Dict[str, int],
    lane_waiting_count: Dict[str, int],
    active_blockages: List[Any],
    current_phase: int,
) -> List[Dict[str, str]]:
    """Build the blockage-aware LLM prompt.

    Constructs a two-message list (system + user) matching the format
    returned by ``utils/my_utils.py::getPrompt``.  The caller converts
    this list to a flat string before tokenisation.

    Args:
        lane_vehicle_count: Total vehicle counts per lane from
            ``SumoEnv.get_state()``.
        lane_waiting_count: Waiting vehicle counts per lane from
            ``SumoEnv.get_state()``.
        active_blockages:   List of :class:`~utils.blockage_manager.Blockage`
            objects currently active (may be empty; caller ensures this is
            only called when blockages are relevant).
        current_phase:      Current green-phase action index (0–3).

    Returns:
        ``[{"role": "system", "content": ...}, {"role": "user", "content": ...}]``
        mirroring ``getPrompt`` in ``utils/my_utils.py``.
    """
    system_content = _load_system_prompt()

    # Build intersection description (same verbosity style as getPrompt).
    approach_desc_parts: List[str] = []
    for direction, lane_names in _APPROACH_LANES.items():
        approach_desc_parts.append(
            f"    {direction} approach: {', '.join(lane_names)}"
        )
    approach_desc = "\n".join(approach_desc_parts)

    # Phase definitions (mirror the "Relieves:" lines in state2text).
    phase_def_parts: List[str] = []
    for action, label in _PHASE_LABELS.items():
        phase_def_parts.append(f"    Phase {action}: {label}")
    phase_def = "\n".join(phase_def_parts)

    state_txt = _state_to_text(lane_vehicle_count, lane_waiting_count)
    blockage_txt = _blockage_to_text(active_blockages)

    user_content = (
        "A traffic light regulates a four-approach intersection with Northern, Southern, "
        "Eastern, and Western sections.  Each approach has two inbound lanes: lane 0 "
        "(through traffic) and lane 1 (left-turn traffic).  Lane IDs follow the convention "
        "{direction}2TLS_{index} (e.g. W2TLS_0).\n\n"
        "Intersection layout:\n"
        f"{approach_desc}\n\n"
        "Signal phases available (choose one action index 0–3):\n"
        f"{phase_def}\n\n"
        f"Current active phase: {current_phase}\n\n"
        "Current traffic state:\n"
        f"{state_txt}\n"
        "LANE BLOCKAGE CONTEXT\n"
        "The following blockages are currently active and may restrict vehicle flow:\n"
        f"{blockage_txt}\n\n"
        "Please answer:\n"
        "Which phase action (0, 1, 2, or 3) will most significantly improve traffic "
        "conditions during the next signal cycle, taking into account both congestion "
        "levels and any active lane blockages?\n\n"
        "Requirements:\n"
        "- Let's think step by step.\n"
        "- You can only choose one of the phase actions: 0, 1, 2, or 3.\n"
        "- You must follow these steps: "
        "Step 1: Analyse the current traffic state and blockage context. "
        "Step 2: State your chosen phase action.\n"
        "- Your choice can only be given after finishing the analysis.\n"
        "- Your choice must be identified by the tag: <phase>YOUR_CHOICE</phase>."
    )

    return [
        {"role": "system",  "content": system_content},
        {"role": "user",    "content": user_content},
    ]


def get_waittime_prompt(
    lane_vehicle_count: Dict[str, int],
    lane_waiting_count: Dict[str, int],
    current_phase: int,
) -> List[Dict[str, str]]:
    """Build the standard wait-time LLM prompt (no blockage context).

    Used as the fallback when ``--prompt_type blockage_aware`` is requested
    but the current scenario has no active blockages.  Mirrors the
    ``getPrompt`` style with ``<phase>`` response tags for SUMO.

    Args:
        lane_vehicle_count: Total vehicle counts per lane.
        lane_waiting_count: Waiting vehicle counts per lane.
        current_phase:      Current green-phase action index (0–3).

    Returns:
        Two-message list matching the format of ``get_blockage_prompt``.
    """
    system_content = _load_system_prompt()

    phase_def_parts: List[str] = []
    for action, label in _PHASE_LABELS.items():
        phase_def_parts.append(f"    Phase {action}: {label}")
    phase_def = "\n".join(phase_def_parts)

    state_txt = _state_to_text(lane_vehicle_count, lane_waiting_count)

    user_content = (
        "A traffic light regulates a four-approach intersection. "
        "Each approach has two inbound lanes: lane 0 (through) and lane 1 (left-turn).\n\n"
        "Signal phases available (choose one action index 0–3):\n"
        f"{phase_def}\n\n"
        f"Current active phase: {current_phase}\n\n"
        "Current traffic state (early queued = vehicles waiting at stop line):\n"
        f"{state_txt}\n"
        "Please answer:\n"
        "Which phase action (0, 1, 2, or 3) will most significantly improve traffic "
        "conditions during the next signal cycle?\n\n"
        "Requirements:\n"
        "- Let's think step by step.\n"
        "- You can only choose one of the phase actions: 0, 1, 2, or 3.\n"
        "- You must follow these steps: "
        "Step 1: Analyse the queue lengths and waiting vehicles for each phase. "
        "Step 2: State your chosen phase action.\n"
        "- Your choice can only be given after finishing the analysis.\n"
        "- Your choice must be identified by the tag: <phase>YOUR_CHOICE</phase>."
    )

    return [
        {"role": "system",  "content": system_content},
        {"role": "user",    "content": user_content},
    ]


def get_commonsense_prompt(
    lane_vehicle_count: Dict[str, int],
    lane_waiting_count: Dict[str, int],
    current_phase: int,
) -> List[Dict[str, str]]:
    """Commonsense-style prompt without wait-time or blockage detail.

    Mirrors the ``getPrompt`` / ``prompt_commonsense.json`` style from the
    CityFlow pipeline, adapted for SUMO phase indices.

    Args:
        lane_vehicle_count: Total vehicle counts per lane.
        lane_waiting_count: Waiting vehicle counts per lane.
        current_phase:      Current green-phase action index (0–3).

    Returns:
        Two-message list matching the format of ``get_blockage_prompt``.
    """
    system_content = _load_system_prompt()

    phase_def_parts: List[str] = []
    for action, label in _PHASE_LABELS.items():
        phase_def_parts.append(f"    Phase {action}: {label}")
    phase_def = "\n".join(phase_def_parts)

    state_txt = _state_to_text(lane_vehicle_count, lane_waiting_count)

    user_content = (
        "A traffic light regulates a four-approach intersection. "
        "Each approach has two inbound lanes: lane 0 (through) and lane 1 (left-turn). "
        "The intersection layout and available signal phases are described below.\n\n"
        "Signal phases available (choose one action index 0–3):\n"
        f"{phase_def}\n\n"
        f"Current active phase: {current_phase}\n\n"
        "Traffic state:\n"
        f"{state_txt}\n"
        "Please answer:\n"
        "Which is the most effective traffic signal that will most significantly "
        "improve the traffic condition during the next phase?\n\n"
        "Requirements:\n"
        "- Let's think step by step.\n"
        "- You can only choose one of the phase actions: 0, 1, 2, or 3.\n"
        "- You must follow these steps: "
        "Step 1: Provide your analysis for identifying the optimal signal. "
        "Step 2: Answer your chosen phase action.\n"
        "- Your choice can only be given after finishing the analysis.\n"
        "- Your choice must be identified by the tag: <phase>YOUR_CHOICE</phase>."
    )

    return [
        {"role": "system",  "content": system_content},
        {"role": "user",    "content": user_content},
    ]
