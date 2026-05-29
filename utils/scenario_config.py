"""Loader for SUMO blockage scenario JSON files.

JSON format
-----------
.. code-block:: json

    {
      "scenario_name": "accident_westbound",
      "description": "Single vehicle accident blocking lane 0 of west approach at t=300",
      "blockages": [
        {
          "blockage_id": "b1",
          "lane_id": "W2TLS_0",
          "position": 80.0,
          "start_step": 300,
          "end_step": 900,
          "method": "obstacle_vehicle",
          "severity": 1.0
        }
      ]
    }

A null ``end_step`` in JSON maps to Python ``None`` (lasts until episode end).
Lane IDs must match the network naming convention: ``{direction}2TLS_{index}``
(e.g. ``W2TLS_0``, ``N2TLS_1``).

Example scenario files live in ``data/sumo/scenarios/``.
"""

import json
from typing import List, Optional, Tuple

from .blockage_manager import Blockage


def load_scenario(path: str) -> Tuple[str, str, List[Blockage]]:
    """Load a scenario JSON file and parse its blockage definitions.

    Args:
        path: Filesystem path to the scenario ``.json`` file.

    Returns:
        A three-tuple ``(scenario_name, description, blockages)`` where
        *blockages* is a (possibly empty) list of :class:`Blockage` objects.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError:          If required JSON fields are missing.
        ValueError:        If a blockage dict contains invalid values.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    scenario_name: str = data["scenario_name"]
    description: str = data["description"]

    blockages: List[Blockage] = []
    for raw in data.get("blockages", []):
        end_step: Optional[int] = raw["end_step"]  # preserves None for JSON null
        blockages.append(
            Blockage(
                blockage_id=str(raw["blockage_id"]),
                lane_id=str(raw["lane_id"]),
                position=float(raw["position"]),
                start_step=int(raw["start_step"]),
                end_step=int(end_step) if end_step is not None else None,
                method=str(raw["method"]),
                severity=float(raw.get("severity", 1.0)),
                intersection_id=str(raw["intersection_id"]),
            )
        )

    return scenario_name, description, blockages
