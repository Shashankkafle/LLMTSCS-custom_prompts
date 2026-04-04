"""SUMO-based single-intersection traffic signal control environment.

Mirrors the public interface of CityFlowEnv so existing agents can switch
backends by changing a flag rather than rewriting controller code.

Dependencies: traci and sumolib ship with SUMO (eclipse-sumo pip package or
system installation).  Do not import this module until SUMO is installed.
"""

import logging
import os
import socket
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import sumolib
import traci

from .blockage_manager import BlockageManager

logger = logging.getLogger("sumo_env")


class SumoEnv:
    """SUMO backend for a single signalised intersection.

    Wraps TraCI to expose the same reset / step / get_state interface as
    CityFlowEnv.  One *decision step* consists of an optional yellow
    transition followed by green_duration simulation steps.

    Green phase action → SUMO phase index mapping
    (network-specific — update if the tlLogic program in net.xml changes):
        Action 0 → Phase 0  (N-S through, 30 s default)
        Action 1 → Phase 2  (N-S left,    30 s default)
        Action 2 → Phase 4  (E-W through, 30 s default)
        Action 3 → Phase 6  (E-W left,    30 s default)
    Yellow phases interleave at odd indices (1, 3, 5, 7).
    """

    # ASSUMPTION: Phase indices 0, 2, 4, 6 are green phases as defined in
    # data/sumo/single_intersection/net.xml.  Odd indices are yellow.
    # This mapping must be updated if the tlLogic program changes.
    ACTION_TO_SUMO_PHASE: Dict[int, int] = {0: 0, 1: 2, 2: 4, 3: 6}

    def __init__(
        self,
        config: Dict[str, Any],
        blockage_manager: Optional[BlockageManager] = None,
    ) -> None:
        """Initialise SumoEnv.

        Args:
            config: Configuration dict.  Recognised keys:

                sumocfg_path (str):  Path to the .sumocfg file.
                tls_id       (str):  Traffic-light ID in the SUMO network.
                num_steps    (int):  Maximum simulation steps per episode.
                yellow_duration (int):  Steps to hold yellow phase.  Default 3.
                green_duration  (int):  Steps to hold each green phase.  Default 10.
                use_gui      (bool): Launch sumo-gui instead of sumo.  Default False.

            blockage_manager: Optional BlockageManager.  If provided,
                step() calls blockage_manager.step() each simulation step and
                get_state() includes blocked lane IDs.
        """
        self.sumocfg_path: str = config["sumocfg_path"]
        self.tls_id: str = config["tls_id"]
        self.num_steps: int = config["num_steps"]
        self.yellow_duration: int = config.get("yellow_duration", 3)
        self.green_duration: int = config.get("green_duration", 10)
        self.use_gui: bool = config.get("use_gui", False)
        self.blockage_manager: Optional[BlockageManager] = blockage_manager

        # Parse the network file once at construction time.
        net_path = self._resolve_net_path()
        self._net: sumolib.net.Net = sumolib.net.readNet(
            net_path, withInternal=False
        )
        self._approach_lanes: Dict[str, List[str]] = self._build_approach_lanes()
        self._all_lanes: List[str] = [
            lane_id
            for lanes in self._approach_lanes.values()
            for lane_id in lanes
        ]

        # Episode state — reset on each reset() call.
        self._current_action: int = 0
        self._step_count: int = 0
        self._departure_times: Dict[str, int] = {}
        self._travel_times: List[float] = []
        self._arrived_count: int = 0
        self._connected: bool = False

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _resolve_net_path(self) -> str:
        """Extract the net-file path from the .sumocfg XML."""
        tree = ET.parse(self.sumocfg_path)
        root = tree.getroot()
        net_file_elem = root.find(".//net-file")
        if net_file_elem is None:
            raise ValueError(
                f"No <net-file> element found in {self.sumocfg_path}"
            )
        net_value: str = net_file_elem.get("value", "")
        base_dir = os.path.dirname(os.path.abspath(self.sumocfg_path))
        return os.path.join(base_dir, net_value)

    def _build_approach_lanes(self) -> Dict[str, List[str]]:
        """Build direction → lane-ID list from sumolib network.

        Uses the edge naming convention {direction}2TLS for inbound edges
        (e.g. N2TLS, S2TLS).  Lane IDs follow SUMO convention: {edge_id}_{i}.
        """
        direction_to_edge: Dict[str, str] = {
            "N": "N2TLS",
            "S": "S2TLS",
            "E": "E2TLS",
            "W": "W2TLS",
        }
        result: Dict[str, List[str]] = {}
        for direction, edge_id in direction_to_edge.items():
            edge = self._net.getEdge(edge_id)
            result[direction] = [lane.getID() for lane in edge.getLanes()]
        return result

    @staticmethod
    def _find_free_port() -> int:
        """Bind to an ephemeral port, release it, and return the number."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _start_sumo(self) -> None:
        """Launch SUMO subprocess and open a TraCI connection on a random port."""
        binary = "sumo-gui" if self.use_gui else "sumo"
        port = self._find_free_port()
        sumo_cmd = [
            binary,
            "-c", self.sumocfg_path,
            "--remote-port", str(port),
            "--no-step-log",
            "--no-warnings",
        ]
        traci.start(sumo_cmd, port=port)
        self._connected = True
        logger.info(
            "TraCI connected on port %d  (binary=%s, cfg=%s)",
            port, binary, self.sumocfg_path,
        )

    def _update_travel_times(self) -> None:
        """Record departures and arrivals for average-travel-time tracking.

        Called once per simulation step (inside step() and reset()).
        traci.vehicle.getIDList() returns vehicles currently in the network;
        their first appearance is treated as their departure time.
        traci.simulation.getArrivedIDList() returns vehicles that completed
        their trip during the most-recently-executed simulation step.
        """
        for veh_id in traci.vehicle.getIDList():
            if veh_id not in self._departure_times:
                self._departure_times[veh_id] = self._step_count

        for veh_id in traci.simulation.getArrivedIDList():
            if veh_id in self._departure_times:
                travel_time = float(
                    self._step_count - self._departure_times[veh_id]
                )
                self._travel_times.append(travel_time)
                self._arrived_count += 1

    # ------------------------------------------------------------------ #
    #  Public interface                                                     #
    # ------------------------------------------------------------------ #

    def reset(self) -> Dict[str, Any]:
        """Reset the simulation and return the initial state dict.

        Closes any live TraCI connection, starts a fresh SUMO instance, and
        clears all episode-level tracking.

        Returns:
            Initial state dict (see get_state()).
        """
        if self._connected:
            traci.close()
            self._connected = False

        self._current_action = 0
        self._step_count = 0
        self._departure_times = {}
        self._travel_times = []
        self._arrived_count = 0

        if self.blockage_manager is not None:
            self.blockage_manager.clear_all()

        self._start_sumo()
        traci.trafficlight.setPhase(
            self.tls_id, self.ACTION_TO_SUMO_PHASE[0]
        )
        logger.info(
            "Episode started  (num_steps=%d, green_duration=%d, yellow_duration=%d)",
            self.num_steps, self.green_duration, self.yellow_duration,
        )
        return self.get_state()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Advance the simulation by one decision step.

        If the requested action differs from the current phase, a yellow
        transition is applied first.

        Phase-change sequence:
          1. Set yellow phase = ACTION_TO_SUMO_PHASE[current_action] + 1
             (matches the interleaved yellow phases in the tlLogic).
          2. Advance yellow_duration simulation steps.
          3. Set new green phase = ACTION_TO_SUMO_PHASE[action].
          4. Advance green_duration simulation steps.

        If no phase change, steps 1–2 are skipped.

        Args:
            action: Green phase action index (0–3).

        Returns:
            (state, reward, done, info) where reward is negative total
            waiting-vehicle count (same convention as CityFlowEnv).
        """
        if action != self._current_action:
            # ASSUMPTION: yellow phase index = green phase index + 1.
            # Holds for the 8-phase tlLogic where odd indices are yellow.
            yellow_phase = self.ACTION_TO_SUMO_PHASE[self._current_action] + 1
            logger.debug(
                "Phase change %d→%d: applying yellow (SUMO phase %d) for %d steps",
                self._current_action, action, yellow_phase, self.yellow_duration,
            )
            traci.trafficlight.setPhase(self.tls_id, yellow_phase)
            for _ in range(self.yellow_duration):
                traci.simulationStep()
                self._step_count += 1
                self._update_travel_times()
                if self.blockage_manager is not None:
                    self.blockage_manager.step(self._step_count)

            self._current_action = action

        green_phase = self.ACTION_TO_SUMO_PHASE[action]
        traci.trafficlight.setPhase(self.tls_id, green_phase)

        for _ in range(self.green_duration):
            traci.simulationStep()
            self._step_count += 1
            self._update_travel_times()
            if self.blockage_manager is not None:
                self.blockage_manager.step(self._step_count)

        state = self.get_state()
        reward = -float(sum(state["lane_waiting_vehicle_count"].values()))
        done = self._step_count >= self.num_steps

        info: Dict[str, Any] = {
            "step": self._step_count,
            "arrived": self._arrived_count,
            "average_travel_time": self.get_average_travel_time(),
        }

        if done:
            logger.info(
                "Episode ended at step %d  arrived=%d  ATT=%.2f s",
                self._step_count,
                self._arrived_count,
                info["average_travel_time"],
            )

        return state, reward, done, info

    def get_state(self) -> Dict[str, Any]:
        """Return the current simulation state.

        Returns:
            Dict with keys:
              lane_vehicle_count        : {lane_id: int}
              lane_waiting_vehicle_count: {lane_id: int}
              lane_vehicle_speed        : {lane_id: float}  mean speed in m/s;
                                          0.0 when the lane is empty
              current_phase             : int  action index (not SUMO phase index)
              blocked_lanes             : list[str]  lane IDs with active blockages;
                                          empty list when no blockage_manager
        """
        lane_vehicle_count: Dict[str, int] = {}
        lane_waiting_vehicle_count: Dict[str, int] = {}
        lane_vehicle_speed: Dict[str, float] = {}

        for lane_id in self._all_lanes:
            count = traci.lane.getLastStepVehicleNumber(lane_id)
            waiting = traci.lane.getLastStepHaltingNumber(lane_id)
            mean_speed = (
                traci.lane.getLastStepMeanSpeed(lane_id) if count > 0 else 0.0
            )
            lane_vehicle_count[lane_id] = count
            lane_waiting_vehicle_count[lane_id] = waiting
            lane_vehicle_speed[lane_id] = mean_speed

        blocked_lanes: List[str] = (
            self.blockage_manager.get_blocked_lane_ids()
            if self.blockage_manager is not None
            else []
        )

        return {
            "lane_vehicle_count": lane_vehicle_count,
            "lane_waiting_vehicle_count": lane_waiting_vehicle_count,
            "lane_vehicle_speed": lane_vehicle_speed,
            "current_phase": self._current_action,
            "blocked_lanes": blocked_lanes,
        }

    def get_average_travel_time(self) -> float:
        """Mean travel time of all vehicles that completed their trip this episode.

        Travel time is measured in simulation steps (== seconds for step-length 1.0).

        Returns:
            Mean travel time, or 0.0 if no vehicles have arrived yet.
        """
        if not self._travel_times:
            return 0.0
        return sum(self._travel_times) / len(self._travel_times)

    def close(self) -> None:
        """Close the TraCI connection and terminate the SUMO subprocess."""
        if self._connected:
            traci.close()
            self._connected = False

    # ------------------------------------------------------------------ #
    #  Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def approach_lanes(self) -> Dict[str, List[str]]:
        """Lane IDs grouped by approach direction.

        Returns:
            {'N': ['N2TLS_0', 'N2TLS_1'],
             'S': ['S2TLS_0', 'S2TLS_1'],
             'E': ['E2TLS_0', 'E2TLS_1'],
             'W': ['W2TLS_0', 'W2TLS_1']}
        """
        return self._approach_lanes
