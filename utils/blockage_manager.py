"""Lane blockage injection for SUMO simulations.

Provides BlockageManager, which activates and deactivates lane blockages on a
pre-defined schedule during a running TraCI session.  Two methods are
supported:

  obstacle_vehicle  — inserts a stationary vehicle that physically blocks the
                      lane.  Requires a two-step insertion process because SUMO
                      does not place newly-added vehicles until after the next
                      simulationStep() call.

  speed_restriction — reduces the lane's max speed (down to 0 at severity 1.0).
                      Immediately effective; original speed restored on removal.

This module requires traci (ships with SUMO).  Import only after SUMO is
installed and a TraCI session is active.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import traci

logger = logging.getLogger("blockage_manager")


@dataclass
class Blockage:
    """Definition of a single lane blockage event.

    Attributes:
        blockage_id: Unique string identifier.
        lane_id:     SUMO lane ID (e.g. ``"W2TLS_0"``).
        position:    Position along the lane in metres from the lane start.
        start_step:  Simulation step at which the blockage becomes active.
        end_step:    Simulation step at which the blockage is removed.
                     ``None`` means the blockage lasts until the episode ends.
        method:      ``"obstacle_vehicle"`` or ``"speed_restriction"``.
        severity:    0.0–1.0.  Only used by ``speed_restriction``; 1.0 = full
                     block (speed set to 0), <1.0 = partial speed reduction.
                     Ignored by ``obstacle_vehicle``.
    """

    blockage_id: str
    lane_id: str
    position: float
    start_step: int
    end_step: Optional[int]
    method: str
    severity: float = 1.0


class BlockageManager:
    """Manages a schedule of lane blockages in a live SUMO simulation.

    Typical usage::

        manager = BlockageManager(blockage_schedule)
        env = SumoEnv(config, blockage_manager=manager)
        state = env.reset()          # calls manager.clear_all()
        while not done:
            state, reward, done, info = env.step(action)
            # env.step() calls manager.step() each simulation step internally
    """

    def __init__(self, blockage_schedule: List[Blockage]) -> None:
        """Store the blockage schedule and initialise internal state.

        Args:
            blockage_schedule: Ordered list of Blockage objects.  Multiple
                blockages may share the same lane_id.
        """
        self._schedule: List[Blockage] = blockage_schedule
        # Currently active blockages, keyed by blockage_id.
        self._active: Dict[str, Blockage] = {}
        # obstacle_vehicle: vehicles pending moveTo in the next step.
        # Value is True while moveTo has not yet been applied.
        # ASSUMPTION: traci.vehicle.add() queues the vehicle; it only enters
        # the simulation after the next simulationStep().  Therefore moveTo
        # must be called in the step *after* add().  See _activate_obstacle().
        self._pending_position: Dict[str, bool] = {}
        # speed_restriction: original lane speeds to restore on deactivation.
        self._original_speeds: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def step(self, current_step: int) -> None:
        """Process blockage activations, positioning, and deactivations.

        Must be called once per simulation step, *after* traci.simulationStep().

        Args:
            current_step: The simulation step number that just completed.
        """
        for blockage in self._schedule:
            bid = blockage.blockage_id

            # Activate at start_step.
            if current_step == blockage.start_step and bid not in self._active:
                self._activate(blockage)

            # Finish obstacle vehicle positioning (two-step insertion).
            # This block handles the second step: after simulationStep() has
            # placed the vehicle in the network we can call moveTo/setSpeed.
            veh_id = f"obstacle_{bid}"
            if self._pending_position.get(veh_id, False):
                # ASSUMPTION: SUMO does not process newly inserted vehicles
                # until after simulationStep() is called.  moveTo/setSpeed must
                # therefore be deferred to the call to step() that follows the
                # first simulationStep() after traci.vehicle.add().
                try:
                    traci.vehicle.moveTo(veh_id, blockage.lane_id, blockage.position)
                    traci.vehicle.setSpeed(veh_id, 0.0)
                    # SpeedMode 0 disables all speed influencing (safe speed,
                    # traffic-light braking, etc.), keeping the vehicle stationary.
                    traci.vehicle.setSpeedMode(veh_id, 0)
                    # Prevent automatic lane-change behaviour.
                    traci.vehicle.setLaneChangeMode(veh_id, 0)
                    self._pending_position[veh_id] = False
                    logger.debug(
                        "Obstacle %s positioned on %s at %.1f m",
                        veh_id, blockage.lane_id, blockage.position,
                    )
                except traci.TraCIException:
                    # Vehicle not yet available in the network; retry next step.
                    pass

            # Deactivate at end_step.
            if (
                bid in self._active
                and blockage.end_step is not None
                and current_step >= blockage.end_step
            ):
                self._deactivate(blockage)

    def get_active_blockages(self) -> List[Blockage]:
        """Return a list of all currently active Blockage objects."""
        return list(self._active.values())

    def get_blocked_lane_ids(self) -> List[str]:
        """Return lane IDs for all currently active blockages."""
        return [b.lane_id for b in self._active.values()]

    def clear_all(self) -> None:
        """Remove every active blockage and restore original lane conditions.

        Called by SumoEnv.reset() to guarantee a clean state at the start of
        each episode.  Safe to call when no TraCI session is active (e.g.
        before the first reset).
        """
        for blockage in list(self._active.values()):
            try:
                self._deactivate(blockage)
            except traci.TraCIException:
                pass
        self._active.clear()
        self._pending_position.clear()
        self._original_speeds.clear()

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _activate(self, blockage: Blockage) -> None:
        """Activate a blockage according to its method."""
        self._active[blockage.blockage_id] = blockage
        logger.debug(
            "Activating blockage '%s' on lane %s  method=%s  severity=%.2f",
            blockage.blockage_id, blockage.lane_id,
            blockage.method, blockage.severity,
        )
        if blockage.method == "obstacle_vehicle":
            self._activate_obstacle_vehicle(blockage)
        elif blockage.method == "speed_restriction":
            self._activate_speed_restriction(blockage)
        else:
            raise ValueError(
                f"Unknown blockage method '{blockage.method}'. "
                "Expected 'obstacle_vehicle' or 'speed_restriction'."
            )

    def _activate_obstacle_vehicle(self, blockage: Blockage) -> None:
        """Insert a stationary vehicle to physically block the lane.

        Two-step process (required by SUMO's insertion model):

        Step 1  (this call): Add the vehicle to SUMO's pending-insertion queue
                             via traci.vehicle.add().  SUMO will not place it
                             until simulationStep() is called.

        Step 2  (next step()): After simulationStep() has inserted the vehicle,
                               call moveTo/setSpeed/setSpeedMode to freeze it at
                               the target position.  This is handled in step().
        """
        bid = blockage.blockage_id
        # Derive the edge ID from the lane ID by stripping the trailing _{index}.
        edge_id = blockage.lane_id.rsplit("_", 1)[0]
        route_id = f"obstacle_route_{bid}"
        veh_id = f"obstacle_{bid}"

        traci.route.add(route_id, [edge_id])
        traci.vehicle.add(
            vehID=veh_id,
            routeID=route_id,
            typeID="DEFAULT_VEHTYPE",
            depart="now",
            departLane="first",
            departPos="0",
            departSpeed="0",
        )
        # Flag for deferred positioning in the next call to step().
        self._pending_position[veh_id] = True
        logger.debug(
            "obstacle_vehicle %s inserted on edge %s (lane %s); "
            "positioning deferred to next step",
            veh_id, edge_id, blockage.lane_id,
        )

    def _activate_speed_restriction(self, blockage: Blockage) -> None:
        """Reduce lane max speed to simulate a partial or full blockage.

        At severity=1.0 the new max speed is 0.0 (full block).
        At severity<1.0 the speed is reduced proportionally:
            new_speed = original_speed * (1 - severity)
        """
        lane_id = blockage.lane_id
        original = traci.lane.getMaxSpeed(lane_id)
        self._original_speeds[lane_id] = original
        new_speed = original * (1.0 - blockage.severity)
        traci.lane.setMaxSpeed(lane_id, new_speed)
        logger.debug(
            "speed_restriction on %s: %.2f → %.2f m/s  (severity=%.2f)",
            lane_id, original, new_speed, blockage.severity,
        )

    def _deactivate(self, blockage: Blockage) -> None:
        """Remove a blockage and restore normal lane conditions."""
        bid = blockage.blockage_id
        logger.debug("Deactivating blockage '%s'", bid)

        if blockage.method == "obstacle_vehicle":
            veh_id = f"obstacle_{bid}"
            try:
                traci.vehicle.remove(veh_id)
            except traci.TraCIException:
                pass
            self._pending_position.pop(veh_id, None)

        elif blockage.method == "speed_restriction":
            lane_id = blockage.lane_id
            if lane_id in self._original_speeds:
                traci.lane.setMaxSpeed(lane_id, self._original_speeds.pop(lane_id))

        self._active.pop(bid, None)
