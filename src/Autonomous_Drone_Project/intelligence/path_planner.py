from __future__ import annotations
import logging
from typing import Any
from enum import Enum

class FlightMode(Enum):
    NORMAL = "normal"
    SPRINT = "sprint"

class PathPlanner:
    """Converts perception + RL decisions into physical velocity commands."""

    def __init__(
        self,
        logger: logging.Logger,
        mode: FlightMode = FlightMode.NORMAL,
        default_duration: float = 2.0,
        obstacle_width_threshold: float = 160.0,
        speed_normal: float = 8.0,
        speed_sprint: float = 15.0,
        speed_avoidance: float = 2.0,
        speed_sprint_avoidance: float = 6.0
    ):
        self.logger = logger
        self.mode = mode
        self.default_duration = default_duration
        self.obstacle_width_threshold = obstacle_width_threshold
        self.speed_normal = speed_normal
        self.speed_sprint = speed_sprint
        self.speed_avoidance = speed_avoidance
        self.speed_sprint_avoidance = speed_sprint_avoidance

    def plan_next_move(self, detections: list[dict[str, Any]], rl_action: str) -> dict[str, float]:
        obstacle_ahead = self._is_obstacle_ahead(detections)
        if self.mode == FlightMode.SPRINT:
            return self._plan_sprint(obstacle_ahead, rl_action)
        return self._plan_normal(obstacle_ahead, rl_action)

    def _plan_normal(self, obstacle_ahead: bool, rl_action: str) -> dict[str, float]:
        if obstacle_ahead:
            self.logger.info("Obstacle detected ahead. Executing avoidance.")
            return self._build_command(vx=0.0, vy=self.speed_avoidance, vz=0.0)

        if rl_action == "forward":
            return self._build_command(vx=self.speed_normal, vy=0.0, vz=0.0)
        if rl_action == "left":
            return self._build_command(vx=0.0, vy=-self.speed_avoidance, vz=0.0)
        if rl_action == "right":
            return self._build_command(vx=0.0, vy=self.speed_avoidance, vz=0.0)
        if rl_action == "hover":
            return self._build_command(vx=0.0, vy=0.0, vz=0.0)

        self.logger.debug("Unrecognized RL action '%s'. Hovering.", rl_action)
        return self._build_command(vx=0.0, vy=0.0, vz=0.0)

    def _plan_sprint(self, obstacle_ahead: bool, rl_action: str) -> dict[str, float]:
        if obstacle_ahead:
            self.logger.warning("Obstacle detected! Executing avoidance based on RL.")
            if rl_action == "left":
                return self._build_command(vx=0.0, vy=-self.speed_sprint_avoidance, vz=0.0)
            return self._build_command(vx=0.0, vy=self.speed_sprint_avoidance, vz=0.0)

        self.logger.debug("Path clear. Cruising forward.")
        return self._build_command(vx=self.speed_sprint, vy=0.0, vz=0.0)

    def _is_obstacle_ahead(self, detections: list[dict[str, Any]]) -> bool:
        if not detections:
            return False
        for obj in detections:
            bbox = obj.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 3:
                self.logger.warning("Malformed bounding box data received: %s", bbox)
                continue
            try:
                x1, _, x2, *_ = bbox
                width = max(float(x2) - float(x1), 0.0)
                if width > self.obstacle_width_threshold:
                    return True
            except (ValueError, TypeError) as e:
                self.logger.error("Failed to parse bounding box coordinates %s: %s", bbox, e)
                continue
        return False

    def _build_command(self, vx: float, vy: float, vz: float) -> dict[str, float]:
        return {
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "duration": self.default_duration
        }