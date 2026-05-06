from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

class DroneController:
    """Async wrapper around AirSim with a single dedicated thread for all RPC."""

    def __init__(self, logger):
        self.logger = logger
        self.client = None
        self.connected = False
        # CRITICAL: max_workers=1 guarantees the AirSim client is never accessed
        # concurrently, eliminating msgpack BufferError and Tornado IOLoop races.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="airsim_ctl")

    # -------------------------------------------------------------------------
    # Internal blocking helpers (run exclusively inside self._executor)
    # -------------------------------------------------------------------------
    def _connect_sync(self):
        import airsim  # type: ignore
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        return client

    def _takeoff_sync(self, client):
        client.takeoffAsync().join()

    def _land_sync(self, client):
        client.landAsync().join()

    def _move_sync(self, client, vx, vy, vz, duration):
        client.moveByVelocityAsync(vx, vy, vz, duration).join()

    def _get_pose_sync(self, client):
        state = client.getMultirotorState()
        return {
            "x": state.kinematics_estimated.position.x_val,
            "y": state.kinematics_estimated.position.y_val,
            "z": state.kinematics_estimated.position.z_val,
        }

    def _disconnect_sync(self, client):
        client.armDisarm(False)
        client.enableApiControl(False)

    # -------------------------------------------------------------------------
    # Public async API
    # -------------------------------------------------------------------------
    async def connect(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            self.client = await loop.run_in_executor(self._executor, self._connect_sync)
            self.connected = True
            self.logger.info("Connected to AirSim.")
        except Exception as exc:
            self.logger.warning("AirSim connection unavailable. Running in dry mode: %s", exc)

    async def disconnect(self) -> None:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._disconnect_sync, self.client)
        self.logger.info("Drone disconnected.")

    async def takeoff(self) -> None:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._takeoff_sync, self.client)
        self.logger.info("Takeoff command sent.")

    async def land(self) -> None:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._land_sync, self.client)
        self.logger.info("Land command sent.")

    async def move(self, vx: float, vy: float, vz: float, duration: float) -> None:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor, self._move_sync, self.client, vx, vy, vz, duration
            )
        self.logger.debug("Move command: vx=%.2f vy=%.2f vz=%.2f d=%.2f", vx, vy, vz, duration)

    async def get_pose(self) -> Dict[str, Any]:
        if self.client is not None and self.connected:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._get_pose_sync, self.client)
        return {"x": 0.0, "y": 0.0, "z": 0.0}