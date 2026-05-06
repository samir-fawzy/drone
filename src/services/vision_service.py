from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import numpy as np
import cv2
from typing import Any, List, Dict
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor  

# =============================================================================
# Multiprocessing Worker Globals (CPU-heavy YOLO)
# =============================================================================
_process_model = None
_process_weights_path = None
_process_device = None

def _init_vision_worker(weights_path: str, device: str):
    global _process_model, _process_weights_path, _process_device
    _process_weights_path = weights_path
    _process_device = device
    from ultralytics import YOLO
    _process_model = YOLO(weights_path)
    _process_model.predict(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False, device=device)

def _mp_detect_frame(payload: tuple) -> List[Dict[str, Any]]:
    global _process_model, _process_device
    if _process_model is None:
        return []
    frame_bytes, shape, dtype_str, min_conf = payload
    frame = np.frombuffer(frame_bytes, dtype=np.dtype(dtype_str)).copy().reshape(shape)
    results = _process_model(frame, verbose=False, device=_process_device)
    detections: List[Dict[str, Any]] = []
    for result in results:
        names = result.names
        for box in result.boxes:
            conf = float(box.conf.item())
            if conf < min_conf:
                continue
            cls_id = int(box.cls.item())
            coords = box.xyxy[0].tolist()
            detections.append({
                "label": names.get(cls_id, str(cls_id)),
                "confidence": conf,
                "bbox": coords,
            })
    return detections


# =============================================================================
# Async Vision Service
# =============================================================================
class VisionService:
    """Owns a private AirSim camera thread + a process pool for YOLO."""

    def __init__(
        self,
        project_root: Path,
        logger: logging.Logger,
        model_filename: str = "best.pt",
        camera_id: str = "0",
        min_confidence: float = 0.5,
        max_workers: int = 2,
        device: str = "cpu",
    ):
        self.logger = logger
        self.project_root = project_root
        self.camera_id = camera_id
        self.min_confidence = min_confidence
        self.device = device

        self.weights_path = project_root / "models" / model_filename
        self.save_dir = project_root / "data" / "raw_images"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Dedicated thread for AirSim camera RPC (isolated from controller)
        self._airsim_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="airsim_cam")
        self._airsim_client = None

        # Process pool for CPU-heavy YOLO (bypasses Python GIL)
        mp_context = mp.get_context("spawn")
        self._infer_executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=_init_vision_worker,
            initargs=(str(self.weights_path), device),
        )
        self.logger.info(
            "VisionService ready | camera thread: 1 | inference pool: %s workers | device: %s",
            max_workers, device,
        )

    # -------------------------------------------------------------------------
    # AirSim camera helpers (single-threaded)
    # -------------------------------------------------------------------------
    def _init_airsim_sync(self):
        import airsim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        return client

    async def _get_airsim_client(self):
        if self._airsim_client is None:
            loop = asyncio.get_running_loop()
            self._airsim_client = await loop.run_in_executor(
                self._airsim_executor, self._init_airsim_sync
            )
        return self._airsim_client

    def _fetch_frame_sync(self, client, camera_id: str):
        import airsim
        responses = client.simGetImages([
            airsim.ImageRequest(camera_id, airsim.ImageType.Scene, False, False)
        ])
        if not responses:
            return np.array([])
        response = responses[0]
        if not response.image_data_uint8:
            return np.array([])
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        return img1d.reshape(response.height, response.width, 3)

    # -------------------------------------------------------------------------
    # Public async API
    # -------------------------------------------------------------------------
    async def get_frame(self) -> np.ndarray:
        """Async frame fetch. Safe to call in parallel with controller.get_pose()."""
        client = await self._get_airsim_client()
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._airsim_executor, self._fetch_frame_sync, client, self.camera_id
            )
        except Exception as exc:
            self.logger.error("Failed to retrieve frame: %s", exc)
            return np.array([])

    async def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Offload YOLO to process pool."""
        if frame.size == 0:
            return []
        payload = (frame.tobytes(), frame.shape, frame.dtype.str, self.min_confidence)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._infer_executor, _mp_detect_frame, payload)

    async def save_detected_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> None:
        """Non-blocking JPEG encode + disk write."""
        if frame.size == 0 or len(detections) == 0:
            return

        def _encode_and_write():
            debug_frame = frame.copy()
            for obj in detections:
                x1, y1, x2, y2 = map(int, obj["bbox"])
                label = f"{obj['label']} {obj['confidence']:.2f}"
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = self.save_dir / f"detected_{timestamp}.jpg"
            cv2.imwrite(str(file_path), debug_frame)
            return file_path

        file_path = await asyncio.to_thread(_encode_and_write)
        self.logger.debug("Saved detection log: %s", file_path)

    def shutdown(self):
        self._infer_executor.shutdown(wait=True)
        self._airsim_executor.shutdown(wait=True)
        self.logger.info("VisionService shut down.")

    async def save_frame(self,frame:np.array) -> None:
            if frame.size == 0:
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = self.save_dir / f"detected_{timestamp}.jpg"
            cv2.imwrite(str(file_path), frame)