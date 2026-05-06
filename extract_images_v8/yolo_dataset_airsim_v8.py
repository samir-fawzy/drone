"""
AirSim Segmentation YOLO Extractor — V8 (Async + Multiprocessing)
=======================================================================
Changes over V7:
    ASYNC   — position_drone, capture_images, AirSim API calls wrapped
              in asyncio so the drone moves + waits without blocking the
              main thread.
    MULTIPROCESSING — extract_yolo_labels_multi_class, save_debug_image,
              save_frame_metadata, cv2.imwrite all run in a
              ProcessPoolExecutor so CPU-heavy image work runs in
              parallel with the next drone move.

Architecture:
    Main loop (asyncio event loop)
        └─► async drone_capture_loop()          ← non-blocking AirSim I/O
                └─► asyncio.get_event_loop()
                        .run_in_executor(POOL)  ← CPU work offloaded
                            └─► _process_frame_worker()   (child process)
                                    ├─ extract_yolo_labels_multi_class()
                                    ├─ cv2.imwrite (rgb + debug)
                                    └─- save_frame_metadata()

Version : 8.0.0
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import airsim
import cv2
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed

# ---------------------------------------------------------------------------
# 0. Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
DATASET_DIR: Path = Path("dataset")
IMAGES_DIR:  Path = DATASET_DIR / "images"
LABELS_DIR:  Path = DATASET_DIR / "labels"
DEBUG_DIR:   Path = DATASET_DIR / "debug"
META_DIR:    Path = DATASET_DIR / "metadata"


def setup_directories() -> None:
    for d in (IMAGES_DIR, LABELS_DIR, DEBUG_DIR, META_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Enums & Config
# ---------------------------------------------------------------------------
class YoloClassID(IntEnum):
    VEHICLE  = 0
    PERSON   = 1
    BUILDING = 2
    LAMPPOST = 3
    TREE     = 4


@dataclass(frozen=True)
class ClassConfig:
    name: str
    yolo_id: YoloClassID
    regex_patterns: Tuple[str, ...]
    search_keywords: Tuple[str, ...]
    color_tolerance: int = 8
    approx_size_m: float = 4.5
    min_aspect_ratio: float = 0.3
    max_aspect_ratio: float = 2.2
    min_fill_ratio: float = 0.35
    merge_px: int = 0


@dataclass(frozen=True)
class FlightSettings:
    camera_name: str = "0"
    altitudes_m: Tuple[float, ...] = (5.0, 15.0 ,20 ,30.0)
    min_orbit_radius_m: float = 5.0
    cam_offset: airsim.Vector3r = field(
        default_factory=lambda: airsim.Vector3r(0.8, 0.0, 0.3)
    )
    pitch_max_deg: float = -70.0
    post_shutter_delay_s: float = 1.0

    @property
    def airsim_altitudes_ned(self) -> Tuple[float, ...]:
        return tuple(-a for a in self.altitudes_m)

    @property
    def pitch_max_rad(self) -> float:
        return math.radians(self.pitch_max_deg)


@dataclass(frozen=True)
class VisionSettings:
    max_objects_per_class: int  = 10000
    min_contour_area_px: int    = 15
    min_norm_area: float        = 0.00003
    max_norm_area: float        = 0.65
    nms_iou_threshold: float    = 0.45
    morph_open_kernel: int      = 3
    morph_close_kernel: int     = 3
    close_altitude_threshold_m: float = 10.0


@dataclass(frozen=True)
class CameraPose:
    cx: float; cy: float; cz: float
    yaw_rad: float; pitch_rad: float


# ---------------------------------------------------------------------------
# 3. Global Registries
# ---------------------------------------------------------------------------
CLASS_CONFIGS: Tuple[ClassConfig, ...] = (
    ClassConfig(
        name="vehicle", yolo_id=YoloClassID.VEHICLE,
        regex_patterns=(
            ".*saloon.*",".*hatchback.*",".*suv.*",".*sedan.*",".*coupe.*",
            ".*Vehicle_.*",".*Car_.*",".*Auto.*",".*Truck.*",".*Van.*"),
        search_keywords=("saloon","hatchback","suv","sedan","coupe",
                         "vehicle_","car_","auto","truck","van"),
        color_tolerance=15, approx_size_m=4.5,
        min_aspect_ratio=0.4, max_aspect_ratio=3.5, min_fill_ratio=0.35,
        merge_px=0,
    ),
    ClassConfig(
        name="person", yolo_id=YoloClassID.PERSON,
        regex_patterns=(
            ".*person.*",".*pedestrian.*",".*Walker.*",".*Character.*",
            ".*Man.*",".*Woman.*",".*NPC.*",".*Civilian.*",".*Ped.*",
        ),
        search_keywords=("person","pedestrian","walker","character",
                         "man","woman","npc","civilian","ped"),
        color_tolerance=15, approx_size_m=0.8,
        min_aspect_ratio=0.15, max_aspect_ratio=5.0, min_fill_ratio=0.20,
        merge_px=0,
    ),
    ClassConfig(
        name="building", yolo_id=YoloClassID.BUILDING,
        regex_patterns=(
            ".*Apartment.*",".*BG_Building.*",".*Building_.*",".*House_.*",
            ".*Skyscraper.*",".*Gas_Station.*",".*Shop.*",".*Store.*",
            ".*Office.*",".*Warehouse.*",".*Mall.*",".*Hospital.*",".*School.*",
        ),
        search_keywords=("apartment","bg_building","building_","house_",
                         "skyscraper","gas_station","shop","store",
                         "office","warehouse","mall","hospital","school"),
        color_tolerance=15, approx_size_m=12.0,
        min_aspect_ratio=0.3, max_aspect_ratio=3.0, min_fill_ratio=0.30,
        merge_px=0,
    ),
    ClassConfig(
        name="lamppost", yolo_id=YoloClassID.LAMPPOST,
        regex_patterns=(
            ".*BP_LightPost.*",".*Lighting_Pole.*",".*prp_streetLight.*",
            ".*StreetLight.*",".*LightPole.*",".*LampPost.*",".*Pole.*",
            ".*Street_Light.*",".*Light_.*",
        ),
        search_keywords=("bp_lightpost","lighting_pole","prp_streetlight",
                         "streetlight","lightpole","lamppost","pole",
                         "street_light","light_"),
        color_tolerance=15, approx_size_m=2.0,
        min_aspect_ratio=0.05, max_aspect_ratio=8.0, min_fill_ratio=0.10,
        merge_px=4,
    ),
    ClassConfig(
        name="tree", yolo_id=YoloClassID.TREE,
        regex_patterns=(
            ".*Birch.*",".*Oak.*",".*SM_Tree.*",".*Tree*",".*flg_tree.*",
            ".*Foliage.*",".*Pine.*",".*Maple.*",".*Shrub.*",
           ".*Greenery.*",
        ),
        search_keywords=("birch","oak","sm_tree","tree_","flg_tree",
                         "foliage","pine","maple","shrub","greenery"),
        color_tolerance=15, approx_size_m=8.0,
        min_aspect_ratio=0.25, max_aspect_ratio=4.0, min_fill_ratio=0.15,
        merge_px=8,
    ),
)

FLIGHT_CFG = FlightSettings()
VISION_CFG  = VisionSettings()

# ---------------------------------------------------------------------------
# 4. ProcessPoolExecutor — shared across the whole run
#    cpu_count() - 1  so AirSim / Unreal still gets a core.
# ---------------------------------------------------------------------------
_CPU_WORKERS = max(1, (os.cpu_count() or 2) - 1)
# Created once in main(), passed down.  Not a module-level global so child
# processes don't accidentally inherit an open pool.
PROCESS_POOL: concurrent.futures.ProcessPoolExecutor | None = None


# ---------------------------------------------------------------------------
# 5. Engine Fixes & Navigation
# ---------------------------------------------------------------------------
def fix_unreal_rendering(client: airsim.MultirotorClient) -> None:
    logger.info("Applying Unreal Engine render fixes …")
    for cmd in (
        "r.MotionBlurQuality 0","r.PostProcessAAQuality 0",
        "r.EyeAdaptationQuality 0","r.ContactShadows 0","r.Tonemapper.Quality 0",
    ):
        client.simRunConsoleCommand(cmd)
    time.sleep(0.5)
r"""
$$\text{Valid}(P) = \begin{cases}
\text{False}, & \text{if } (|x| \le \epsilon) \land (|y| \le \epsilon) \land (|z| \le \epsilon) \
\text{False}, & \text{if } x \notin \mathbb{R} \lor y \notin \mathbb{R} \lor z \notin \mathbb{R} \
\text{True}, & \text{otherwise}
\end{cases}$$
"""

def is_valid_pose(pose: airsim.Pose) -> bool:
    p = pose.position
    if (math.isclose(p.x_val,0.0,abs_tol=0.01)
            and math.isclose(p.y_val,0.0,abs_tol=0.01)
            and math.isclose(p.z_val,0.0,abs_tol=0.01)):
        return False
    return all(math.isfinite(v) for v in (p.x_val, p.y_val, p.z_val))


def compute_camera_orbit(
    target_pos: airsim.Vector3r, altitude: float, obj_size_m: float
) -> CameraPose:
    abs_alt = abs(altitude)
    min_r   = max(FLIGHT_CFG.min_orbit_radius_m, obj_size_m * 1.2)
    radius  = random.uniform(
        max(min_r, abs_alt * 0.7),
        max(min_r + 7.0, abs_alt * 3.5),
    )
    angle = random.uniform(0.0, 2.0 * math.pi)
    cx, cy, cz = (
        target_pos.x_val + radius * math.cos(angle),
        target_pos.y_val + radius * math.sin(angle),
        altitude,
    )
    dx, dy, dz = target_pos.x_val - cx, target_pos.y_val - cy, target_pos.z_val - cz
    yaw   = math.atan2(dy, dx)
    pitch = min(
        -math.atan2(abs(dz), max(math.hypot(dx, dy), 0.001)),
        FLIGHT_CFG.pitch_max_rad,
    )
    return CameraPose(cx, cy, cz, yaw, pitch)


# ---------------------------------------------------------------------------
# 6. ASYNC — Drone positioning & image capture
#    Why async here:
#      simSetVehiclePose / simSetCameraPose are blocking HTTP calls to AirSim.
#      Wrapping them in run_in_executor(None) (thread pool) lets the event
#      loop stay responsive — especially useful when we later await the
#      ProcessPool future for the previous frame at the same time.
# ---------------------------------------------------------------------------
async def async_position_drone(
    client: airsim.MultirotorClient,
    pose: CameraPose,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """
    Non-blocking drone + camera positioning.
    AirSim calls are synchronous C++ RPC — run them in the default
    ThreadPoolExecutor so the asyncio loop is never blocked.
    """
    vpos = airsim.Vector3r(
        pose.cx - FLIGHT_CFG.cam_offset.x_val,
        pose.cy - FLIGHT_CFG.cam_offset.y_val,
        pose.cz - FLIGHT_CFG.cam_offset.z_val,
    )

    def _set_poses():
        client.simSetVehiclePose(
            airsim.Pose(vpos, airsim.to_quaternion(0.0, 0.0, 0.0)),
            ignore_collision=True,
        )
        client.simSetCameraPose(
            FLIGHT_CFG.camera_name,
            airsim.Pose(
                FLIGHT_CFG.cam_offset,
                airsim.to_quaternion(pose.pitch_rad, 0.0, pose.yaw_rad),
            ),
        )

    await loop.run_in_executor(None, _set_poses)
    # Non-blocking sleep — releases the event loop during the shutter delay
    await asyncio.sleep(FLIGHT_CFG.post_shutter_delay_s)


@retry(stop=stop_after_attempt(5), wait=wait_fixed(0.5), reraise=True)
def _capture_sync(client: airsim.MultirotorClient) -> Tuple[np.ndarray, np.ndarray]:
    """Synchronous capture — called from a thread via run_in_executor."""
    resps = client.simGetImages([
        airsim.ImageRequest(FLIGHT_CFG.camera_name, airsim.ImageType.Scene,        False, False),
        airsim.ImageRequest(FLIGHT_CFG.camera_name, airsim.ImageType.Segmentation, False, False),
    ])
    if (len(resps) < 2 or resps[0].width == 0 or resps[1].width == 0
            or resps[0].height == 0 or resps[1].height == 0):
        raise ValueError("Incomplete capture.")

    def _to_arr(r):
        return np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3)

    return _to_arr(resps[0]), _to_arr(resps[1])


async def async_capture_images(
    client: airsim.MultirotorClient,
    loop: asyncio.AbstractEventLoop,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Non-blocking image capture.
    The actual simGetImages RPC runs in a thread so the event loop can
    overlap it with the ProcessPool future from the previous frame.
    """
    return await loop.run_in_executor(None, _capture_sync, client)


# ---------------------------------------------------------------------------
# 7. DYNAMIC SEGMENTATION ID DISCOVERY  (unchanged — sequential by design,
#    must own the AirSim seg state exclusively)
# ---------------------------------------------------------------------------
_BLACKLIST_PATTERNS = (
    ".*[Ss]ky.*",".*[Tt]errain.*",".*[Ll]andscape.*",".*[Ff]loor.*",
    ".*[Aa]tmosphere.*",".*[Gg]rass.*",".*[Rr]oad.*",".*[Ss]treet.*",
    ".*[Ss]idewalk.*",".*[Pp]avement.*",".*[Gg]izmo.*",
)


def _reset_all(client: airsim.MultirotorClient) -> None:
    client.simSetSegmentationObjectID(".*", 0, is_name_regex=True)
    for pat in _BLACKLIST_PATTERNS:
        client.simSetSegmentationObjectID(pat, 0, is_name_regex=True)
    time.sleep(0.3)


def _position_directly_above(
    client: airsim.MultirotorClient,
    target_pos: airsim.Vector3r,
    altitude_ned: float = -12.0,
) -> None:
    r"""
    P_{\text{target}} = \begin{bmatrix} x_t \\ y_t \\ z_t \end{bmatrix}, \quad O_{\text{cam}} = \begin{bmatrix} x_c \\ y_c \\ z_c \end{bmatrix} \end{equation*}m
    """
    vpos = airsim.Vector3r(
        target_pos.x_val - FLIGHT_CFG.cam_offset.x_val,
        target_pos.y_val - FLIGHT_CFG.cam_offset.y_val,
        altitude_ned     - FLIGHT_CFG.cam_offset.z_val,
    )
    client.simSetVehiclePose(
        airsim.Pose(vpos, airsim.to_quaternion(0.0, 0.0, 0.0)),
        ignore_collision=True,
    )
    client.simSetCameraPose(
        FLIGHT_CFG.camera_name,
        airsim.Pose(
            FLIGHT_CFG.cam_offset,
            airsim.to_quaternion(math.radians(-90.0), 0.0, 0.0),
        ),
    )


def _set_id_with_retry(
    client: airsim.MultirotorClient,
    obj_name: str,
    seg_id: int,
    retries: int = 3,
) -> bool:
    for attempt in range(retries):
        ok = client.simSetSegmentationObjectID(obj_name, seg_id, is_name_regex=False)
        if ok:
            return True
        time.sleep(0.1)
    escaped = obj_name.replace("_", r"_")
    ok = client.simSetSegmentationObjectID(f"^{escaped}$", seg_id, is_name_regex=True)
    return ok


_GRID_PROBE_POINTS: Tuple[airsim.Vector3r, ...] = (
    airsim.Vector3r(   0,    0, 0),
    airsim.Vector3r( 100,    0, 0),
    airsim.Vector3r(   0,  100, 0),
    airsim.Vector3r(-100,    0, 0),
    airsim.Vector3r(   0, -100, 0),
)
_GRID_PROBE_ALT_NED: float = -50.0


def discover_segmentation_ids(
    client: airsim.MultirotorClient,
    targets: Dict[str, List[str]],
) -> Tuple[Dict[str, int], Dict[str, Tuple[int, int, int]]]:
    """Sequential by design — must own AirSim seg state exclusively."""
    
    # ---------------------------------------------------------
    # 1. Initialization and Setup
    # ---------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PASS 1: Dynamic Segmentation ID Discovery (V8)")
    logger.info("=" * 60)

    MAX_CANDIDATES = 10
    DISCOVERY_ALT  = -12.0

    discovered_ids:    Dict[str, int]                  = {}
    discovered_colors: Dict[str, Tuple[int, int, int]] = {}

    # Iterate through each target class we want to segment
    for temp_id, cfg in enumerate(CLASS_CONFIGS, start=1):
        
        # ---------------------------------------------------------
        # 2. Candidate Filtering
        # Filter objects to ensure they have valid, non-origin coordinates
        # ---------------------------------------------------------
        obj_list   = targets.get(cfg.name, [])
        candidates = [
            o for o in obj_list
            if is_valid_pose(client.simGetObjectPose(o))
        ][:MAX_CANDIDATES]

        # ---------------------------------------------------------
        # 3. Fallback: Grid Probing (If no valid objects are found)
        # If the class exists but lacks specific coordinates, scan the map
        # ---------------------------------------------------------
        if not candidates:
            logger.warning(
                "[%s] No static objects found — attempting multi-point grid probe …",
                cfg.name,
            )
            _reset_all(client)
            for pat in cfg.regex_patterns:
                client.simSetSegmentationObjectID(pat, temp_id, is_name_regex=True)
            time.sleep(0.5)

            colour_found: Optional[Tuple[int, int, int]] = None
            for probe_pt in _GRID_PROBE_POINTS:
                _position_directly_above(client, probe_pt, _GRID_PROBE_ALT_NED)
                time.sleep(0.5)
                try:
                    _, seg = _capture_sync(client)
                except Exception as exc:
                    logger.warning(
                        "  Capture failed at probe (%.0f, %.0f): %s",
                        probe_pt.x_val, probe_pt.y_val, exc,
                    )
                    continue

                # Statistical Analysis: Find the dominant non-black color in the image
                seg_flat       = seg.reshape(-1, 3)
                unique, counts = np.unique(seg_flat, axis=0, return_counts=True)
                non_black      = ~np.all(unique == [0, 0, 0], axis=1)
                unique, counts = unique[non_black], counts[non_black]

                if len(unique) > 0:
                    colour_found = tuple(int(c) for c in unique[np.argmax(counts)])
                    logger.info(
                        "  ✓ Grid probe @ (%.0f, %.0f): colour=%s  pixels=%d",
                        probe_pt.x_val, probe_pt.y_val,
                        colour_found, counts[np.argmax(counts)],
                    )
                    break

            if colour_found:
                discovered_ids[cfg.name]    = temp_id
                discovered_colors[cfg.name] = colour_found
            else:
                logger.error(
                    "  [%s] All grid probe points failed — class MISSING.",
                    cfg.name,
                )
            
            # Move to the next class after completing the fallback logic
            continue

        # ---------------------------------------------------------
        # 4. Main Discovery Logic: Probing Known Candidates
        # Position the drone above valid objects to extract their assigned color
        # ---------------------------------------------------------
        colour_found = None
        for sample_obj in candidates:
            _reset_all(client)
            primary_regex = cfg.regex_patterns[0]
            
            # Find all objects whose names match
            # Assign them the segmentation ID
            # Assign the temporary ID to the class in the simulator
            client.simSetSegmentationObjectID(primary_regex, temp_id, is_name_regex=True)
            _set_id_with_retry(client, sample_obj, temp_id)
            time.sleep(0.4)

            # This asks the simulator: “Where is this object and how is it oriented?”
            # x = 10 → forward
            # y = 5 → right
            # z = -2 → height
            # pitch, roll, yaw
            # Teleport drone directly above the target object
            pose = client.simGetObjectPose(sample_obj)
            logger.info(
                "[%s] Probing '%s' (ID=%d) top-down @ %.1f m NED",
                cfg.name, sample_obj, temp_id, DISCOVERY_ALT,
            )

            # Place the drone directly above this object at a fixed height, and point the camera straight down
            _position_directly_above(client, pose.position, DISCOVERY_ALT)
            time.sleep(FLIGHT_CFG.post_shutter_delay_s + 0.5)

            try:
                _, seg = _capture_sync(client)
            except Exception as exc:
                logger.warning("  Capture failed: %s — trying next candidate", exc)
                continue

            # ---------------------------------------------------------
            # 5. Statistical Color Extraction
            # Analyze the captured segmentation image to find the object's color
            # ---------------------------------------------------------
            seg_flat       = seg.reshape(-1, 3)
            unique, counts = np.unique(seg_flat, axis=0, return_counts=True)
            non_black      = ~np.all(unique == [0, 0, 0], axis=1)
            unique, counts = unique[non_black], counts[non_black]

            # Handle cases where the object is hidden or occluded (renders as black)
            if len(unique) == 0:
                logger.warning("  No non-black pixels — object not visible, trying next")
                cv2.imwrite(
                    str(DEBUG_DIR / f"discover_{cfg.name}_BLANK_{sample_obj}.png"), seg
                )
                continue

            # Extract the most frequent color representing the object
            colour_found = tuple(int(c) for c in unique[np.argmax(counts)])
            logger.info("  ✓ colour=%s  pixels=%d", colour_found, counts[np.argmax(counts)])
            cv2.imwrite(
                str(DEBUG_DIR / f"discover_{cfg.name}_id{temp_id}_{colour_found}.png"), seg
            )
            break

        # Log an error if all attempts to find the class color failed
        if colour_found is None:
            logger.error(
                "[%s] All %d candidates failed — class will be MISSING from dataset!",
                cfg.name, len(candidates),
            )
            continue

        # Save successful discoveries
        discovered_ids[cfg.name]    = temp_id
        discovered_colors[cfg.name] = colour_found

    # ---------------------------------------------------------
    # 6. Finalization and Data Export
    # Save the mapping of class names to their IDs and discovered RGB colors
    # ---------------------------------------------------------
    logger.info("Discovery done. IDs=%s", discovered_ids)
    logger.info("            Colours=%s", discovered_colors)

    with open(DEBUG_DIR / "discovered_ids.json", "w", encoding="utf-8") as fh:
        json.dump(
            {k: {"id": v, "color": list(discovered_colors[k])}
             for k, v in discovered_ids.items()},
            fh, indent=2,
        )

    return discovered_ids, discovered_colors


# ---------------------------------------------------------------------------
# 8. Apply Discovered IDs (sequential — owns AirSim seg state)
# ---------------------------------------------------------------------------
def apply_discovered_ids(
    client: airsim.MultirotorClient,
    targets: Dict[str, List[str]],
    discovered_ids: Dict[str, int],
) -> None:
    logger.info("Applying discovered IDs to all scene objects …")
    _reset_all(client)

    for cfg in CLASS_CONFIGS:
        if cfg.name not in discovered_ids:
            continue
        seg_id = discovered_ids[cfg.name]
        for obj_name in targets.get(cfg.name, []):
            client.simSetSegmentationObjectID(obj_name, seg_id, is_name_regex=False)
        for pat in cfg.regex_patterns:
            client.simSetSegmentationObjectID(pat, seg_id, is_name_regex=True)

    time.sleep(0.5)
    logger.info("All IDs applied.")


# ---------------------------------------------------------------------------
# 9. Union-Find Box Merger
# ---------------------------------------------------------------------------
class _UnionFind:
    __slots__ = ("parent", "rank")
    def __init__(self, n):
        self.parent = list(range(n)); self.rank = [0]*n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]
        return x
    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr: return
        if self.rank[xr] < self.rank[yr]:   self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]: self.parent[yr] = xr
        else: self.parent[yr] = xr; self.rank[xr] += 1


def _boxes_near(b1, b2, max_dist):
    xd = max(0, max(b1[0],b2[0]) - min(b1[2],b2[2]))
    yd = max(0, max(b1[1],b2[1]) - min(b1[3],b2[3]))
    return xd <= max_dist and yd <= max_dist


def merge_nearby_rects(
    rects: List[Tuple[int,int,int,int]], max_dist: int
) -> List[Tuple[int,int,int,int]]:
    if not rects or max_dist == 0:
        return rects
    boxes = [[x,y,x+w,y+h] for x,y,w,h in rects]
    n  = len(boxes)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i+1, n):
            if _boxes_near(boxes[i], boxes[j], max_dist):
                uf.union(i, j)
    groups: Dict[int, list] = {}
    for idx, b in enumerate(boxes):
        groups.setdefault(uf.find(idx), []).append(b)
    merged = []
    for grp in groups.values():
        x1 = min(b[0] for b in grp); y1 = min(b[1] for b in grp)
        x2 = max(b[2] for b in grp); y2 = max(b[3] for b in grp)
        merged.append((x1, y1, x2-x1, y2-y1))
    return merged


# ---------------------------------------------------------------------------
# 10. NMS
# ---------------------------------------------------------------------------
def _iou(a, b) -> float:
    ax,ay,aw,ah = a; bx,by,bw,bh = b
    ix1,iy1 = max(ax,bx), max(ay,by)
    ix2,iy2 = min(ax+aw,bx+bw), min(ay+ah,by+bh)
    inter   = max(0,ix2-ix1)*max(0,iy2-iy1)
    union   = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0.0


def _io_min(a, b) -> float:
    ax,ay,aw,ah = a; bx,by,bw,bh = b
    ix1,iy1 = max(ax,bx), max(ay,by)
    ix2,iy2 = min(ax+aw,bx+bw), min(ay+ah,by+bh)
    inter   = max(0,ix2-ix1)*max(0,iy2-iy1)
    min_area = min(aw*ah, bw*bh)
    return inter/min_area if min_area > 0 else 0.0


def nms_filter(
    rects: List[Tuple[int,int,int,int]],
    iou_threshold: float = 0.45,
    containment_threshold: float = 0.80,
) -> List[Tuple[int,int,int,int]]:
    if not rects:
        return []
    rects_sorted = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)
    keep = []
    for r in rects_sorted:
        suppress = False
        for k in keep:
            if _iou(r, k) >= iou_threshold:
                suppress = True; break
            if _io_min(r, k) >= containment_threshold:
                suppress = True; break
        if not suppress:
            keep.append(r)
    return keep


# ---------------------------------------------------------------------------
# 11. MULTI-CLASS YOLO Extraction  (pure CPU — runs inside child process)
# ---------------------------------------------------------------------------
def _detect_color_collisions(
    seg_bgr: np.ndarray,
    discovered_colors: Dict[str, Tuple[int, int, int]],
    tolerance: int = 25,
) -> Set[str]:
    skip: Set[str] = set()
    names  = list(discovered_colors.keys())
    arrays = {n: np.array(discovered_colors[n], dtype=np.uint8) for n in names}
    seg_flat = seg_bgr.reshape(-1, 3)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ca, cb = arrays[a], arrays[b]
            lo_a = np.clip(ca.astype(np.int16) - tolerance, 0, 255).astype(np.uint8)
            hi_a = np.clip(ca.astype(np.int16) + tolerance, 0, 255).astype(np.uint8)
            lo_b = np.clip(cb.astype(np.int16) - tolerance, 0, 255).astype(np.uint8)
            hi_b = np.clip(cb.astype(np.int16) + tolerance, 0, 255).astype(np.uint8)
            mask_a = np.all((seg_flat >= lo_a) & (seg_flat <= hi_a), axis=1)
            mask_b = np.all((seg_flat >= lo_b) & (seg_flat <= hi_b), axis=1)
            if int(np.sum(mask_a & mask_b)) > 5:
                skip.add(a); skip.add(b)
    return skip


def extract_yolo_labels_multi_class(
    seg_bgr: np.ndarray,
    discovered_colors: Dict[str, Tuple[int,int,int]],
    altitude: float,
) -> List[str]:
    labels: List[str] = []
    img_h, img_w = seg_bgr.shape[:2]
    abs_alt = abs(altitude)

    open_k  = 3 if abs_alt < 10 else 5
    close_k = 3 if abs_alt < 10 else 7
    k_open  = np.ones((open_k,  open_k),  np.uint8)
    k_close = np.ones((close_k, close_k), np.uint8)
    use_close = abs_alt >= VISION_CFG.close_altitude_threshold_m

    _MIN_AREA = {"vehicle":20,"person":6,"building":30,"lamppost":4,"tree":12}
    _px_per_m = max(5.0, 300.0 / max(abs_alt, 1.0))
    _MAX_BBOX = {
        "vehicle":  int((_px_per_m * 4.5) * (_px_per_m * 2.5) * 3),
        "person":   int((_px_per_m * 0.6) * (_px_per_m * 0.5) * 4),
        "building": img_h * img_w,
        "lamppost": img_h * img_w,
        "tree":     int((_px_per_m * 6.0) * (_px_per_m * 6.0) * 2),
    }

    skip_classes = _detect_color_collisions(seg_bgr, discovered_colors)

    for cfg in CLASS_CONFIGS:
        if cfg.name not in discovered_colors or cfg.name in skip_classes:
            continue

        min_area_px = _MIN_AREA.get(cfg.name, VISION_CFG.min_contour_area_px)
        colour = np.array(discovered_colors[cfg.name], dtype=np.uint8)
        tol    = cfg.color_tolerance
        lower  = np.clip(colour.astype(np.int16) - tol, 0, 255).astype(np.uint8)
        upper  = np.clip(colour.astype(np.int16) + tol, 0, 255).astype(np.uint8)

        mask = cv2.inRange(seg_bgr, lower, upper)
        if cv2.countNonZero(mask) == 0:
            continue

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        if use_close:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        raw_rects: List[Tuple[int,int,int,int]] = []

        for i in range(1, num_labels):
            x  = stats[i, cv2.CC_STAT_LEFT]
            y  = stats[i, cv2.CC_STAT_TOP]
            w  = stats[i, cv2.CC_STAT_WIDTH]
            h  = stats[i, cv2.CC_STAT_HEIGHT]
            ar = stats[i, cv2.CC_STAT_AREA]

            if ar < min_area_px:
                continue
            if w * h > _MAX_BBOX.get(cfg.name, img_h * img_w):
                continue
            if cfg.name != "lamppost":
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect > cfg.max_aspect_ratio or aspect < cfg.min_aspect_ratio:
                    continue
                sub  = mask[y:y+h, x:x+w]
                fill = cv2.countNonZero(sub) / max(w*h, 1)
                if fill < cfg.min_fill_ratio:
                    continue
            raw_rects.append((x, y, w, h))

        merged = merge_nearby_rects(raw_rects, cfg.merge_px)
        merged = nms_filter(merged, iou_threshold=VISION_CFG.nms_iou_threshold,
                            containment_threshold=0.80)

        for (xp, yp, wp, hp) in merged:
            xp = max(0, min(xp, img_w-1))
            yp = max(0, min(yp, img_h-1))
            wp = max(1, min(wp, img_w-xp))
            hp = max(1, min(hp, img_h-yp))
            xc = max(0.0, min(1.0, (xp + wp/2.0) / img_w))
            yc = max(0.0, min(1.0, (yp + hp/2.0) / img_h))
            nw = max(0.0, min(1.0, wp / img_w))
            nh = max(0.0, min(1.0, hp / img_h))
            if VISION_CFG.min_norm_area <= nw*nh <= VISION_CFG.max_norm_area:
                labels.append(
                    f"{cfg.yolo_id.value} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
                )
    return labels


# ---------------------------------------------------------------------------
# 12. MULTIPROCESSING worker — runs entirely in a child process
#     Receives plain Python / numpy objects (picklable).
#     Returns (frame_id, label_count) so the parent can log progress.
#
#     Why multiprocessing (not threading) here:
#       cv2.inRange, connectedComponentsWithStats, morphologyEx,
#       imwrite — all release the GIL but are still serial inside one
#       process.  A separate *process* gets its own Python interpreter
#       and can run truly in parallel with AirSim I/O on the main process.
# ---------------------------------------------------------------------------
def _process_frame_worker(
    frame_id:         int,
    rgb_bgr:          np.ndarray,
    seg_bgr:          np.ndarray,
    discovered_colors: Dict[str, Tuple[int,int,int]],
    altitude:         float,
    obj_name:         str,
    class_name:       str,
    cam_pose_dict:    dict,          # CameraPose serialised as dict
) -> Tuple[int, int]:
    """
    All CPU-heavy work for one frame.
    Returns (frame_id, num_labels).
    """
    labels = extract_yolo_labels_multi_class(seg_bgr, discovered_colors, altitude)

    if not labels:
        return frame_id, 0

    # ── Save RGB image ──────────────────────────────────────────────────────
    cv2.imwrite(str(IMAGES_DIR / f"drone_{frame_id:05d}.jpg"), rgb_bgr)

    # ── Save label file ─────────────────────────────────────────────────────
    with open(LABELS_DIR / f"drone_{frame_id:05d}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(labels))

    # ── Save debug image ────────────────────────────────────────────────────
    debug  = rgb_bgr.copy()
    img_h, img_w = debug.shape[:2]
    CLASS_COLORS = {0:(0,255,0),1:(255,128,0),2:(0,128,255),3:(255,0,255),4:(0,255,255)}
    for lbl in labels:
        parts = lbl.strip().split()
        if len(parts) != 5: continue
        cid, xc, yc, nw, nh = map(float, parts)
        w = int(nw*img_w); h = int(nh*img_h)
        x = int(xc*img_w - w/2); y = int(yc*img_h - h/2)
        col = CLASS_COLORS.get(int(cid), (255,255,255))
        cv2.rectangle(debug, (x,y), (x+w,y+h), col, 2)
        cv2.putText(debug, f"ID:{int(cid)}", (x, max(y-5,15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.imwrite(str(DEBUG_DIR / f"debug_frame_{frame_id:05d}.jpg"), debug)

    # ── Save metadata ───────────────────────────────────────────────────────
    meta = {
        "frame_id": frame_id, "target_object": obj_name,
        "target_class": class_name, "altitude_m": abs(altitude),
        "camera_pose": cam_pose_dict,
        "num_detections": len(labels), "labels": labels,
    }
    with open(META_DIR / f"frame_{frame_id:05d}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    return frame_id, len(labels)


# ---------------------------------------------------------------------------
# 13. Object Discovery
# ---------------------------------------------------------------------------
_OBJECT_BLACKLIST: Tuple[str, ...] = (
    "sidewalk", "pavement", "road", "terrain", "landscape",
    "grass", "floor", "sky", "atmosphere", "gizmo",
    "path", "curb", "kerb", "ground", "dirt",
    "water", "river", "lake", "ocean", "barrier",
    "fence", "guardrail", "invisible", "collision",
    "trigger", "volume", "zone",
    "pedestrianpath",
    "sidewalk_extra",
    "manager",
    "camera",
    "planter",
    "parking",
    "bush",           # prevents Bush_* matching "bus" in vehicle keywords
)


def _is_blacklisted(name: str) -> bool:
    low = name.lower()
    return any(bl in low for bl in _OBJECT_BLACKLIST)


def discover_objects(client: airsim.MultirotorClient) -> Dict[str, List[str]]:
    """
    Discover and classify scene objects into predefined categories.

    Pipeline:
    1. Retrieve all objects from the AirSim scene.
    2. Filter out irrelevant objects using a blacklist (e.g., cameras, terrain).
    3. Classify remaining objects into classes based on name matching.

    Classification logic:
    - Each class defines a set of search keywords.
    - If an object name contains any keyword of a class,
      it is assigned to that class.
    - The first matching class is used (no multi-class assignment).

    Returns:
        Dictionary mapping class name → list of object names
    """

    logger.info("Discovering scene objects …")

    # Retrieve all objects in the scene
    all_objs = client.simListSceneObjects()

    # Initialize output structure: {class_name: [objects]}
    targets: Dict[str, List[str]] = {cfg.name: [] for cfg in CLASS_CONFIGS}

    n_skipped = 0

    for obj in all_objs:

        # Skip unwanted or non-relevant objects
        if _is_blacklisted(obj):
            n_skipped += 1
            continue

        # Normalize name for case-insensitive matching
        low = obj.lower()

        # Assign object to the first matching class based on keywords
        for cfg in CLASS_CONFIGS:
            if any(k in low for k in cfg.search_keywords):
                targets[cfg.name].append(obj)
                break

    logger.info("  Blacklist filtered %d objects.", n_skipped)

    # Shuffle objects to improve dataset diversity
    for cfg in CLASS_CONFIGS:
        random.shuffle(targets[cfg.name])
        logger.info("  [%10s] %d objects selected", cfg.name, len(targets[cfg.name]))

    return targets
# ---------------------------------------------------------------------------
# 14. ASYNC Pass 2 — Dataset Capture
#     Pattern: "fire and forget into process pool"
#
#     Timeline per object/altitude:
#       t=0   async_position_drone()   ← awaited (non-blocking)
#       t=1s  async_capture_images()   ← awaited (non-blocking)
#       t=1s  submit frame to POOL     ← non-blocking, returns Future
#       t=1s  move to next altitude    ← drone moves while pool works
#       ...
#       t=end await all pending futures
# ---------------------------------------------------------------------------
async def _capture_pass2(
    client:            airsim.MultirotorClient,
    targets:           Dict[str, List[str]],
    discovered_colors: Dict[str, Tuple[int,int,int]],
    loop:              asyncio.AbstractEventLoop,
    pool:              concurrent.futures.ProcessPoolExecutor,
) -> int:
    """
    Fully async capture loop.
    Returns total number of saved images.
    """
    image_count   = 0
    pending: List[concurrent.futures.Future] = []

    for cfg in CLASS_CONFIGS:
        objects = targets.get(cfg.name, [])
        if not objects:
            logger.info("No static objects for class '%s'; skipping.", cfg.name)
            continue

        for obj_name in objects:
            obj_pose = client.simGetObjectPose(obj_name)
            if not is_valid_pose(obj_pose):
                logger.warning("Invalid pose for '%s'; skipping.", obj_name)
                continue

            for altitude in FLIGHT_CFG.airsim_altitudes_ned:
                cam_pose = compute_camera_orbit(
                    obj_pose.position, altitude, cfg.approx_size_m
                )

                # ── Non-blocking: move drone + wait shutter delay ──────────
                await async_position_drone(client, cam_pose, loop)

                # ── Non-blocking: capture both images ─────────────────────
                try:
                    rgb_bgr, seg_bgr = await async_capture_images(client, loop)
                except Exception as exc:
                    logger.warning("Capture failed %s @ %.1f m: %s",
                                   obj_name, abs(altitude), exc)
                    continue

                if image_count == 0:
                    cv2.imwrite(str(DEBUG_DIR/"test_segmentation_colors.png"), seg_bgr)

                # ── Submit CPU work to process pool (non-blocking) ─────────
                # Serialise CameraPose as a plain dict for pickling
                cam_dict = {
                    "x": cam_pose.cx, "y": cam_pose.cy, "z": cam_pose.cz,
                    "yaw_deg":   math.degrees(cam_pose.yaw_rad),
                    "pitch_deg": math.degrees(cam_pose.pitch_rad),
                }
                future = pool.submit(
                    _process_frame_worker,
                    image_count,
                    rgb_bgr,
                    seg_bgr,
                    discovered_colors,
                    altitude,
                    obj_name,
                    cfg.name,
                    cam_dict,
                )
                pending.append(future)
                image_count += 1   # optimistic increment; worker returns 0 if empty

                logger.info(
                    "  Submitted frame %05d | %s @ %.1f m | pool queue: %d",
                    image_count - 1, obj_name, abs(altitude), len(pending),
                )

    # ── Drain remaining futures ────────────────────────────────────────────
    logger.info("Waiting for %d pending frames to finish …", len(pending))
    saved = 0
    for fut in concurrent.futures.as_completed(pending):
        try:
            fid, n_labels = fut.result()
            if n_labels > 0:
                saved += 1
                logger.info("  [DONE] frame %05d — %d labels", fid, n_labels)
            else:
                logger.info("  [SKIP] frame %05d — no labels (not saved)", fid)
        except Exception as exc:
            logger.error("  Worker error: %s", exc)

    return saved


# ---------------------------------------------------------------------------
# 15. Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 65)
    logger.info("  AirSim YOLO Extractor — V8 (Async + Multiprocessing)")
    logger.info("  CPU workers: %d", _CPU_WORKERS)
    logger.info("=" * 65)

    setup_directories()

    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        logger.info("Connected to AirSim.")
    except Exception as exc:
        logger.error("Connection failed: %s", exc)
        return

    fix_unreal_rendering(client)

    # ── PASS 1 (sequential — owns AirSim seg state) ───────────────────────

    # prepare valid objects that not in black list and return there objects
    targets = discover_objects(client)

    discovered_ids, discovered_colors = discover_segmentation_ids(client, targets)

    if not discovered_ids:
        logger.error("FATAL: Could not discover ANY segmentation IDs. Aborting.")
        return

    apply_discovered_ids(client, targets, discovered_ids)

    # ── PASS 2 (async I/O + multiprocessing CPU) ──────────────────────────
    logger.info("=" * 50)
    logger.info("PASS 2: Dataset Capture (Async + Multiprocessing)")
    logger.info("=" * 50)

    with concurrent.futures.ProcessPoolExecutor(max_workers=_CPU_WORKERS) as pool:
        loop  = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            saved = loop.run_until_complete(
                _capture_pass2(client, targets, discovered_colors, loop, pool)
            )
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            saved = 0
        finally:
            loop.close()

    client.enableApiControl(False)
    logger.info("API control released.")
    logger.info("PIPELINE COMPLETE. Total images saved: %d", saved)


if __name__ == "__main__":
    main()