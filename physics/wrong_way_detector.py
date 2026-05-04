import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple, List

import numpy as np


@dataclass
class TrackMotionState:
    points: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=20))
    wrong_way_count: int = 0
    normal_count: int = 0
    last_frame_id: int = -1
    last_angle_deg: Optional[float] = None


class DirectionField:
    """
    Adaptive direction field for wrong-way detection.

    Each grid cell can be:
    - learning: still collecting movement directions
    - frozen: direction is stable, use it for wrong-way judgment
    - ignore: direction is unstable, do not judge this cell
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        grid_cols: int = 20,
        grid_rows: int = 12,
        angle_bins: int = 16,
        min_samples_to_freeze: int = 80,
        freeze_ratio: float = 0.88,
        max_dominant_switches_for_freeze: int = 4,
        min_samples_to_ignore: int = 140,
        ignore_ratio: float = 0.72,
        dominant_switch_threshold: int = 10,
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.angle_bins = angle_bins

        self.cell_w = max(1, frame_w // grid_cols)
        self.cell_h = max(1, frame_h // grid_rows)

        self.hist = np.zeros((grid_rows, grid_cols, angle_bins), dtype=np.float32)
        self.sample_count = np.zeros((grid_rows, grid_cols), dtype=np.int32)

        self.last_dominant_bin = np.full((grid_rows, grid_cols), -1, dtype=np.int32)
        self.dominant_switch_count = np.zeros((grid_rows, grid_cols), dtype=np.int32)

        self.frozen = np.zeros((grid_rows, grid_cols), dtype=bool)
        self.ignore = np.zeros((grid_rows, grid_cols), dtype=bool)
        self.frozen_bin = np.full((grid_rows, grid_cols), -1, dtype=np.int32)

        self.min_samples_to_freeze = min_samples_to_freeze
        self.freeze_ratio = freeze_ratio
        self.max_dominant_switches_for_freeze = max_dominant_switches_for_freeze

        self.min_samples_to_ignore = min_samples_to_ignore
        self.ignore_ratio = ignore_ratio
        self.dominant_switch_threshold = dominant_switch_threshold

    def _cell_index(self, x: float, y: float) -> Tuple[int, int]:
        col = int(np.clip(x / self.cell_w, 0, self.grid_cols - 1))
        row = int(np.clip(y / self.cell_h, 0, self.grid_rows - 1))
        return row, col

    def _angle_to_bin(self, angle_deg: float) -> int:
        angle_deg = angle_deg % 360.0
        return int(angle_deg / 360.0 * self.angle_bins) % self.angle_bins

    def _bin_to_angle(self, idx: int) -> float:
        return (idx + 0.5) * (360.0 / self.angle_bins)

    def _dominant_bin_and_ratio(self, row: int, col: int) -> Tuple[int, float]:
        hist = self.hist[row, col]
        total = float(hist.sum())
        if total <= 0:
            return -1, 0.0

        idx = int(np.argmax(hist))
        ratio = float(hist[idx] / total)
        return idx, ratio

    def _update_cell_state(self, row: int, col: int):
        if self.ignore[row, col] or self.frozen[row, col]:
            return

        sample_n = int(self.sample_count[row, col])
        dominant_bin, dominant_ratio = self._dominant_bin_and_ratio(row, col)
        if dominant_bin < 0:
            return

        prev_dominant_bin = int(self.last_dominant_bin[row, col])
        if prev_dominant_bin >= 0 and prev_dominant_bin != dominant_bin:
            self.dominant_switch_count[row, col] += 1

        self.last_dominant_bin[row, col] = dominant_bin
        switch_n = int(self.dominant_switch_count[row, col])

        if sample_n >= self.min_samples_to_ignore:
            if dominant_ratio < self.ignore_ratio or switch_n >= self.dominant_switch_threshold:
                self.ignore[row, col] = True
                return

        if sample_n >= self.min_samples_to_freeze:
            if dominant_ratio >= self.freeze_ratio and switch_n <= self.max_dominant_switches_for_freeze:
                self.frozen[row, col] = True
                self.frozen_bin[row, col] = dominant_bin

    def update(self, x: float, y: float, angle_deg: float, weight: float = 1.0, radius_cells: int = 0):
        row, col = self._cell_index(x, y)
        angle_bin = self._angle_to_bin(angle_deg)

        for rr in range(max(0, row - radius_cells), min(self.grid_rows, row + radius_cells + 1)):
            for cc in range(max(0, col - radius_cells), min(self.grid_cols, col + radius_cells + 1)):
                if self.frozen[rr, cc] or self.ignore[rr, cc]:
                    continue

                self.hist[rr, cc, angle_bin] += weight
                self.sample_count[rr, cc] += 1
                self._update_cell_state(rr, cc)

    def get_cell_state(self, x: float, y: float) -> str:
        row, col = self._cell_index(x, y)

        if self.ignore[row, col]:
            return "ignore"
        if self.frozen[row, col]:
            return "frozen"
        return "learning"

    def dominant_angle(self, x: float, y: float, min_samples: int = 10) -> Optional[Tuple[float, float]]:
        row, col = self._cell_index(x, y)

        if self.ignore[row, col]:
            return None

        if self.frozen[row, col]:
            angle_bin = int(self.frozen_bin[row, col])
            if angle_bin < 0:
                return None

            hist = self.hist[row, col]
            total = float(hist.sum())
            ratio = float(hist[angle_bin] / total) if total > 0 else 1.0
            return self._bin_to_angle(angle_bin), ratio

        if self.sample_count[row, col] < min_samples:
            return None

        dominant_bin, dominant_ratio = self._dominant_bin_and_ratio(row, col)
        if dominant_bin < 0:
            return None

        return self._bin_to_angle(dominant_bin), dominant_ratio

def compute_angle_deg(dx: float, dy: float) -> float:
    return (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0


def compute_angle_diff_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


class WrongWayDetector:
    """
    TrackGuard native wrong-way detector.

    Input:
        active_tracks from TrackManager

    Output:
        List[Dict] with TrackGuard-style behaviour detections
    """

    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = config
        
        self.frame_w = int(config.get("frame_w", 1280))
        self.frame_h = int(config.get("frame_h", 720))

        self.grid_cols = int(config.get("grid_cols", 20))
        self.grid_rows = int(config.get("grid_rows", 12))
        self.angle_bins = int(config.get("angle_bins", 16))

        self.move_threshold = float(config.get("move_threshold", 18.0))
        self.min_field_samples = int(config.get("min_field_samples", 14))

        self.learn_frames = int(config.get("learn_frames", 150))
        self.wrong_way_angle = float(config.get("wrong_way_angle", 80.0))
        self.wrong_way_frames = int(config.get("wrong_way_frames", 8))
        self.trail_len = int(config.get("trail_len", 20))

        self.min_samples_to_freeze = int(config.get("min_samples_to_freeze", 80))
        self.freeze_ratio = float(config.get("freeze_ratio", 0.88))
        self.max_dominant_switches_for_freeze = int(config.get("max_dominant_switches_for_freeze", 4))

        self.min_samples_to_ignore = int(config.get("min_samples_to_ignore", 140))
        self.ignore_ratio = float(config.get("ignore_ratio", 0.72))
        self.dominant_switch_threshold = int(config.get("dominant_switch_threshold", 10))

        self.direction_field = DirectionField(
            frame_w=self.frame_w,
            frame_h=self.frame_h,
            grid_cols=self.grid_cols,
            grid_rows=self.grid_rows,
            angle_bins=self.angle_bins,
            min_samples_to_freeze=self.min_samples_to_freeze,
            freeze_ratio=self.freeze_ratio,
            max_dominant_switches_for_freeze=self.max_dominant_switches_for_freeze,
            min_samples_to_ignore=self.min_samples_to_ignore,
            ignore_ratio=self.ignore_ratio,
            dominant_switch_threshold=self.dominant_switch_threshold,
        )

        self.track_states: Dict[int, TrackMotionState] = defaultdict(
            lambda: TrackMotionState(points=deque(maxlen=self.trail_len))
        )

        self.frame_count = 0

    def _get_track_id(self, track) -> int:
        return int(getattr(track, "track_id", -1))

    def _get_bbox(self, track) -> Optional[List[float]]:
        bbox = getattr(track, "bbox", None)

        if bbox is None:
            current_detection = getattr(track, "current_detection", None)
            if isinstance(current_detection, dict):
                bbox = current_detection.get("bbox")

        if bbox is None:
            return None

        return [float(v) for v in bbox]

    def _get_center(self, track) -> Optional[Tuple[float, float]]:
        center = getattr(track, "center", None)

        if center is None:
            current_detection = getattr(track, "current_detection", None)
            if isinstance(current_detection, dict):
                center = current_detection.get("center")

        if center is not None and len(center) >= 2:
            return float(center[0]), float(center[1])

        bbox = self._get_bbox(track)
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _get_class_name(self, track) -> str:
        current_detection = getattr(track, "current_detection", None)

        if isinstance(current_detection, dict):
            for key in ["class_name", "label", "class", "category"]:
                if key in current_detection:
                    return str(current_detection[key])

        return "vehicle"

    def _estimate_radius_cells(self, bbox: List[float]) -> int:
        x1, y1, x2, y2 = bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        radius = round(
            max(
                bw / max(1, self.direction_field.cell_w),
                bh / max(1, self.direction_field.cell_h),
            ) / 2
        ) - 1

        return max(0, int(radius))

    def _cleanup_states(self, active_track_ids: set):
        stale_ids = [track_id for track_id in self.track_states.keys() if track_id not in active_track_ids]
        for track_id in stale_ids:
            self.track_states.pop(track_id, None)

    def detect(self, active_tracks, velocity_field=None):
        self.frame_count += 1
        detections = []

        active_track_ids = set()

        for track in active_tracks:
            track_id = self._get_track_id(track)
            if track_id < 0:
                continue

            bbox = self._get_bbox(track)
            center = self._get_center(track)

            if bbox is None or center is None:
                continue

            active_track_ids.add(track_id)

            cx, cy = center
            x1, y1, x2, y2 = bbox

            state = self.track_states[track_id]
            state.points.append((cx, cy))
            state.last_frame_id = self.frame_count

            if len(state.points) < 6:
                continue

            p0 = state.points[0]
            p1 = state.points[-1]

            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            dist = math.hypot(dx, dy)

            if dist < self.move_threshold:
                continue

            move_angle_deg = compute_angle_deg(dx, dy)
            state.last_angle_deg = move_angle_deg

            radius_cells = self._estimate_radius_cells(bbox)
            self.direction_field.update(cx, cy, move_angle_deg, weight=1.0, radius_cells=radius_cells)

            cell_state = self.direction_field.get_cell_state(cx, cy)
            dominant_direction = self.direction_field.dominant_angle(
                cx,
                cy,
                min_samples=self.min_field_samples,
            )

            if cell_state == "ignore":
                state.wrong_way_count = 0
                state.normal_count = 0
                continue

            if self.frame_count <= self.learn_frames:
                continue

            if cell_state != "frozen" or dominant_direction is None:
                continue

            ref_angle_deg, ref_ratio = dominant_direction
            angle_diff = compute_angle_diff_deg(move_angle_deg, ref_angle_deg)

            if angle_diff > self.wrong_way_angle:
                state.wrong_way_count += 1
                state.normal_count = 0
            else:
                state.normal_count += 1
                state.wrong_way_count = max(0, state.wrong_way_count - 1)

            if state.wrong_way_count >= self.wrong_way_frames:
                confidence = min(1.0, state.wrong_way_count / max(1, self.wrong_way_frames))

                detections.append({
                    "behaviour_type": "wrong_way",
                    "severity": "high",
                    "track_id": track_id,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "center": [float(cx), float(cy)],
                    "class_name": self._get_class_name(track),
                    "confidence": confidence,
                    "wrong_way_count": state.wrong_way_count,
                    "wrong_way_frames": self.wrong_way_frames,
                    "move_angle_deg": float(move_angle_deg),
                    "reference_angle_deg": float(ref_angle_deg),
                    "angle_diff_deg": float(angle_diff),
                    "reference_ratio": float(ref_ratio),
                    "cell_state": cell_state,
                })

        self._cleanup_states(active_track_ids)

        return detections