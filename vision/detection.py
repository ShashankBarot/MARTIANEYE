"""
vision/detection.py  —  OpenCV 3.x / 4.x / 4.7+ compatible
-------------------------------------------------------------
JetPack 4  ships with OpenCV 3.x  → uses the 3.x ArUco API
JetPack 4.6 ships with OpenCV 4.x → uses the legacy 4.x ArUco API
JetPack 5+ ships with OpenCV 4.7+ → uses the new ArucoDetector class

This file detects the version at import time and routes to the right code path.
Your mission logic never needs to know which API is running underneath.

OpenCV 3.x ArUco API differences vs 4.x:
  DETECT  : aruco.detectMarkers()              ← same in all versions
  PARAMS  : aruco.DetectorParameters_create()  ← same in 3.x and 4.6
  POSE    : aruco.estimatePoseSingleMarkers()  ← 3.x ONLY (removed in 4.7)
  AXES    : aruco.drawAxis()                   ← 3.x ONLY (renamed drawFrameAxes in 4.x)
  GENERATE: aruco.drawMarker()                 ← 3.x / 4.6 (renamed generateImageMarker in 4.7)
"""

import cv2
import cv2.aruco as aruco
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# ---------------------------------------------------------------------------
# Version detection — runs once at import
# ---------------------------------------------------------------------------
_CV_MAJOR = int(cv2.__version__.split(".")[0])   # 3, 4, etc.

# API routing flags
_HAS_ARUCO_DETECTOR   = hasattr(aruco, "ArucoDetector")           # OpenCV 4.7+
_HAS_PARAMS_CREATE    = hasattr(aruco, "DetectorParameters_create")  # OpenCV 3.x / 4.6
_HAS_ESTIMATE_POSE    = hasattr(aruco, "estimatePoseSingleMarkers")  # OpenCV 3.x only
_HAS_DRAW_AXIS        = hasattr(aruco, "drawAxis")                   # OpenCV 3.x only
_HAS_DRAW_FRAME_AXES  = hasattr(aruco, "drawFrameAxes")              # OpenCV 4.x+

if _CV_MAJOR == 3:
    _API_LABEL = "OpenCV 3.x (JetPack 4)"
elif _HAS_ARUCO_DETECTOR:
    _API_LABEL = "OpenCV 4.7+ (new API)"
else:
    _API_LABEL = "OpenCV 4.x legacy"

print(f"[Detection] {cv2.__version__} detected — using {_API_LABEL} ArUco path")


# ---------------------------------------------------------------------------
# Data classes — identical regardless of OpenCV version
# ---------------------------------------------------------------------------

@dataclass
class ArUcoResult:
    """One detected ArUco marker."""
    marker_id:   int
    corners:     np.ndarray        # shape (4, 2)  —  pixel coords, clockwise from top-left
    centre:      Tuple[int, int]   # pixel centre of the marker
    rvec:        Optional[np.ndarray] = None   # rotation vector (needs real calibration)
    tvec:        Optional[np.ndarray] = None   # translation vector
    distance_m:  Optional[float]      = None   # metres to marker  (None until calibrated)


@dataclass
class YOLOResult:
    """One YOLO detection."""
    class_name:  str
    confidence:  float
    bbox:        Tuple[int, int, int, int]   # x1, y1, x2, y2
    centre:      Tuple[int, int]


@dataclass
class DetectionFrame:
    """All detections from one frame. Passed to MissionController each loop."""
    aruco_markers: List[ArUcoResult] = field(default_factory=list)
    yolo_objects:  List[YOLOResult]  = field(default_factory=list)
    frame_rgb:     Optional[np.ndarray] = None   # annotated BGR image for display

    @property
    def has_landing_marker(self) -> bool:
        """True when the landing pad marker (ID 0) is visible."""
        return any(m.marker_id == 0 for m in self.aruco_markers)

    @property
    def landing_marker(self) -> Optional["ArUcoResult"]:
        for m in self.aruco_markers:
            if m.marker_id == 0:
                return m
        return None

    @property
    def has_target(self) -> bool:
        return len(self.yolo_objects) > 0


# ---------------------------------------------------------------------------
# Camera calibration placeholders
# *** Replace with your real values once you run calibration ***
#
# Until then: marker corners are detected correctly, but:
#   - distance_m  will be unreliable (placeholder focal length)
#   - rvec / tvec are ballpark only
#
# To calibrate: print a 9x6 chessboard, take 20 photos, run calibrate.py
# (ask me for the calibration script when you're ready)
# ---------------------------------------------------------------------------
DEFAULT_CAMERA_MATRIX = np.array([
    [600.0,   0.0, 320.0],
    [  0.0, 600.0, 240.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

DEFAULT_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

MARKER_SIZE_M = 0.15   # ← measure your printed marker side length in metres


# ---------------------------------------------------------------------------
# VisionDetector
# ---------------------------------------------------------------------------

class VisionDetector:
    """
    Runs ArUco marker detection and optional YOLO object detection on each frame.

    Works on OpenCV 3.x (JetPack 4), 4.x, and 4.7+. Same interface everywhere.

    Quick start (ArUco only, no GPU required):
        detector = VisionDetector(use_yolo=False)
        result   = detector.process(frame)
        if result.has_landing_marker:
            print(result.landing_marker.centre)

    With YOLO:
        detector = VisionDetector(use_yolo=True, yolo_target_classes=["person"])
    """

    def __init__(
        self,
        aruco_dict_type = aruco.DICT_4X4_250,
        yolo_model_path: str = "yolov8n.pt",
        yolo_confidence: float = 0.5,
        yolo_target_classes: Optional[List[str]] = None,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs:   Optional[np.ndarray] = None,
        use_yolo: bool = True,
    ):
        # ----------------------------------------------------------------
        # ArUco init — three code paths for three API generations
        # ----------------------------------------------------------------
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)

        if _HAS_ARUCO_DETECTOR:
            # OpenCV 4.7+ — new class-based API
            self.aruco_params   = aruco.DetectorParameters()
            self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        else:
            # OpenCV 3.x and 4.6 — DetectorParameters_create() free function
            self.aruco_params   = aruco.DetectorParameters_create()
            self.aruco_detector = None   # detectMarkers() is a free function here

        self.camera_matrix = camera_matrix if camera_matrix is not None else DEFAULT_CAMERA_MATRIX
        self.dist_coeffs   = dist_coeffs   if dist_coeffs   is not None else DEFAULT_DIST_COEFFS

        # ----------------------------------------------------------------
        # YOLO init — lazy import so ArUco-only mode has no dependency
        # ----------------------------------------------------------------
        self.use_yolo            = use_yolo
        self.yolo_confidence     = yolo_confidence
        self.yolo_target_classes = yolo_target_classes
        self.yolo_model          = None

        if use_yolo:
            try:
                from ultralytics import YOLO
                print("[VisionDetector] Loading YOLO model — this takes ~10s first run...")
                self.yolo_model = YOLO(yolo_model_path)
                print("[VisionDetector] YOLO ready.")
            except ImportError:
                print("[VisionDetector] WARNING: ultralytics not installed.")
                print("  Run:  pip3 install ultralytics")
                print("  Continuing with ArUco only.")
                self.use_yolo = False

    # ----------------------------------------------------------------
    # Public API — call once per frame
    # ----------------------------------------------------------------

    def process(self, frame: np.ndarray) -> DetectionFrame:
        """
        Detect all markers and objects in one BGR frame.

        Args:
            frame : BGR numpy array from cv2.VideoCapture.read()

        Returns:
            DetectionFrame with .aruco_markers, .yolo_objects, .frame_rgb
        """
        annotated = frame.copy()
        result    = DetectionFrame()

        result.aruco_markers = self._detect_aruco(frame, annotated)

        if self.use_yolo and self.yolo_model is not None:
            result.yolo_objects = self._detect_yolo(frame, annotated)

        result.frame_rgb = annotated
        return result

    # ----------------------------------------------------------------
    # ArUco detection
    # ----------------------------------------------------------------

    def _detect_aruco(self, frame: np.ndarray, annotated: np.ndarray) -> List[ArUcoResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Step 1: detect marker corners ---
        if _HAS_ARUCO_DETECTOR:
            # OpenCV 4.7+
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            # OpenCV 3.x and 4.6 — identical call signature
            corners, ids, _ = aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )

        results = []
        if ids is None:
            return results

        # Draw green outlines + IDs on annotated frame
        aruco.drawDetectedMarkers(annotated, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            c  = corners[i][0]                  # shape (4, 2)
            cx = int(np.mean(c[:, 0]))
            cy = int(np.mean(c[:, 1]))

            rvec = tvec = distance_m = None

            # --- Step 2: pose estimation ---
            # OpenCV 3.x uses estimatePoseSingleMarkers() (free function, per-marker)
            # OpenCV 4.x uses solvePnP() directly
            try:
                if _HAS_ESTIMATE_POSE:
                    # OpenCV 3.x path
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                        corners[i:i+1],   # expects array of corners
                        MARKER_SIZE_M,
                        self.camera_matrix,
                        self.dist_coeffs
                    )
                    rvec = rvecs[0]
                    tvec = tvecs[0]
                else:
                    # OpenCV 4.x path — solvePnP per marker
                    obj_pts = np.array([
                        [-MARKER_SIZE_M/2,  MARKER_SIZE_M/2, 0],
                        [ MARKER_SIZE_M/2,  MARKER_SIZE_M/2, 0],
                        [ MARKER_SIZE_M/2, -MARKER_SIZE_M/2, 0],
                        [-MARKER_SIZE_M/2, -MARKER_SIZE_M/2, 0],
                    ], dtype=np.float64)
                    ok, rvec, tvec = cv2.solvePnP(
                        obj_pts, c.astype(np.float64),
                        self.camera_matrix, self.dist_coeffs
                    )
                    if not ok:
                        rvec = tvec = None

                if tvec is not None:
                    distance_m = float(np.linalg.norm(tvec))

                    # --- Step 3: draw pose axes ---
                    if _HAS_DRAW_AXIS:
                        # OpenCV 3.x  — drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length)
                        aruco.drawAxis(
                            annotated, self.camera_matrix, self.dist_coeffs,
                            rvec, tvec, MARKER_SIZE_M * 0.5
                        )
                    elif _HAS_DRAW_FRAME_AXES:
                        # OpenCV 4.x+ — drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, length)
                        cv2.drawFrameAxes(
                            annotated, self.camera_matrix, self.dist_coeffs,
                            rvec, tvec, MARKER_SIZE_M * 0.5
                        )

            except Exception as e:
                # Pose fails with placeholder calibration — detection still works fine
                pass

            # Label: ID + distance if known
            label = f"ID:{marker_id}" + (f"  {distance_m:.2f}m" if distance_m else "")
            cv2.putText(annotated, label, (cx - 40, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 80), -1)

            results.append(ArUcoResult(
                marker_id  = int(marker_id),
                corners    = c,
                centre     = (cx, cy),
                rvec       = rvec,
                tvec       = tvec,
                distance_m = distance_m,
            ))

        return results

    # ----------------------------------------------------------------
    # YOLO detection  (OpenCV version doesn't matter here)
    # ----------------------------------------------------------------

    def _detect_yolo(self, frame: np.ndarray, annotated: np.ndarray) -> List[YOLOResult]:
        raw     = self.yolo_model(frame, verbose=False)[0]
        results = []

        for box in raw.boxes:
            conf = float(box.conf[0])
            if conf < self.yolo_confidence:
                continue

            cls_id   = int(box.cls[0])
            cls_name = self.yolo_model.names[cls_id]

            if self.yolo_target_classes and cls_name not in self.yolo_target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(annotated, f"{cls_name} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            cv2.circle(annotated, (cx, cy), 4, (255, 165, 0), -1)

            results.append(YOLOResult(
                class_name = cls_name,
                confidence = conf,
                bbox       = (x1, y1, x2, y2),
                centre     = (cx, cy),
            ))

        return results