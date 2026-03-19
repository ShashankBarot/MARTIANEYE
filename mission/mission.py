"""
mission/mission.py
------------------
Mission decision logic — Phase 2 version.

This connects vision output (DetectionFrame) to MAVLink control commands.
Currently uses SIMULATED control (print statements). Phase 3 will swap in real
MAVLink calls by replacing the MavlinkControl import with real implementation.

State machine:
    SEARCH  → drone orbits or hovers, looking for marker / target
    APPROACH → target (YOLO) or marker (ArUco) visible, drone moves toward it
    ALIGN   → landing marker is centred, drone descends
    LAND    → precision landing in progress
    DONE    → mission complete
"""

from enum import Enum, auto
from typing import Optional
from vision.detection import DetectionFrame, ArUcoResult


class MissionState(Enum):
    SEARCH  = auto()
    APPROACH = auto()
    ALIGN    = auto()
    LAND     = auto()
    DONE     = auto()


class MissionController:
    """
    Consumes DetectionFrame objects and issues velocity / mode commands.
    
    How to use in your main loop:
        mission = MissionController(control=mavlink_ctrl)
        while True:
            frame = cam.read()
            detection = detector.process(frame)
            mission.update(detection)
    """

    # Pixel threshold: marker centre is "aligned" when within this many px
    ALIGN_THRESHOLD_PX = 30

    # Frame dimensions (default 640×480 → centre is 320,240)
    FRAME_CX = 320
    FRAME_CY = 240

    def __init__(self, control):
        """
        Args:
            control : any object with the MavlinkControl interface
                      (real or simulated — both work here)
        """
        self.control = control
        self.state   = MissionState.SEARCH
        self._frame_count = 0

    def update(self, detection: DetectionFrame):
        """
        Call this once per frame with the latest detection results.
        Updates the state machine and issues the appropriate command.
        """
        self._frame_count += 1

        if self.state == MissionState.SEARCH:
            self._handle_search(detection)

        elif self.state == MissionState.APPROACH:
            self._handle_approach(detection)

        elif self.state == MissionState.ALIGN:
            self._handle_align(detection)

        elif self.state == MissionState.LAND:
            self._handle_land(detection)

        elif self.state == MissionState.DONE:
            pass  # Mission complete — do nothing

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_search(self, detection: DetectionFrame):
        """Drone is searching. Check if anything useful is visible."""
        if detection.has_landing_marker:
            print(f"[Mission] Landing marker ID:0 detected! Switching to APPROACH.")
            self._transition(MissionState.APPROACH)

        elif detection.has_target:
            obj = detection.yolo_objects[0]
            print(f"[Mission] Target detected: {obj.class_name} ({obj.confidence:.2f}). Approaching.")
            self._transition(MissionState.APPROACH)

        else:
            # Nothing found — execute search pattern (Phase 4 will expand this)
            if self._frame_count % 60 == 0:   # log every ~2s at 30fps
                print("[Mission] Searching... no marker or target visible.")
            self.control.search_hover()

    def _handle_approach(self, detection: DetectionFrame):
        """Move toward the detected marker or target."""
        marker = detection.landing_marker

        if marker is None and not detection.has_target:
            print("[Mission] Lost sight of target. Returning to SEARCH.")
            self._transition(MissionState.SEARCH)
            return

        if marker:
            # Navigate toward marker centre
            cx, cy = marker.centre
            self._velocity_toward_pixel(cx, cy)

            # Check if we're close enough to start aligning
            dx = abs(cx - self.FRAME_CX)
            dy = abs(cy - self.FRAME_CY)
            if dx < self.ALIGN_THRESHOLD_PX * 2 and dy < self.ALIGN_THRESHOLD_PX * 2:
                print("[Mission] Marker centred enough. Switching to ALIGN.")
                self._transition(MissionState.ALIGN)
        else:
            # YOLO target — approach it
            obj = detection.yolo_objects[0]
            self._velocity_toward_pixel(*obj.centre)

    def _handle_align(self, detection: DetectionFrame):
        """Fine-tune alignment over landing marker before descent."""
        marker = detection.landing_marker

        if marker is None:
            print("[Mission] Lost landing marker during alignment. Going back to APPROACH.")
            self._transition(MissionState.APPROACH)
            return

        cx, cy = marker.centre
        dx = cx - self.FRAME_CX
        dy = cy - self.FRAME_CY

        if abs(dx) < self.ALIGN_THRESHOLD_PX and abs(dy) < self.ALIGN_THRESHOLD_PX:
            print(f"[Mission] Aligned! Marker at ({cx},{cy}). Starting LAND.")
            self._transition(MissionState.LAND)
        else:
            # Small corrective nudge
            self._velocity_toward_pixel(cx, cy, speed_scale=0.3)
            print(f"[Mission] Aligning... offset dx={dx} dy={dy}")

    def _handle_land(self, detection: DetectionFrame):
        """Descend slowly while keeping marker centred."""
        marker = detection.landing_marker

        if marker:
            cx, cy = marker.centre
            dx = cx - self.FRAME_CX
            dy = cy - self.FRAME_CY
            # Lateral correction + slow descent
            self.control.land_with_correction(dx=dx, dy=dy)
            print(f"[Mission] Landing... offset dx={dx} dy={dy}")
        else:
            # Marker lost close to ground — continue blind descent
            print("[Mission] Marker lost during landing — continuing descent.")
            self.control.land()

        # Phase 3: detect ground contact via Pixhawk landed state
        # For now we just simulate completion after N frames
        if self._frame_count > 30:
            print("[Mission] Landed! Mission DONE.")
            self._transition(MissionState.DONE)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: MissionState):
        print(f"[Mission] {self.state.name} → {new_state.name}")
        self.state = new_state
        self._frame_count = 0   # reset counter on each state entry

    def _velocity_toward_pixel(self, px: int, py: int, speed_scale: float = 1.0):
        """
        Convert pixel offset from frame centre into velocity commands.
        
        Pixel offset → velocity is a proportional controller (P-controller).
        Tune GAIN to match your drone's responsiveness.
        
        +x in image  = drone moves RIGHT  → positive vy
        +y in image  = drone moves DOWN in image = drone moves FORWARD (camera points down)
        """
        GAIN = 0.005  # tune this: metres/s per pixel of error

        dx = px - self.FRAME_CX
        dy = py - self.FRAME_CY

        vx = dy * GAIN * speed_scale   # forward/back (camera y → body x)
        vy = dx * GAIN * speed_scale   # left/right   (camera x → body y)

        self.control.set_velocity(vx=vx, vy=vy, vz=0.0)