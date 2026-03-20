"""
mission/mission.py  —  Precision Landing v2
--------------------------------------------
Improvements over v1:
  1. Dead zone          — ignore tiny errors (< DEAD_ZONE_PX pixels)
  2. P-controller       — velocity proportional to error, clamped to max speed
  3. Low-pass filter    — smooth out noisy dx/dy from camera
  4. Adaptive descent   — only descend when centred, slower when very close
  5. Stability check    — require N consecutive centred frames before DONE

State machine:
    SEARCH → APPROACH → ALIGN → LAND → DONE
"""

from enum import Enum, auto
from typing import Optional
from vision.detection import DetectionFrame, ArUcoResult


# ---------------------------------------------------------------------------
# Tuning parameters — adjust these for your real drone
# ---------------------------------------------------------------------------

# Frame centre (pixels) — matches your camera resolution
FRAME_CX = 320
FRAME_CY = 240

# Dead zone — errors smaller than this are ignored (no movement command sent)
# Prevents jitter when marker is almost centred
DEAD_ZONE_PX = 8

# P-controller gain — converts pixel error to velocity (m/s per pixel)
# Higher = faster response but more oscillation
# Start low and increase carefully on real drone
P_GAIN = 0.004

# Maximum horizontal correction speed (m/s)
# Prevents violent movements when far off-centre
MAX_HORIZ_VEL = 0.3

# Alignment threshold — marker must be within this many pixels to be "aligned"
ALIGN_THRESHOLD_PX = 25

# Stability check — how many consecutive frames must be aligned before landing
# Higher = more stable but slower
STABILITY_FRAMES_ALIGN = 15   # frames needed to enter LAND from ALIGN
STABILITY_FRAMES_LAND  = 20   # frames needed to declare DONE

# Descent speeds (m/s) — positive = downward in NED frame
DESCENT_SLOW = 0.08    # used when close to centre
DESCENT_FAST = 0.15    # used when well centred (not used in current logic)

# Low-pass filter strength (0.0 to 1.0)
# Higher = smoother but slower to react
# 0.7 means: new_value = 0.3 * raw + 0.7 * previous
LOW_PASS_ALPHA = 0.7


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class MissionState(Enum):
    SEARCH  = auto()
    APPROACH = auto()
    ALIGN    = auto()
    LAND     = auto()
    DONE     = auto()


# ---------------------------------------------------------------------------
# Mission controller
# ---------------------------------------------------------------------------

class MissionController:
    """
    Consumes DetectionFrame objects each loop and issues MAVLink commands.

    How to use:
        mission = MissionController(control=mavlink_ctrl)
        while True:
            detection = detector.process(frame)
            mission.update(detection)
    """

    def __init__(self, control):
        self.control = control
        self.state   = MissionState.SEARCH

        # Stability counters
        self._stable_frames = 0    # consecutive frames within alignment threshold
        self._frame_count   = 0    # frames since last state transition

        # Low-pass filter state — starts at frame centre (no error)
        self._filtered_dx = 0.0
        self._filtered_dy = 0.0

    # ------------------------------------------------------------------
    # Main update — call once per frame
    # ------------------------------------------------------------------

    def update(self, detection: DetectionFrame):
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
            pass

    # ------------------------------------------------------------------
    # State: SEARCH
    # ------------------------------------------------------------------

    def _handle_search(self, detection: DetectionFrame):
        if detection.has_landing_marker:
            print(f"[Mission] Landing marker ID:0 detected — switching to APPROACH")
            self._transition(MissionState.APPROACH)
        else:
            if self._frame_count % 60 == 0:
                print("[Mission] Searching... no marker visible")
            self.control.search_hover()

    # ------------------------------------------------------------------
    # State: APPROACH  — fast move toward marker using full P-controller
    # ------------------------------------------------------------------

    def _handle_approach(self, detection: DetectionFrame):
        marker = detection.landing_marker

        if marker is None:
            print("[Mission] Marker lost — returning to SEARCH")
            self._transition(MissionState.SEARCH)
            return

        dx, dy = self._get_filtered_error(marker)
        dist   = (dx**2 + dy**2) ** 0.5

        # Send velocity toward marker
        self._send_corrective_velocity(dx, dy, speed_scale=1.0)

        # Transition to ALIGN when close enough
        if dist < ALIGN_THRESHOLD_PX * 2:
            print(f"[Mission] Close enough (dist={dist:.1f}px) — switching to ALIGN")
            self._transition(MissionState.ALIGN)

    # ------------------------------------------------------------------
    # State: ALIGN  — fine corrections, dead zone active, stability count
    # ------------------------------------------------------------------

    def _handle_align(self, detection: DetectionFrame):
        marker = detection.landing_marker

        if marker is None:
            print("[Mission] Marker lost during alignment — back to APPROACH")
            self._stable_frames = 0
            self._transition(MissionState.APPROACH)
            return

        dx, dy = self._get_filtered_error(marker)
        dist   = (dx**2 + dy**2) ** 0.5

        if dist < DEAD_ZONE_PX:
            # Inside dead zone — stop moving, count stable frames
            self.control.set_velocity(vx=0.0, vy=0.0, vz=0.0)
            self._stable_frames += 1
            print(f"[Mission] Aligned — stable {self._stable_frames}/{STABILITY_FRAMES_ALIGN} frames  (dist={dist:.1f}px)")
        else:
            # Outside dead zone — small corrective nudge
            self._stable_frames = 0   # reset stability counter on any movement
            self._send_corrective_velocity(dx, dy, speed_scale=0.4)
            print(f"[Mission] Aligning — dx={dx:.1f} dy={dy:.1f} dist={dist:.1f}px")

        # Transition to LAND only after sustained stability
        if self._stable_frames >= STABILITY_FRAMES_ALIGN:
            print(f"[Mission] Stable for {STABILITY_FRAMES_ALIGN} frames — switching to LAND")
            self._stable_frames = 0
            self._transition(MissionState.LAND)

    # ------------------------------------------------------------------
    # State: LAND  — controlled descent + micro corrections + stability check
    # ------------------------------------------------------------------

    def _handle_land(self, detection: DetectionFrame):
        marker = detection.landing_marker

        if marker is None:
            # Lost marker near ground — continue blind descent briefly
            print("[Mission] Marker lost during landing — continuing descent")
            self.control.set_velocity(vx=0.0, vy=0.0, vz=DESCENT_SLOW)
            return

        dx, dy = self._get_filtered_error(marker)
        dist   = (dx**2 + dy**2) ** 0.5

        # Adaptive descent:
        #   centred (dead zone) → descend at full rate
        #   slightly off        → descend slower, correct laterally
        #   far off             → hold altitude, correct laterally
        if dist < DEAD_ZONE_PX:
            # Perfectly centred — full descent, no lateral
            vz = DESCENT_SLOW
            vx, vy = 0.0, 0.0
            self._stable_frames += 1
            print(f"[Mission] Landing centred — descending  vz={vz:.2f}  stable={self._stable_frames}/{STABILITY_FRAMES_LAND}")
        elif dist < ALIGN_THRESHOLD_PX:
            # Slightly off — slow descent + correction
            vz     = DESCENT_SLOW * 0.5
            vx, vy = self._compute_velocity(dx, dy, speed_scale=0.3)
            self._stable_frames = 0
            print(f"[Mission] Landing correcting — dx={dx:.1f} dy={dy:.1f} vz={vz:.2f}")
        else:
            # Too far off — hold altitude, correct laterally
            vz     = 0.0
            vx, vy = self._compute_velocity(dx, dy, speed_scale=0.5)
            self._stable_frames = 0
            print(f"[Mission] Landing drifted — holding altitude, correcting  dx={dx:.1f} dy={dy:.1f}")

        self.control.set_velocity(vx=vx, vy=vy, vz=vz)

        # DONE only after sustained centred descent
        if self._stable_frames >= STABILITY_FRAMES_LAND:
            print(f"[Mission] Landed! Stable for {STABILITY_FRAMES_LAND} frames.")
            self._transition(MissionState.DONE)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_filtered_error(self, marker: ArUcoResult):
        """
        Compute pixel error from frame centre, then apply low-pass filter.
        Low-pass filter smooths out noisy marker detection frame-to-frame.

        Raw error:      dx = marker_x - frame_centre_x
        Filtered error: filtered = alpha * previous + (1 - alpha) * raw
        """
        raw_dx = float(marker.centre[0] - FRAME_CX)
        raw_dy = float(marker.centre[1] - FRAME_CY)

        # Low-pass filter — blend new reading with previous filtered value
        self._filtered_dx = LOW_PASS_ALPHA * self._filtered_dx + (1 - LOW_PASS_ALPHA) * raw_dx
        self._filtered_dy = LOW_PASS_ALPHA * self._filtered_dy + (1 - LOW_PASS_ALPHA) * raw_dy

        return self._filtered_dx, self._filtered_dy

    def _compute_velocity(self, dx: float, dy: float, speed_scale: float = 1.0):
        """
        P-controller: velocity = gain * error, clamped to max speed.

        dx positive = marker is RIGHT of centre → drone moves RIGHT → positive vy
        dy positive = marker is BELOW centre   → drone moves FORWARD → positive vx
        (assumes camera points downward)
        """
        vx = P_GAIN * dy * speed_scale
        vy = P_GAIN * dx * speed_scale

        # Clamp to max velocity
        vx = max(-MAX_HORIZ_VEL, min(MAX_HORIZ_VEL, vx))
        vy = max(-MAX_HORIZ_VEL, min(MAX_HORIZ_VEL, vy))

        return vx, vy

    def _send_corrective_velocity(self, dx: float, dy: float, speed_scale: float = 1.0):
        """Send horizontal correction. Dead zone applies — no command if error is tiny."""
        dist = (dx**2 + dy**2) ** 0.5

        if dist < DEAD_ZONE_PX:
            self.control.set_velocity(vx=0.0, vy=0.0, vz=0.0)
            return

        vx, vy = self._compute_velocity(dx, dy, speed_scale)
        self.control.set_velocity(vx=vx, vy=vy, vz=0.0)

    def _transition(self, new_state: MissionState):
        print(f"[Mission] {self.state.name} → {new_state.name}")
        self.state        = new_state
        self._frame_count = 0
        self._stable_frames = 0
        # Reset filter on state change so old error doesn't bleed into new state
        self._filtered_dx = 0.0
        self._filtered_dy = 0.0