"""
control/mavlink_control.py — Phase 3
--------------------------------------
Real MAVLink implementation using pymavlink.
Drops in as a direct replacement for the simulation version —
MissionController never needs to change.

Connection modes:
  UART  (Pixhawk physical):  '/dev/ttyTHS1' or '/dev/ttyUSB0'
  UDP   (SITL / QGC bridge): 'udpin:0.0.0.0:14550'
  TCP   (MAVProxy forward):  'tcp:127.0.0.1:5760'

Usage:
    # SIMULATION (no Pixhawk) — run ArduPilot SITL + MAVProxy first
    ctrl = MavlinkControl(connection_string='udpin:0.0.0.0:14550', simulate=False)

    # REAL PIXHAWK via UART on Jetson Nano
    ctrl = MavlinkControl(connection_string='/dev/ttyTHS1', simulate=False)

    ctrl.connect()
    ctrl.arm()
    ctrl.takeoff(alt=2.0)
    ctrl.set_velocity(vx=0.1, vy=0.0, vz=0.0)
    ctrl.land()

Phase 3 checklist:
  [x] connect() — heartbeat wait + mode set
  [x] arm() / disarm()
  [x] takeoff(alt)
  [x] land() / rtl()
  [x] set_velocity(vx, vy, vz) — SET_POSITION_TARGET_LOCAL_NED
  [x] get_altitude() — from GLOBAL_POSITION_INT
  [x] get_position() — local NED from LOCAL_POSITION_NED
  [x] is_landed() — from EXTENDED_SYS_STATE
  [x] search_hover() — hold position
  [x] land_with_correction(dx, dy) — precision landing nudge
  [x] Simulation fallback — set simulate=True for print-only mode
"""

import time
import threading
import math
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ── Simulation stub (used when simulate=True or pymavlink not installed) ─────

class _SimulatedBackend:
    """Print-only backend — identical to old mavlink_control.py behaviour."""
    def __init__(self): pass
    def connect(self): print("[MAVLink-SIM] Connected (simulation)")
    def arm(self): print("[MAVLink-SIM] ARM")
    def disarm(self): print("[MAVLink-SIM] DISARM")
    def takeoff(self, alt): print(f"[MAVLink-SIM] TAKEOFF {alt}m")
    def land(self): print("[MAVLink-SIM] LAND")
    def rtl(self): print("[MAVLink-SIM] RTL")
    def set_velocity(self, vx, vy, vz):
        print(f"[MAVLink-SIM] VEL vx={vx:+.3f} vy={vy:+.3f} vz={vz:+.3f}")
    def get_altitude(self): return 2.5   # fake 2.5m hover
    def get_lidar_cm(self): return 250   # fake 2.5m hover
    def get_position(self): return (0.0, 0.0, -2.5)
    def is_landed(self): return False
    def search_hover(self): print("[MAVLink-SIM] SEARCH HOVER")
    def land_with_correction(self, dx, dy):
        GAIN=0.003; MAX_V=0.3; VZ=0.1
        vx=max(-MAX_V,min(MAX_V,dy*GAIN)); vy=max(-MAX_V,min(MAX_V,dx*GAIN))
        print(f"[MAVLink-SIM] LAND CORRECT vx={vx:+.3f} vy={vy:+.3f} vz=+{VZ:.2f}")
    def send_heartbeat(self): pass
    def close(self): print("[MAVLink-SIM] Closed")


# ── Real MAVLink backend ──────────────────────────────────────────────────────

class _RealBackend:
    """
    pymavlink implementation.
    Handles GUIDED mode, arming, velocity commands, telemetry reads.
    """

    # MAVLink flight modes for ArduCopter
    MODE_GUIDED = 4
    MODE_LAND   = 9
    MODE_RTL    = 6
    MODE_STABILIZE = 0

    def __init__(self, connection_string: str, baudrate: int, timeout: float):
        self.connection_string = connection_string
        self.baudrate  = baudrate
        self.timeout   = timeout
        self._mav      = None
        self._lock     = threading.Lock()

        # Cached telemetry (updated by background thread)
        self._altitude_m   = 0.0
        self._pos_ned      = (0.0, 0.0, 0.0)   # North, East, Down (m)
        self._landed       = True
        self._armed        = False
        self._lidar_cm     = 0
        self._telem_thread = None
        self._running      = False

    def connect(self):
        from pymavlink import mavutil

        logger.info("MAVLink: connecting to %s", self.connection_string)
        print(f"[MAVLink] Connecting to {self.connection_string} ...")

        self._mav = mavutil.mavlink_connection(
            self.connection_string,
            baud=self.baudrate,
        )

        # Wait for heartbeat — confirms Pixhawk is alive
        print("[MAVLink] Waiting for heartbeat...")
        self._mav.wait_heartbeat(timeout=self.timeout)
        print(f"[MAVLink] Heartbeat received from system "
              f"{self._mav.target_system} component {self._mav.target_component}")

        # Start telemetry reader thread
        self._running = True
        self._telem_thread = threading.Thread(
            target=self._telemetry_loop, daemon=True, name="mavlink-telem"
        )
        self._telem_thread.start()

        # Switch to GUIDED mode
        self._set_mode(self.MODE_GUIDED)
        time.sleep(0.5)
        print("[MAVLink] Ready in GUIDED mode.")

    def arm(self):
        logger.info("MAVLink: ARM")
        self._mav.arducopter_arm()
        # Wait for arm confirmation
        for _ in range(30):
            if self._armed:
                print("[MAVLink] Armed.")
                return
            time.sleep(0.2)
        print("[MAVLink] WARNING: arm confirmation timeout")

    def disarm(self):
        self._mav.arducopter_disarm()
        print("[MAVLink] Disarmed.")

    def takeoff(self, alt: float):
        """
        Command takeoff to alt metres above home.
        Must be armed and in GUIDED mode first.
        """
        self._set_mode(self.MODE_GUIDED)
        time.sleep(0.3)

        self._mav.mav.command_long_send(
            self._mav.target_system,
            self._mav.target_component,
            176,   # MAV_CMD_NAV_TAKEOFF
            0,
            0, 0, 0, 0, 0, 0,
            alt
        )
        print(f"[MAVLink] Takeoff → {alt}m")

        # Wait until we reach ~80% of target altitude
        deadline = time.time() + 30
        while time.time() < deadline:
            if self._altitude_m >= alt * 0.8:
                print(f"[MAVLink] Reached {self._altitude_m:.1f}m")
                return
            time.sleep(0.5)
        print("[MAVLink] WARNING: takeoff altitude not confirmed")

    def land(self):
        self._set_mode(self.MODE_LAND)
        print("[MAVLink] LAND mode set.")

    def rtl(self):
        self._set_mode(self.MODE_RTL)
        print("[MAVLink] RTL mode set.")

    def set_velocity(self, vx: float, vy: float, vz: float):
        """
        Send SET_POSITION_TARGET_LOCAL_NED velocity command in body frame.
        vx = forward (m/s), vy = right (m/s), vz = down (m/s, positive = descend)
        """
        if self._mav is None:
            return

        # Type mask: ignore position + acceleration, use velocity only
        # 0b110111111000 = 0x0DF8
        TYPE_MASK = 0x0DF8

        self._mav.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,  # time_boot_ms
            self._mav.target_system,
            self._mav.target_component,
            8,          # MAV_FRAME_BODY_NED (body frame)
            TYPE_MASK,
            0, 0, 0,    # position (ignored)
            vx, vy, vz, # velocity
            0, 0, 0,    # acceleration (ignored)
            0, 0        # yaw, yaw_rate (ignored)
        )

    def get_lidar_cm(self) -> int:
        return self._lidar_cm

    def get_altitude(self) -> float:
        """Return current altitude in metres above home."""
        return self._altitude_m

    def get_position(self) -> Tuple[float, float, float]:
        """Return (north_m, east_m, down_m) relative to home."""
        return self._pos_ned

    def is_landed(self) -> bool:
        return self._landed

    def search_hover(self):
        """Hold current position — send zero velocity."""
        self.set_velocity(0.0, 0.0, 0.0)

    def land_with_correction(self, dx: int, dy: int):
        """
        Precision landing nudge.
        dx, dy = pixel offset from frame centre.
        Converts to small velocity + slow descent.
        """
        GAIN  = 0.003
        MAX_V = 0.3
        VZ    = 0.08   # slow descent 8 cm/s

        vx = max(-MAX_V, min(MAX_V,  dy * GAIN))
        vy = max(-MAX_V, min(MAX_V,  dx * GAIN))
        self.set_velocity(vx, vy, VZ)

    def send_heartbeat(self):
        """Send GCS heartbeat to keep Pixhawk from triggering GCS failsafe."""
        if self._mav:
            self._mav.mav.heartbeat_send(6, 8, 0, 0, 0)  # GCS heartbeat

    def close(self):
        self._running = False
        if self._mav:
            self._mav.close()
        print("[MAVLink] Connection closed.")

    # ── Private ──────────────────────────────────────────────────────────────

    def _set_mode(self, mode_id: int):
        self._mav.mav.set_mode_send(
            self._mav.target_system,
            209,       # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            mode_id
        )

    def _telemetry_loop(self):
        """
        Background thread — reads incoming MAVLink messages and updates
        cached telemetry values.  Never blocks the main control loop.
        """
        while self._running:
            try:
                msg = self._mav.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    continue

                mtype = msg.get_type()

                if mtype == 'GLOBAL_POSITION_INT':
                    # altitude relative to home in mm → convert to m
                    self._altitude_m = msg.relative_alt / 1000.0

                elif mtype == 'LOCAL_POSITION_NED':
                    self._pos_ned = (float(msg.x), float(msg.y), float(msg.z))

                elif mtype == 'EXTENDED_SYS_STATE':
                    # landed_state: 1=UNDEFINED, 2=ON_GROUND, 3=IN_AIR, 4=TAKEOFF, 5=LANDING
                    self._landed = (msg.landed_state == 2)

                elif mtype == 'HEARTBEAT':
                    self._armed = bool(msg.base_mode & 0x80)

                elif mtype == 'RANGEFINDER':
                    self._lidar_cm = int(msg.distance * 100)  # metres -> cm

                elif mtype == 'VFR_HUD':
                    # Better altitude source than GLOBAL_POSITION_INT on bench
                    self._altitude_m = float(msg.alt)

            except Exception as exc:
                logger.debug("Telemetry loop error: %s", exc)
                time.sleep(0.05)


# ── Public interface — this is what mission.py imports ───────────────────────

class MavlinkControl:
    """
    Unified MAVLink interface.

    Set simulate=True  → print-only mode (old behaviour, no Pixhawk needed)
    Set simulate=False → real pymavlink (needs Pixhawk or SITL)

    Connection strings:
        '/dev/ttyTHS1'          — Pixhawk via UART on Jetson Nano
        '/dev/ttyUSB0'          — Pixhawk via USB
        'udpin:0.0.0.0:14550'  — SITL or MAVProxy UDP forward
        'tcp:127.0.0.1:5760'   — MAVProxy TCP bridge
    """

    def __init__(self,
                 connection_string: str = 'udpin:0.0.0.0:14550',
                 baudrate: int = 57600,
                 timeout: float = 30.0,
                 simulate: bool = True):

        self.simulate = simulate

        if simulate:
            self._backend = _SimulatedBackend()
            print("[MavlinkControl] SIMULATION MODE — no Pixhawk needed")
        else:
            try:
                from pymavlink import mavutil  # test import
                self._backend = _RealBackend(connection_string, baudrate, timeout)
                print(f"[MavlinkControl] REAL MODE — {connection_string}")
            except ImportError:
                print("[MavlinkControl] WARNING: pymavlink not found — falling back to simulation")
                self._backend = _SimulatedBackend()
                self.simulate = True

        # Heartbeat thread — keeps GCS failsafe from triggering
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="mavlink-hb"
        )
        self._hb_running = False

    def connect(self):
        """Connect to Pixhawk (or SITL). Call before arm()."""
        self._backend.connect()
        self._hb_running = True
        self._hb_thread.start()

    def arm(self):           self._backend.arm()
    def disarm(self):        self._backend.disarm()
    def takeoff(self, alt):  self._backend.takeoff(alt)
    def hover(self):         self._backend.search_hover()
    def land(self):          self._backend.land()
    def rtl(self):           self._backend.rtl()
    def search_hover(self):  self._backend.search_hover()

    def set_velocity(self, vx: float, vy: float, vz: float):
        self._backend.set_velocity(vx, vy, vz)

    def send_velocity(self, vx: float, vy: float, vz: float):
        self._backend.set_velocity(vx, vy, vz)

    def land_with_correction(self, dx: int, dy: int):
        self._backend.land_with_correction(dx, dy)

    def get_lidar_cm(self) -> int:
        return self._backend.get_lidar_cm()

    def get_altitude(self) -> float:
        return self._backend.get_altitude()

    def get_position(self) -> Tuple[float, float, float]:
        return self._backend.get_position()

    def is_landed(self) -> bool:
        return self._backend.is_landed()

    def close(self):
        self._hb_running = False
        self._backend.close()

    def _heartbeat_loop(self):
        while self._hb_running:
            self._backend.send_heartbeat()
            time.sleep(1.0)