"""
main.py — Phase 3
------------------
Integrates:
  - Camera (CSI or video file fallback)
  - Vision (ArUco detection)
  - Mission state machine
  - MAVLink control (real or simulated)
  - Stream server (live video + telemetry to laptop dashboard)
  - LiDAR altitude (when sensor is connected)

Run modes:
    python3 main.py                  # simulation, video file camera
    python3 main.py --real-mavlink   # real Pixhawk via UART
    python3 main.py --csi-camera     # use CSI camera (needs barrel jack power)
    python3 main.py --no-stream      # disable WiFi streaming

IRoC-U 2026 Task coverage:
    Task 7 (Data Transfer) ← stream server always running
    Task 2 (Hover)         ← LiDAR altitude hold via MAVLink
"""

import argparse
import cv2
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('main')


def main():
    parser = argparse.ArgumentParser(description='ASCEND Autonomous Drone — Phase 3')
    parser.add_argument('--real-mavlink', action='store_true',
                        help='Connect to real Pixhawk (default: simulation)')
    parser.add_argument('--mavlink-port', default='/dev/ttyTHS1',
                        help='MAVLink port (default: /dev/ttyTHS1)')
    parser.add_argument('--mavlink-udp', action='store_true',
                        help='Use UDP for MAVLink (SITL/QGC bridge mode)')
    parser.add_argument('--csi-camera', action='store_true',
                        help='Use CSI camera (default: video file)')
    parser.add_argument('--video', default='test.mp4',
                        help='Video file for testing (default: test.mp4)')
    parser.add_argument('--no-stream', action='store_true',
                        help='Disable WiFi streaming to laptop')
    parser.add_argument('--lidar', action='store_true',
                        help='Enable TF-Luna LiDAR on /dev/ttyTHS1')
    args = parser.parse_args()

    print("=" * 55)
    print("  ASCEND Autonomous Drone — Phase 3")
    print("=" * 55)

    # ── 1. Camera ─────────────────────────────────────────────────────────────
    from vision.camera import Camera

    cam_source = 0 if not args.csi_camera else 'csi'
    cam = Camera(source=cam_source, width=640, height=480)
    try:
        cam.start()
    except RuntimeError as e:
        print(f"\n[Main] Camera failed: {e}")
        if args.csi_camera:
            print("[Main] Falling back to video file...")
            cam = Camera(source=args.video)
            cam.start()
        else:
            sys.exit(1)

    # ── 2. Vision ─────────────────────────────────────────────────────────────
    from vision.detection import VisionDetector
    detector = VisionDetector(use_yolo=False)

    # ── 3. MAVLink ────────────────────────────────────────────────────────────
    from control.mavlink_control import MavlinkControl

    if args.real_mavlink:
        conn = ('udpin:0.0.0.0:14550' if args.mavlink_udp
                else args.mavlink_port)
        ctrl = MavlinkControl(connection_string=conn, simulate=False)
        ctrl.connect()
    else:
        ctrl = MavlinkControl(simulate=True)

    # ── 4. LiDAR (optional) ───────────────────────────────────────────────────
    lidar = None
    if args.lidar:
        try:
            from sensors.lidar import LidarReader
            lidar = LidarReader(port='/dev/ttyTHS1')
            if not lidar.start():
                print("[Main] LiDAR failed to start — continuing without it")
                lidar = None
            else:
                print("[Main] LiDAR started.")
        except Exception as e:
            print(f"[Main] LiDAR error: {e}")

    # ── 5. Mission ────────────────────────────────────────────────────────────
    from mission.mission import MissionController
    mission = MissionController(control=ctrl)

    # ── 6. Stream server ──────────────────────────────────────────────────────
    stream_server = None
    if not args.no_stream:
        try:
            from comms.stream_server import StreamServer, update_frame, update_telemetry
            stream_server = StreamServer()
            stream_server.start()
        except Exception as e:
            print(f"[Main] Stream server failed: {e} — continuing without streaming")
            update_frame    = lambda f: None
            update_telemetry = lambda **kw: None
    else:
        update_frame     = lambda f: None
        update_telemetry = lambda **kw: None

    print("\n[Main] All systems ready. Running mission loop.\n")

    # ── 7. Main loop ──────────────────────────────────────────────────────────
    frame_count = 0
    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("[Main] No frame — stopping.")
                break

            # Vision
            detection = detector.process(frame)

            # LiDAR
            lidar_cm = lidar.get_distance() if lidar else None

            # Mission update
            mission.update(detection)

            # Stream update
            annotated = detection.frame_rgb if detection.frame_rgb is not None else frame
            update_frame(annotated)
            update_telemetry(
                altitude_m  = ctrl.get_altitude(),
                lidar_cm    = lidar_cm or 0,
                pos_n       = ctrl.get_position()[0],
                pos_e       = ctrl.get_position()[1],
                state       = mission.state.name,
                battery_pct = 85,   # Phase 3+: read from MAVLink BATTERY_STATUS
                armed       = False, # Phase 3+: read from ctrl
            )

            # Log every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                lidar_str = f"{lidar_cm}cm" if lidar_cm else "N/A"
                logger.info(
                    "Frame %d | State: %s | Markers: %d | LiDAR: %s | Alt: %.2fm",
                    frame_count,
                    mission.state.name,
                    len(detection.aruco_markers),
                    lidar_str,
                    ctrl.get_altitude(),
                )

            # Stop after 600 frames in test mode
            if not args.real_mavlink and frame_count >= 600:
                print("[Main] Test complete (600 frames).")
                break

    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C — shutting down.")

    finally:
        cam.stop()
        if lidar:
            lidar.stop()
        if stream_server:
            stream_server.stop()
        if args.real_mavlink:
            ctrl.close()
        print("[Main] Done.")


if __name__ == '__main__':
    main()