"""
main.py — Phase 2
------------------
Full pipeline: Camera → ArUco/YOLO detection → Mission logic → Control.

Run on your Jetson Nano:
    python3 main.py

To test without a real camera, put a video file named test.mp4 in the project root.
To test ArUco detection, generate a test marker image first:
    python3 generate_test_marker.py
"""

import cv2
import sys
from vision.camera import Camera
from vision.detection import VisionDetector
from mission.mission import MissionController
from control.mavlink_control import MavlinkControl


def main():
    print("=" * 50)
    print(" Autonomous Drone — Phase 2 startup")
    print("=" * 50)

    # ---------------------------------------------------------------
    # 1. Camera — swap source= to 0 for USB cam, or CSI pipeline
    # ---------------------------------------------------------------
    cam = Camera(source="test.mp4", width=640, height=480)
    cam.start()

    # ---------------------------------------------------------------
    # 2. Vision detector
    #    use_yolo=False  → ArUco only (faster, no GPU needed)
    #    use_yolo=True   → ArUco + YOLO (requires ~2GB RAM for yolov8n)
    # ---------------------------------------------------------------
    detector = VisionDetector(
        use_yolo=False,
        yolo_confidence=0.5,
        yolo_target_classes=["person"],   # only detect people for now
    )

    # ---------------------------------------------------------------
    # 3. Simulated MAVLink control (Phase 3 will replace this)
    # ---------------------------------------------------------------
    control = MavlinkControl()

    # ---------------------------------------------------------------
    # 4. Mission controller
    # ---------------------------------------------------------------
    mission = MissionController(control=control)

    print("\n[Main] Pipeline ready. Press Q to quit.\n")

    # ---------------------------------------------------------------
    # 5. Main loop
    # ---------------------------------------------------------------
    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("[Main] No frame received from camera.")
                break

            # Run vision
            detection = detector.process(frame)

            # Run mission state machine
            mission.update(detection)

            # Show annotated frame (SSH: disable this or use XRDP)
            if detection.frame_rgb is not None:
                # cv2.imshow("Drone Vision", detection.frame_rgb)
                pass

            # Print summary every 30 frames (~1s at 30fps)
            if mission._frame_count % 30 == 0:
                print(
                    f"[Main] State: {mission.state.name} | "
                    f"ArUco markers: {len(detection.aruco_markers)} | "
                    f"YOLO objects: {len(detection.yolo_objects)}"
                )

            # # Quit on Q
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("[Main] Q pressed — shutting down.")
            #     break
            if mission._frame_count > 300:
                print("[Main] Test complete.")
                break

    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C — shutting down.")

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[Main] Done.")


if __name__ == "__main__":
    main()