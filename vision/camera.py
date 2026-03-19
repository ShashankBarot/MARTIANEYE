"""
vision/camera.py
----------------
Unified camera source — works with:
  - A video file   (for testing without hardware)
  - A USB camera   (e.g. /dev/video0)
  - Jetson CSI camera (GStreamer pipeline string)

Usage:
    cam = Camera(source="test.mp4")      # video file
    cam = Camera(source=0)               # USB webcam
    cam = Camera(source=Camera.CSI_PIPELINE)  # Jetson CSI

    cam.start()
    frame = cam.read()
    cam.stop()
"""

import cv2
import threading
import time


class Camera:
    """
    Thread-safe camera wrapper.
    
    Runs a background thread that continuously reads frames so your main loop
    is never blocked waiting for the next frame from the hardware.
    """

    # GStreamer pipeline for Jetson Nano CSI camera (IMX219 / Raspberry Pi v2)
    # Change nvarguscamerasrc sensor-id=0 if you have a different sensor.
    CSI_PIPELINE = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, "
        "format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=640, height=480, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
    )

    def __init__(self, source=0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Args:
            source  : int (device index), str (file path or GStreamer pipeline)
            width   : desired capture width in pixels
            height  : desired capture height in pixels
            fps     : desired frames per second
        """
        self.source = source
        self.width  = width
        self.height = height
        self.fps    = fps

        self._cap     = None
        self._frame   = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Open the camera and start background capture thread."""
        if isinstance(self.source, str) and "nvarguscamerasrc" in self.source:
            # CSI camera uses GStreamer — pass CAP_GSTREAMER backend
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"[Camera] Could not open source: {self.source}\n"
                "  → Check the path/device index and try again."
            )

        # Set resolution and FPS (ignored for GStreamer pipelines)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)

        # Grab one frame synchronously so .read() is never None on first call
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("[Camera] Opened but could not read first frame.")
        with self._lock:
            self._frame = frame

        # Start background thread
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[Camera] Started — source: {self.source}")

    def read(self):
        """Return the most recent frame (BGR numpy array), or None if not started."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        """Stop capture thread and release camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        print("[Camera] Stopped.")

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened() and self._running

    # ------------------------------------------------------------------
    # Background capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self):
        """Runs in a separate thread — continuously grabs frames."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                # For video files, loop back to start
                if isinstance(self.source, str):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[Camera] Frame read failed — camera disconnected?")
                    self._running = False
                    break

            with self._lock:
                self._frame = frame

            time.sleep(0.001)  # yield the GIL briefly