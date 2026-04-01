"""
comms/data_transfer.py
-----------------------
Jetson-side server — feeds the laptop dashboard.

Serves:
  HTTP :8080 /stream   → MJPEG live camera stream
  UDP  :9000           → JSON telemetry packets (10Hz)
  HTTP :8081 /list     → list of captured feature images
  HTTP :8081 /image/<name> → download a feature image

Run on Jetson:
    python3 comms/data_transfer.py

This satisfies IRoC-U Qualification Round Task 7: Data Transfer Verification.
"""

import cv2
import json
import os
import socket
import threading
import time
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

logger = logging.getLogger(__name__)

# ── Shared state (written by main loop, read by server threads) ───────────────

_state = {
    'frame':        None,      # latest BGR frame
    'frame_lock':   threading.Lock(),
    'telem':        {          # latest telemetry — update from your main loop
        'altitude_m':  0.0,
        'lidar_cm':    0,
        'pos_n':       0.0,
        'pos_e':       0.0,
        'state':       'IDLE',
        'battery_pct': 100,
        'armed':       False,
        'timestamp':   0.0,
    },
    'telem_lock':   threading.Lock(),
    'images_dir':   'captured_images',
}

os.makedirs(_state['images_dir'], exist_ok=True)


# ── Public API — call these from your main loop ───────────────────────────────

def update_frame(frame):
    """Call this every loop iteration with the latest camera frame."""
    with _state['frame_lock']:
        _state['frame'] = frame.copy() if frame is not None else None


def update_telemetry(**kwargs):
    """
    Update telemetry fields. Call from main loop with current values.
    Example:
        update_telemetry(altitude_m=2.5, lidar_cm=250, state='HOVER', battery_pct=85)
    """
    with _state['telem_lock']:
        _state['telem'].update(kwargs)
        _state['telem']['timestamp'] = time.time()


def save_feature_image(frame, label: str = "feature") -> str:
    """
    Save a detected feature image for transfer to base station.
    Returns the saved filename.
    """
    ts       = int(time.time())
    filename = f"{label}_{ts}.jpg"
    path     = os.path.join(_state['images_dir'], filename)
    cv2.imwrite(path, frame)
    logger.info("Feature image saved: %s", path)
    return filename


# ── MJPEG stream handler ──────────────────────────────────────────────────────

class MJPEGHandler(BaseHTTPRequestHandler):
    """Serves MJPEG stream on /stream, status on /status."""

    def log_message(self, format, *args):
        pass   # suppress access log spam

    def do_GET(self):
        if self.path == '/stream':
            self._serve_mjpeg()
        elif self.path == '/status':
            self._serve_status()
        else:
            self.send_error(404)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]

        try:
            while True:
                with _state['frame_lock']:
                    frame = _state['frame']

                if frame is None:
                    # Send a placeholder black frame
                    frame = _placeholder_frame()

                # Resize for stream efficiency
                h, w = frame.shape[:2]
                if w > 640:
                    frame = cv2.resize(frame, (640, 480))

                ok, jpg = cv2.imencode('.jpg', frame, encode_params)
                if not ok:
                    time.sleep(0.033)
                    continue

                jpg_bytes = jpg.tobytes()
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(jpg_bytes)}\r\n\r\n'.encode())
                self.wfile.write(jpg_bytes)
                self.wfile.write(b'\r\n')
                self.wfile.flush()

                time.sleep(0.033)   # ~30fps cap

        except (BrokenPipeError, ConnectionResetError):
            pass   # client disconnected

    def _serve_status(self):
        with _state['telem_lock']:
            telem = dict(_state['telem'])
        body = json.dumps(telem).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Image server ──────────────────────────────────────────────────────────────

class ImageHandler(BaseHTTPRequestHandler):
    """Serves feature images on :8081."""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/list':
            self._serve_list()
        elif self.path.startswith('/image/'):
            self._serve_image(self.path[7:])
        else:
            self.send_error(404)

    def _serve_list(self):
        d = _state['images_dir']
        try:
            images = sorted([
                f for f in os.listdir(d)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        except Exception:
            images = []
        body = json.dumps({'images': images}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_image(self, name: str):
        # Sanitise filename — no path traversal
        name = os.path.basename(name)
        path = os.path.join(_state['images_dir'], name)
        if not os.path.exists(path):
            self.send_error(404)
            return
        with open(path, 'rb') as f:
            data = f.read()
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ── UDP telemetry sender ──────────────────────────────────────────────────────

class TelemetrySender:
    """Broadcasts JSON telemetry packets to all listeners on the network."""

    def __init__(self, port: int = 9000, broadcast: bool = True):
        self.port      = port
        self.broadcast = broadcast
        self._sock     = None
        self._thread   = None
        self._running  = False

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self.broadcast:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._running = True
        self._thread  = threading.Thread(
            target=self._send_loop, daemon=True, name="telem-tx"
        )
        self._thread.start()
        print(f"[TelemetrySender] Broadcasting on UDP :{self.port}")

    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()

    def _send_loop(self):
        target = ('<broadcast>', self.port) if self.broadcast else ('255.255.255.255', self.port)
        while self._running:
            try:
                with _state['telem_lock']:
                    packet = json.dumps(_state['telem']).encode()
                self._sock.sendto(packet, target)
            except Exception as exc:
                logger.debug("Telem send error: %s", exc)
            time.sleep(0.1)   # 10Hz


# ── Server manager ────────────────────────────────────────────────────────────

class StreamServer:
    """
    Manages all server threads. One instance per Jetson run.

    Usage in main.py:
        from comms.stream_server import StreamServer, update_frame, update_telemetry

        server = StreamServer()
        server.start()

        while True:
            frame = cam.read()
            update_frame(frame)
            update_telemetry(altitude_m=ctrl.get_altitude(), state=mission.state.name)
            detection = detector.process(frame)
            mission.update(detection)

        server.stop()
    """

    def __init__(self,
                 stream_port: int = 8080,
                 image_port:  int = 8081,
                 telem_port:  int = 9000):
        self.stream_port = stream_port
        self.image_port  = image_port
        self.telem_port  = telem_port

        self._stream_server  = None
        self._image_server   = None
        self._telem_sender   = TelemetrySender(port=telem_port)
        self._threads        = []

    def start(self):
        # MJPEG stream server
        self._stream_server = HTTPServer(('0.0.0.0', self.stream_port), MJPEGHandler)
        t1 = threading.Thread(
            target=self._stream_server.serve_forever,
            daemon=True, name="mjpeg-server"
        )
        t1.start()
        self._threads.append(t1)

        # Image server
        self._image_server = HTTPServer(('0.0.0.0', self.image_port), ImageHandler)
        t2 = threading.Thread(
            target=self._image_server.serve_forever,
            daemon=True, name="image-server"
        )
        t2.start()
        self._threads.append(t2)

        # Telemetry UDP sender
        self._telem_sender.start()

        # Print Jetson's IP so you know what to enter in the dashboard
        ip = self._get_local_ip()
        print(f"\n[StreamServer] Started")
        print(f"  Camera stream : http://{ip}:{self.stream_port}/stream")
        print(f"  Images        : http://{ip}:{self.image_port}/list")
        print(f"  Telemetry     : UDP broadcast :{self.telem_port}")
        print(f"\n  On your laptop run:")
        print(f"  python3 base_station/dashboard.py --jetson-ip {ip}\n")

    def stop(self):
        if self._stream_server:
            self._stream_server.shutdown()
        if self._image_server:
            self._image_server.shutdown()
        self._telem_sender.stop()
        print("[StreamServer] Stopped.")

    @staticmethod
    def _get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return '0.0.0.0'


# ── Helper ────────────────────────────────────────────────────────────────────

def _placeholder_frame() -> 'np.ndarray':
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 30, 40)
    cv2.putText(frame, "Camera initialising...", (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 120), 1)
    return frame


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np
    logging.basicConfig(level=logging.INFO)

    server = StreamServer()
    server.start()

    print("Streaming test pattern. Press Ctrl+C to stop.\n")
    frame_n = 0
    try:
        while True:
            # Fake animated frame for testing
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (20, 20, 30)
            cv2.putText(frame, f"ASCEND Test Stream  frame={frame_n}",
                        (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 120), 2)
            update_frame(frame)
            update_telemetry(
                altitude_m  = 2.0 + 0.1 * (frame_n % 20),
                lidar_cm    = 200 - frame_n % 50,
                state       = 'SEARCH',
                battery_pct = max(0, 100 - frame_n // 10),
                armed       = True,
            )
            frame_n += 1
            time.sleep(0.033)
    except KeyboardInterrupt:
        server.stop()