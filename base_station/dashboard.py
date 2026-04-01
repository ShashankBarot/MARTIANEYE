"""
base_station/dashboard.py
--------------------------
Laptop-side base station dashboard for IRoC-U 2026.

Receives telemetry + images from Jetson Nano over WiFi and displays:
  - Live camera stream (MJPEG over HTTP)
  - Telemetry: altitude, position, state, LiDAR distance
  - Battery status
  - Detected feature images (for IRoC-U validation task)
  - Connection status

This covers IRoC-U Qualification Round Task 7: Data Transfer Verification.

Run on your LAPTOP (Windows/Linux/Mac), not on Jetson:
    pip install opencv-python numpy requests
    python3 base_station/dashboard.py --jetson-ip 192.168.1.x

The Jetson runs comms/stream_server.py to serve the data.

Architecture:
    Jetson (stream_server.py) ──WiFi──► Laptop (dashboard.py)
         HTTP :8080  → MJPEG video stream
         UDP  :9000  → telemetry JSON packets
         HTTP :8081  → image transfer (detected features)
"""

import argparse
import json
import socket
import threading
import time
import sys
import os
from datetime import datetime
from typing import Optional

try:
    import cv2
    import numpy as np
    import requests
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Install dependencies: pip install opencv-python numpy requests")
    sys.exit(1)


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_JETSON_IP   = "192.168.1.100"   # change to your Jetson's IP
STREAM_PORT         = 8080              # MJPEG stream port
TELEM_PORT          = 9000              # UDP telemetry port
IMAGE_PORT          = 8081              # HTTP image transfer port
SAVE_DIR            = "received_images" # where to save feature images


# ── Telemetry receiver (UDP) ──────────────────────────────────────────────────

class TelemetryReceiver:
    """Listens for UDP JSON telemetry packets from Jetson."""

    def __init__(self, port: int = TELEM_PORT):
        self.port = port
        self._sock    = None
        self._thread  = None
        self._running = False
        self._lock    = threading.Lock()
        self._latest  = {
            'altitude_m':   0.0,
            'pos_n':        0.0,
            'pos_e':        0.0,
            'lidar_cm':     0,
            'state':        'UNKNOWN',
            'battery_pct':  0,
            'armed':        False,
            'timestamp':    0.0,
            'connected':    False,
        }
        self._last_rx = 0.0

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(('0.0.0.0', self.port))
        self._sock.settimeout(1.0)
        self._running = True
        self._thread  = threading.Thread(
            target=self._recv_loop, daemon=True, name="telem-rx"
        )
        self._thread.start()
        print(f"[Telemetry] Listening on UDP :{self.port}")

    def stop(self):
        self._running = False
        if self._sock:
            self._sock.close()

    def get(self) -> dict:
        with self._lock:
            d = dict(self._latest)
            d['connected'] = (time.time() - self._last_rx) < 3.0
            return d

    def _recv_loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(4096)
                packet = json.loads(data.decode())
                with self._lock:
                    self._latest.update(packet)
                    self._last_rx = time.time()
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                pass
            except Exception as exc:
                if self._running:
                    print(f"[Telemetry] Error: {exc}")


# ── MJPEG stream receiver ─────────────────────────────────────────────────────

class StreamReceiver:
    """
    Connects to MJPEG stream from Jetson and provides latest frame.
    Falls back gracefully if stream is unavailable.
    """

    def __init__(self, url: str):
        self.url      = url
        self._frame   = None
        self._lock    = threading.Lock()
        self._thread  = None
        self._running = False
        self._connected = False

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._stream_loop, daemon=True, name="stream-rx"
        )
        self._thread.start()

    def stop(self):
        self._running = False

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def connected(self) -> bool:
        return self._connected

    def _stream_loop(self):
        while self._running:
            try:
                resp = requests.get(self.url, stream=True, timeout=5)
                self._connected = True
                print(f"[Stream] Connected to {self.url}")

                buf = bytes()
                for chunk in resp.iter_content(chunk_size=4096):
                    if not self._running:
                        break
                    buf += chunk

                    # Find JPEG frame boundaries
                    start = buf.find(b'\xff\xd8')
                    end   = buf.find(b'\xff\xd9')
                    if start != -1 and end != -1 and end > start:
                        jpg  = buf[start:end+2]
                        buf  = buf[end+2:]
                        arr  = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._frame = frame

            except requests.exceptions.ConnectionError:
                self._connected = False
                print(f"[Stream] Cannot connect to {self.url} — retrying in 3s")
                time.sleep(3)
            except Exception as exc:
                self._connected = False
                print(f"[Stream] Error: {exc}")
                time.sleep(3)


# ── Image receiver (HTTP download) ───────────────────────────────────────────

class ImageReceiver:
    """Polls Jetson for newly detected feature images and saves them."""

    def __init__(self, base_url: str, save_dir: str):
        self.base_url = base_url
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self._received = []
        self._lock     = threading.Lock()
        self._thread   = None
        self._running  = False

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop, daemon=True, name="image-rx"
        )
        self._thread.start()

    def stop(self):
        self._running = False

    def get_received(self) -> list:
        with self._lock:
            return list(self._received)

    def _poll_loop(self):
        seen = set()
        while self._running:
            try:
                resp = requests.get(f"{self.base_url}/list", timeout=3)
                images = resp.json().get('images', [])
                for img_name in images:
                    if img_name not in seen:
                        self._download(img_name)
                        seen.add(img_name)
            except Exception:
                pass
            time.sleep(2.0)

    def _download(self, img_name: str):
        try:
            resp = requests.get(f"{self.base_url}/image/{img_name}", timeout=5)
            path = os.path.join(self.save_dir, img_name)
            with open(path, 'wb') as f:
                f.write(resp.content)
            with self._lock:
                self._received.append(path)
            print(f"[Images] Saved: {path}")
        except Exception as exc:
            print(f"[Images] Download failed: {exc}")


# ── Dashboard renderer ────────────────────────────────────────────────────────

class Dashboard:
    """
    Renders the base station dashboard using OpenCV windows.
    Headless-safe option: set HEADLESS=True to save frames to disk instead.
    """

    HEADLESS = False   # set True if running without display

    def __init__(self, jetson_ip: str):
        self.jetson_ip = jetson_ip

        stream_url = f"http://{jetson_ip}:{STREAM_PORT}/stream"
        image_url  = f"http://{jetson_ip}:{IMAGE_PORT}"

        self.telem   = TelemetryReceiver(port=TELEM_PORT)
        self.stream  = StreamReceiver(url=stream_url)
        self.images  = ImageReceiver(base_url=image_url, save_dir=SAVE_DIR)

        self._running = False

    def start(self):
        print(f"\n{'='*55}")
        print(f"  IRoC-U 2026 Base Station Dashboard")
        print(f"  Jetson IP : {self.jetson_ip}")
        print(f"  Stream    : http://{self.jetson_ip}:{STREAM_PORT}/stream")
        print(f"  Telemetry : UDP :{TELEM_PORT}")
        print(f"  Images    : http://{self.jetson_ip}:{IMAGE_PORT}")
        print(f"{'='*55}\n")

        self.telem.start()
        self.stream.start()
        self.images.start()
        self._running = True

    def stop(self):
        self._running = False
        self.telem.stop()
        self.stream.stop()
        self.images.stop()
        cv2.destroyAllWindows()

    def run(self):
        """Main display loop — runs on main thread."""
        self.start()
        print("Press Q in the dashboard window to quit.\n")

        while self._running:
            # Get latest data
            telem = self.telem.get()
            frame = self.stream.get_frame()

            # Build display frame
            display = self._build_display(frame, telem)

            if self.HEADLESS:
                # Save to disk instead of showing window
                ts  = datetime.now().strftime("%H%M%S")
                cv2.imwrite(f"dashboard_{ts}.jpg", display)
                time.sleep(0.5)
            else:
                cv2.imshow("IRoC-U 2026 Base Station", display)
                key = cv2.waitKey(33) & 0xFF   # ~30fps
                if key == ord('q') or key == 27:
                    break

            # Print telemetry to terminal (always, regardless of display mode)
            self._print_telem(telem)

        self.stop()

    def _build_display(self, frame: Optional[np.ndarray], telem: dict) -> np.ndarray:
        """Compose a polished dashboard image."""
        W, H = 1280, 720
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # Background gradient + subtle grid lines for depth
        top = np.array((25, 20, 28), dtype=np.float32)
        bottom = np.array((13, 10, 16), dtype=np.float32)
        for y in range(H):
            a = y / max(1, H - 1)
            color = ((1.0 - a) * top + a * bottom).astype(np.uint8)
            canvas[y, :] = color
        for x in range(0, W, 64):
            cv2.line(canvas, (x, 0), (x, H), (30, 24, 34), 1)
        for y in range(0, H, 48):
            cv2.line(canvas, (0, y), (W, y), (26, 20, 30), 1)

        # Ambient radial glow around center to reduce flatness
        cx, cy = W // 2, H // 2 + 40
        for r in range(420, 140, -24):
            alpha = max(0.0, min(1.0, (420 - r) / 420.0))
            color = (
                int(35 + 35 * alpha),
                int(22 + 30 * alpha),
                int(45 + 30 * alpha),
            )
            cv2.circle(canvas, (cx, cy), r, color, 1)

        # Top bar
        cv2.rectangle(canvas, (0, 0), (W, 60), (31, 24, 34), -1)
        cv2.line(canvas, (0, 60), (W, 60), (58, 44, 62), 1)
        self._text(canvas, "IRoC-U 2026", (22, 39), scale=1.0, color=(232, 230, 240), bold=True)
        self._text(canvas, "ASCEND  Base Station", (208, 38), scale=0.72, color=(206, 194, 214))
        self._text(canvas, "Task 7  Data Transfer", (470, 38), scale=0.72, color=(180, 166, 192))

        # Layout
        left_x, left_y, left_w, left_h = 22, 82, 872, 614
        right_x, right_y, right_w = 912, 82, 346

        # Video panel
        self._card(canvas, left_x, left_y, left_w, left_h, fill=(40, 30, 41), border=(72, 60, 80))
        self._text(canvas, "LIVE FEED", (left_x + 16, left_y + 33), scale=0.56, color=(194, 184, 206), bold=True)

        frame_x, frame_y = left_x + 16, left_y + 46
        frame_w, frame_h = left_w - 32, left_h - 64

        if frame is not None:
            fh, fw = frame.shape[:2]
            # Fit while preserving aspect ratio
            scale = min(frame_w / max(1, fw), frame_h / max(1, fh))
            new_w, new_h = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (new_w, new_h))
            x_off = frame_x + (frame_w - new_w) // 2
            y_off = frame_y + (frame_h - new_h) // 2
            self._card(canvas, frame_x, frame_y, frame_w, frame_h, fill=(16, 14, 20), border=(50, 44, 58))
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        else:
            self._card(canvas, frame_x, frame_y, frame_w, frame_h, fill=(16, 14, 20), border=(50, 44, 58))
            self._text(canvas, "NO VIDEO SIGNAL", (frame_x + 250, frame_y + frame_h // 2 - 12),
                       scale=0.82, color=(132, 124, 140), bold=True)
            if not self.stream.connected:
                self._text(canvas, f"Connecting to {self.jetson_ip}:{STREAM_PORT}",
                           (frame_x + 205, frame_y + frame_h // 2 + 20),
                           scale=0.56, color=(104, 98, 114))

        # Telemetry panel
        self._card(canvas, right_x, right_y, right_w, left_h, fill=(31, 25, 34), border=(62, 52, 70))
        self._text(canvas, "TELEMETRY", (right_x + 16, right_y + 32), scale=0.66, color=(212, 206, 220), bold=True)

        rx, ry = right_x + 16, right_y + 58

        # Connection pill
        connected = telem.get('connected', False)
        conn_color = (64, 216, 106) if connected else (72, 92, 236)
        conn_text = "CONNECTED" if connected else "DISCONNECTED"
        self._card(canvas, rx, ry, 194, 38, fill=(44, 40, 50), border=(86, 78, 94))
        pulse = 2 if int(time.time() * 2) % 2 == 0 else 0
        cv2.circle(canvas, (rx + 16, ry + 19), 6 + pulse, conn_color, -1)
        self._text(canvas, conn_text, (rx + 30, ry + 25), scale=0.58, color=conn_color, bold=True)
        ry += 54

        state_colors = {
            'SEARCH':   (34, 204, 255),
            'APPROACH': (82, 218, 255),
            'ALIGN':    (92, 228, 180),
            'LAND':     (76, 226, 118),
            'DONE':     (95, 240, 130),
            'UNKNOWN':  (180, 180, 180),
        }
        state = telem.get('state', 'UNKNOWN')
        sc = state_colors.get(state, (220, 220, 220))
        self._text(canvas, "MISSION STATE", (rx, ry), scale=0.5, color=(160, 154, 172))
        self._text(canvas, state, (rx, ry + 34), scale=1.2, color=sc, bold=True)
        ry += 54

        self._text(canvas, "FLIGHT DATA", (rx, ry), scale=0.5, color=(160, 154, 172))
        ry += 18

        metric_w = right_w - 32
        row_h = 44
        rows = [
            ("ALTITUDE", f"{telem.get('altitude_m', 0):.2f} m", (126, 214, 255), "alt"),
            ("LIDAR", f"{telem.get('lidar_cm', 0)} cm", (118, 236, 180), "lidar"),
            ("POS N", f"{telem.get('pos_n', 0):.2f} m", (164, 198, 255), "north"),
            ("POS E", f"{telem.get('pos_e', 0):.2f} m", (206, 184, 255), "east"),
        ]
        for label, value, color, icon in rows:
            self._metric_row(canvas, rx, ry, metric_w, row_h, label, value, color, icon)
            ry += row_h + 8

        # Battery with progress bar
        battery_pct = int(telem.get('battery_pct', 0))
        if battery_pct > 50:
            battery_color = (64, 226, 118)
        elif battery_pct > 20:
            battery_color = (0, 196, 255)
        else:
            battery_color = (64, 86, 232)

        self._card(canvas, rx, ry, metric_w, 66, fill=(40, 34, 46), border=(78, 68, 86))
        self._icon(canvas, "battery", rx + 18, ry + 18, battery_color)
        self._text(canvas, "BATTERY", (rx + 36, ry + 24), scale=0.5, color=(152, 144, 166), font="label")
        self._text(canvas, f"{battery_pct}%", (rx + metric_w - 86, ry + 24), scale=0.82, color=battery_color, bold=True, font="title")
        self._progress_bar(canvas, x=rx + 12, y=ry + 38, w=metric_w - 24, h=14,
                           progress=max(0.0, min(1.0, battery_pct / 100.0)), color=battery_color)
        ry += 78

        # Armed + images cards
        armed = telem.get('armed', False)
        n_img = len(self.images.get_received())
        half_w = (metric_w - 8) // 2
        self._card(canvas, rx, ry, half_w, 52, fill=(40, 34, 46), border=(78, 68, 86))
        self._icon(canvas, "armed", rx + 14, ry + 16, (64, 86, 232) if armed else (128, 120, 138))
        self._text(canvas, "ARMED", (rx + 30, ry + 20), scale=0.44, color=(150, 142, 162), font="label")
        self._text(canvas, "YES" if armed else "NO", (rx + 30, ry + 40), scale=0.64,
                   color=(64, 86, 232) if armed else (170, 164, 178), bold=True, font="title")

        ix = rx + half_w + 8
        self._card(canvas, ix, ry, half_w, 52, fill=(40, 34, 46), border=(78, 68, 86))
        self._icon(canvas, "image", ix + 14, ry + 16, (80, 246, 202) if n_img > 0 else (128, 120, 138))
        self._text(canvas, "IMAGES RX", (ix + 30, ry + 20), scale=0.44, color=(150, 142, 162), font="label")
        self._text(canvas, str(n_img), (ix + 30, ry + 40), scale=0.76,
                   color=(80, 246, 202) if n_img > 0 else (170, 164, 178), bold=True, font="title")

        # Footer hints
        ts = datetime.now().strftime("%H:%M:%S")
        self._text(canvas, f"TIME   {ts}", (right_x + 16, H - 24), scale=0.5, color=(132, 124, 144))
        self._text(canvas, "Press Q or ESC to quit", (left_x + 16, H - 24), scale=0.5, color=(132, 124, 144))
        return canvas

    @staticmethod
    def _text(img, text, pos, scale=0.6, color=(255,255,255), bold=False, font="default"):
        font_map = {
            "default": cv2.FONT_HERSHEY_SIMPLEX,
            "title": cv2.FONT_HERSHEY_DUPLEX,
            "label": cv2.FONT_HERSHEY_COMPLEX_SMALL,
            "mono": cv2.FONT_HERSHEY_PLAIN,
        }
        face = font_map.get(font, cv2.FONT_HERSHEY_SIMPLEX)
        thickness = 2 if bold else 1
        cv2.putText(img, str(text), pos,
                    face, scale, color, thickness,
                    cv2.LINE_AA)

    @staticmethod
    def _progress_bar(img, x: int, y: int, w: int, h: int, progress: float, color=(0, 220, 80)):
        """Draw a thin rounded-looking progress bar for telemetry values."""
        cv2.rectangle(img, (x, y), (x + w, y + h), (58, 52, 64), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (86, 78, 92), 1)
        fill_w = int((w - 2) * max(0.0, min(1.0, progress)))
        if fill_w > 0:
            cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fill_w, y + h - 1), color, -1)

    @staticmethod
    def _card(img, x: int, y: int, w: int, h: int, fill=(35, 30, 40), border=(64, 56, 72)):
        """Draw a card with soft shadow to emulate modern dashboard blocks."""
        shadow = np.zeros_like(img)
        cv2.rectangle(shadow, (x + 4, y + 6), (x + w + 4, y + h + 6), (8, 8, 10), -1)
        img[:] = cv2.addWeighted(img, 1.0, shadow, 0.35, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), border, 1)

    @staticmethod
    def _icon(img, kind: str, x: int, y: int, color=(180, 180, 200)):
        """Draw simple vector icons to improve dashboard readability."""
        if kind == "alt":
            cv2.line(img, (x, y + 12), (x + 8, y), color, 2)
            cv2.line(img, (x + 8, y), (x + 16, y + 12), color, 2)
        elif kind == "lidar":
            cv2.circle(img, (x + 8, y + 8), 6, color, 1)
            cv2.circle(img, (x + 8, y + 8), 2, color, -1)
        elif kind == "north":
            cv2.arrowedLine(img, (x + 8, y + 14), (x + 8, y), color, 2, tipLength=0.4)
        elif kind == "east":
            cv2.arrowedLine(img, (x, y + 8), (x + 14, y + 8), color, 2, tipLength=0.4)
        elif kind == "battery":
            cv2.rectangle(img, (x, y), (x + 16, y + 10), color, 1)
            cv2.rectangle(img, (x + 16, y + 3), (x + 18, y + 7), color, -1)
            cv2.rectangle(img, (x + 2, y + 2), (x + 10, y + 8), color, -1)
        elif kind == "armed":
            cv2.circle(img, (x + 8, y + 8), 6, color, 1)
            cv2.line(img, (x + 5, y + 8), (x + 11, y + 8), color, 2)
            cv2.line(img, (x + 8, y + 5), (x + 8, y + 11), color, 2)
        elif kind == "image":
            cv2.rectangle(img, (x, y), (x + 16, y + 12), color, 1)
            cv2.circle(img, (x + 5, y + 4), 1, color, -1)
            cv2.line(img, (x + 2, y + 10), (x + 7, y + 6), color, 1)
            cv2.line(img, (x + 7, y + 6), (x + 13, y + 10), color, 1)

    def _metric_row(self, img, x: int, y: int, w: int, h: int,
                    label: str, value: str, accent, icon: str):
        """Render one telemetry metric as a compact styled card row."""
        self._card(img, x, y, w, h, fill=(40, 34, 46), border=(78, 68, 86))
        cv2.rectangle(img, (x + 1, y + 1), (x + 4, y + h - 1), accent, -1)
        self._icon(img, icon, x + 12, y + 13, accent)
        self._text(img, label, (x + 34, y + 18), scale=0.44, color=(152, 146, 166), font="label")
        self._text(img, value, (x + w - 118, y + 30), scale=0.62, color=(240, 240, 246), bold=True, font="title")

    @staticmethod
    def _print_telem(telem: dict):
        """Print a compact telemetry line to terminal."""
        connected = "●" if telem.get('connected') else "○"
        print(
            f"\r{connected} "
            f"State:{telem.get('state','?'):8s} "
            f"Alt:{telem.get('altitude_m',0):5.2f}m "
            f"LiDAR:{telem.get('lidar_cm',0):4d}cm "
            f"Bat:{telem.get('battery_pct',0):3d}% "
            f"Armed:{'Y' if telem.get('armed') else 'N'}",
            end='', flush=True
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IRoC-U 2026 Base Station Dashboard")
    parser.add_argument(
        '--jetson-ip', default=DEFAULT_JETSON_IP,
        help=f"Jetson Nano IP address (default: {DEFAULT_JETSON_IP})"
    )
    parser.add_argument(
        '--headless', action='store_true',
        help="Run without display — save frames to disk instead"
    )
    args = parser.parse_args()

    if args.headless:
        Dashboard.HEADLESS = True

    dash = Dashboard(jetson_ip=args.jetson_ip)
    try:
        dash.run()
    except KeyboardInterrupt:
        print("\n\n[Dashboard] Shutting down.")
        dash.stop()


if __name__ == '__main__':
    main()