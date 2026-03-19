"""
generate_test_marker.py  —  OpenCV 3.x / 4.x / 4.7+ compatible
----------------------------------------------------------------
Generates ArUco marker PNG files you can open on a screen for testing.
Marker ID 0 is your designated landing pad.

Usage:
    python3 generate_test_marker.py

OpenCV 3.x (JetPack 4)  → uses aruco.drawMarker()
OpenCV 4.6              → uses aruco.drawMarker()
OpenCV 4.7+             → uses aruco.generateImageMarker()
"""

import cv2
import cv2.aruco as aruco
import numpy as np

DICT_TYPE  = aruco.DICT_4X4_250
MARKER_IDS = [0, 1, 2]
SIZE_PX    = 400
BORDER_PX  = 40

aruco_dict = aruco.getPredefinedDictionary(DICT_TYPE)

# Detect which generation of marker-generation API is available
# OpenCV 4.7+  → generateImageMarker(dict, id, size)
# OpenCV 3.x / 4.6 → drawMarker(dict, id, size)
_USE_GENERATE = hasattr(aruco, "generateImageMarker")
print(f"OpenCV {cv2.__version__} — using "
      f"{'generateImageMarker' if _USE_GENERATE else 'drawMarker'}")

for marker_id in MARKER_IDS:
    if _USE_GENERATE:
        marker_img = aruco.generateImageMarker(aruco_dict, marker_id, SIZE_PX)
    else:
        marker_img = aruco.drawMarker(aruco_dict, marker_id, SIZE_PX)

    # White border significantly improves detection reliability at corners
    bordered = cv2.copyMakeBorder(
        marker_img, BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX,
        cv2.BORDER_CONSTANT, value=255
    )

    filename = f"aruco_marker_ID{marker_id}.png"
    cv2.imwrite(filename, bordered)
    print(f"  Saved {filename}  ({bordered.shape[1]}x{bordered.shape[0]}px)")

print()
print("Done! Open aruco_marker_ID0.png on any screen.")
print("Hold it in front of your camera — the detector will see it as the landing marker.")