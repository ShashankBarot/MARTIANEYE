class MavlinkControl:

    def __init__(self):
        print("[Mavlink] Initialized (SIMULATION MODE)")

    def arm(self):
        print("[MAVLink] ARM")

    def takeoff(self, alt):
        print(f"[MAVLink] TAKEOFF to {alt}m")

    def hover(self):
        print("[MAVLink] HOVER")

    def land(self):
        print("[MAVLink] LAND")

    def rtl(self):
        print("[MAVLink] RETURN TO LAUNCH")

    def send_velocity(self, vx, vy, vz):
        print(f"[MAVLink] VELOCITY vx={vx:.3f} vy={vy:.3f} vz={vz:.3f}")

    # --- methods used by mission.py ---

    def set_velocity(self, vx, vy, vz):
        print(f"[MAVLink] SET_VELOCITY vx={vx:+.3f} vy={vy:+.3f} vz={vz:+.3f}")

    def search_hover(self):
        print("[MAVLink] SEARCH HOVER — holding position")

    def land_with_correction(self, dx, dy):
        GAIN  = 0.003
        MAX_V = 0.3
        VZ    = 0.1
        vx = max(-MAX_V, min(MAX_V, dy * GAIN))
        vy = max(-MAX_V, min(MAX_V, dx * GAIN))
        print(f"[MAVLink] LANDING CORRECTION vx={vx:+.3f} vy={vy:+.3f} vz=+{VZ:.2f} (dx={dx} dy={dy})")