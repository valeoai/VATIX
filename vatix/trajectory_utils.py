"""Trajectory helpers: smoothing and the canonical command trajectories.

Waypoints are cumulative meters in the camera frame (x = right, y = forward), origin-relative.
"""

import numpy as np


def smooth_trajectory(traj):
    """SavGol-smooth (window 5, order 2) a (T, 2) or (N, T, 2) trajectory; returns float32."""
    from scipy.signal import savgol_filter

    return savgol_filter(traj, window_length=5, polyorder=2, deriv=0,
                         axis=-2).astype(np.float32)


def command_trajectories(trajectory_length=25, dt=1.0 / 9.0, v_fwd=5.0, yaw_deg=15.0):
    """Canonical commands `left, right, straight, static`, each (T, 2) float32.

    Positions are recorded before integrating, so traj[0] == (0, 0). Left accumulates negative x.
    """
    num_steps = int(trajectory_length)
    out = {}

    for name, omega_deg in (("left", -yaw_deg), ("right", yaw_deg), ("straight", 0.0)):
        omega = np.radians(omega_deg)
        traj = np.zeros((num_steps, 2), dtype=np.float32)
        x, y = 0.0, 0.0
        for t in range(num_steps):
            traj[t] = [x, y]
            theta = omega * t * dt
            x += v_fwd * np.sin(theta) * dt
            y += v_fwd * np.cos(theta) * dt
        out[name] = traj

    # Static: an explicit "stay put" command, distinct from the unconditional null (trajectory=None).
    out["static"] = np.zeros((num_steps, 2), dtype=np.float32)

    return out
