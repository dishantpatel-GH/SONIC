"""IK solver that maps fingertip distances to Inspire hand joint targets.

Computes a 7-element hand joint vector (6 active + 1 zero padding) from
thumb-to-finger distances, mapping to Inspire DFQ motor layout:
  [0] thumb_yaw     [-0.1, 1.3]
  [1] thumb_pitch   [-0.1, 0.6]
  [2] index         [ 0.0, 1.7]
  [3] middle        [ 0.0, 1.7]
  [4] ring          [ 0.0, 1.7]
  [5] pinky         [ 0.0, 1.7]
  [6] padding       (always 0)

All zeros = fully open; positive values = closing.
Left and right hands use the same joint convention (no mirroring needed).
"""

import numpy as np

from decoupled_wbc.control.teleop.solver.solver import Solver

# Inspire DFQ close targets (comfortable grasp, within joint limits)
_THUMB_YAW_CLOSE = 0.6
_THUMB_PITCH_CLOSE = 0.4
_FINGER_CLOSE = 1.4


class G1InspireGripperIKSolver(Solver):
    def __init__(self, side) -> None:
        self.side = "L" if side.lower() == "left" else "R"

    def register_robot(self, robot):
        pass

    def __call__(self, finger_data):
        fingertips = finger_data["position"]

        positions = np.array([finger[:3, 3] for finger in fingertips])
        positions = np.reshape(positions, (-1, 3))

        thumb_pos = positions[4, :]
        index_pos = positions[4 + 5, :]
        middle_pos = positions[4 + 10, :]
        ring_pos = positions[4 + 15, :]
        pinky_pos = positions[4 + 20, :]

        index_dist = np.linalg.norm(thumb_pos - index_pos)
        middle_dist = np.linalg.norm(thumb_pos - middle_pos)
        ring_dist = np.linalg.norm(thumb_pos - ring_pos)
        pinky_dist = np.linalg.norm(thumb_pos - pinky_pos)

        dist_threshold = 0.05

        index_grip = np.clip(1.0 - index_dist, 0.0, 1.0)
        middle_grip = np.clip(1.0 - middle_dist, 0.0, 1.0)
        ring_grip = np.clip(1.0 - ring_dist, 0.0, 1.0)
        pinky_grip = np.clip(1.0 - pinky_dist, 0.0, 1.0)

        def apply_dead_zone(grip, threshold):
            return 0.0 if grip < threshold else grip

        index_grip = apply_dead_zone(index_grip, dist_threshold)
        middle_grip = apply_dead_zone(middle_grip, dist_threshold)
        ring_grip = apply_dead_zone(ring_grip, dist_threshold)
        pinky_grip = apply_dead_zone(pinky_grip, dist_threshold)

        grips = [index_grip, middle_grip, ring_grip, pinky_grip]
        max_grip = max(grips)

        q_desired = np.zeros(7)

        if max_grip == 0:
            return q_desired

        if middle_grip == max_grip:
            q_closed = self._get_full_close()
            q_desired = max_grip * q_closed
        elif index_grip == max_grip:
            q_closed = self._get_index_close()
            q_desired = index_grip * q_closed
        elif ring_grip == max_grip:
            q_closed = self._get_ring_close()
            q_desired = ring_grip * q_closed
        else:
            q_closed = self._get_pinky_close()
            q_desired = pinky_grip * q_closed

        return q_desired

    def _get_full_close(self):
        """All fingers close together (trigger + grip both pressed)."""
        q = np.zeros(7)
        q[0] = _THUMB_YAW_CLOSE
        q[1] = _THUMB_PITCH_CLOSE
        q[2] = _FINGER_CLOSE   # index
        q[3] = _FINGER_CLOSE   # middle
        q[4] = _FINGER_CLOSE   # ring
        q[5] = _FINGER_CLOSE   # pinky
        return q

    def _get_index_close(self):
        """Trigger only: thumb + index pinch, other fingers partially close."""
        q = np.zeros(7)
        q[0] = _THUMB_YAW_CLOSE
        q[1] = _THUMB_PITCH_CLOSE
        q[2] = _FINGER_CLOSE   # index
        q[3] = 0.8             # middle (partial)
        q[4] = 0.6             # ring (less)
        q[5] = 0.6             # pinky (less)
        return q

    def _get_ring_close(self):
        """Grip only: all four fingers curl, thumb follows."""
        q = np.zeros(7)
        q[0] = 0.3
        q[1] = 0.2
        q[2] = _FINGER_CLOSE
        q[3] = _FINGER_CLOSE
        q[4] = _FINGER_CLOSE
        q[5] = _FINGER_CLOSE
        return q

    def _get_pinky_close(self):
        """Pinky-led close (rare gesture)."""
        q = np.zeros(7)
        q[4] = _FINGER_CLOSE   # ring
        q[5] = _FINGER_CLOSE   # pinky
        return q
