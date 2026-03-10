"""IK solver that maps OpenXR 26-joint hand tracking data to Inspire hand joint targets.

Computes a 7-element hand joint vector (6 active + 1 zero padding) from hand tracking
joint positions, mapping to Inspire DFQ motor layout:
  [0] thumb_yaw     [-0.1, 1.3]
  [1] thumb_pitch   [-0.1, 0.6]
  [2] index         [ 0.0, 1.7]
  [3] middle        [ 0.0, 1.7]
  [4] ring          [ 0.0, 1.7]
  [5] pinky         [ 0.0, 1.7]
  [6] padding       (always 0)

OpenXR hand joint layout (26 joints):
  0=Palm, 1=Wrist,
  2=ThumbMetacarpal, 3=ThumbProximal, 4=ThumbDistal, 5=ThumbTip,
  6=IndexMetacarpal, 7=IndexProximal, 8=IndexIntermediate, 9=IndexDistal, 10=IndexTip,
  11-15=Middle, 16-20=Ring, 21-25=Pinky  (same sub-layout as Index)
Each row is [x, y, z, qx, qy, qz, qw].
"""

import numpy as np

from gear_sonic.utils.teleop.solver.solver import Solver

# Inspire DFQ joint limits
THUMB_YAW_MIN, THUMB_YAW_MAX = -0.1, 1.3
THUMB_PITCH_MIN, THUMB_PITCH_MAX = -0.1, 0.6
FINGER_MIN, FINGER_MAX = 0.0, 1.7

# OpenXR 26-joint indices
PALM = 0
WRIST = 1
THUMB_MCP, THUMB_PROX, THUMB_DIST, THUMB_TIP = 2, 3, 4, 5
INDEX_MCP, INDEX_PROX, INDEX_INTER, INDEX_DIST, INDEX_TIP = 6, 7, 8, 9, 10
MIDDLE_MCP, MIDDLE_PROX, MIDDLE_INTER, MIDDLE_DIST, MIDDLE_TIP = 11, 12, 13, 14, 15
RING_MCP, RING_PROX, RING_INTER, RING_DIST, RING_TIP = 16, 17, 18, 19, 20
PINKY_MCP, PINKY_PROX, PINKY_INTER, PINKY_DIST, PINKY_TIP = 21, 22, 23, 24, 25

# Angle (degrees) at which finger curl saturates at FINGER_MAX.
# Typical fist curl produces ~100-140° between base and tip segment.
FINGER_CURL_MAX_ANGLE = 120.0
THUMB_CURL_MAX_ANGLE = 90.0

# Thumb-to-index distance thresholds (meters) for thumb yaw mapping.
# When thumb tip is far from index MCP → thumb is spread → yaw low.
# When thumb tip is close to index MCP → thumb is adducted → yaw high.
THUMB_YAW_DIST_FAR = 0.10   # thumb spread open
THUMB_YAW_DIST_CLOSE = 0.02  # thumb adducted for pinch


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two direction vectors (0 = same dir, 180 = opposite)."""
    u1 = _safe_normalize(v1)
    u2 = _safe_normalize(v2)
    dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def _finger_curl(pos: np.ndarray, mcp: int, prox: int, dist: int, tip: int) -> float:
    """
    Compute finger curl amount from the joint chain.

    When finger is STRAIGHT: base segment (mcp→prox) and tip segment (dist→tip)
    point the same direction → angle ≈ 0° → curl = 0.
    When finger is CURLED: the segments diverge → angle grows → curl increases.
    """
    seg_base = pos[prox] - pos[mcp]
    seg_tip = pos[tip] - pos[dist]
    angle = _angle_between_deg(seg_base, seg_tip)
    t = np.clip(angle / FINGER_CURL_MAX_ANGLE, 0.0, 1.0)
    return float(t * FINGER_MAX)


def _thumb_curl(pos: np.ndarray) -> float:
    """Thumb pitch/curl from the thumb joint chain."""
    seg_base = pos[THUMB_PROX] - pos[THUMB_MCP]
    seg_tip = pos[THUMB_TIP] - pos[THUMB_DIST]
    angle = _angle_between_deg(seg_base, seg_tip)
    t = np.clip(angle / THUMB_CURL_MAX_ANGLE, 0.0, 1.0)
    return float(THUMB_PITCH_MIN + t * (THUMB_PITCH_MAX - THUMB_PITCH_MIN))


def _thumb_yaw(pos: np.ndarray) -> float:
    """
    Thumb adduction via distance between thumb tip and index metacarpal.
    Close distance = adducted (high yaw), far distance = abducted (low yaw).
    """
    dist = float(np.linalg.norm(pos[THUMB_TIP] - pos[INDEX_MCP]))
    t = (THUMB_YAW_DIST_FAR - dist) / (THUMB_YAW_DIST_FAR - THUMB_YAW_DIST_CLOSE + 1e-8)
    t = np.clip(t, 0.0, 1.0)
    return float(THUMB_YAW_MIN + t * (THUMB_YAW_MAX - THUMB_YAW_MIN))


class G1InspireHandTrackingSolver(Solver):
    """Maps 26-joint OpenXR hand tracking state to 7-DOF Inspire hand commands."""

    def __init__(self, side: str) -> None:
        self.side = "L" if side.lower() == "left" else "R"
        self._call_count = 0

    def register_robot(self, robot):
        pass

    def __call__(self, hand_tracking_state: np.ndarray) -> np.ndarray:
        """
        Args:
            hand_tracking_state: shape (26, 7), each row [x, y, z, qx, qy, qz, qw].
                                 Can be None to return zeros.
        Returns:
            q_desired: shape (7,) [thumb_yaw, thumb_pitch, index, middle, ring, pinky, 0]
        """
        q = np.zeros(7, dtype=np.float32)

        if hand_tracking_state is None:
            return q
        hand_tracking_state = np.asarray(hand_tracking_state, dtype=np.float64)
        if hand_tracking_state.shape != (26, 7):
            if self._call_count == 0:
                print(f"[HandTrackSolver-{self.side}] unexpected shape {hand_tracking_state.shape}")
            self._call_count += 1
            return q

        pos = hand_tracking_state[:, :3]

        if self._call_count == 0:
            np.set_printoptions(precision=4, suppress=True)
            print(f"[HandTrackSolver-{self.side}] first call — wrist={pos[WRIST]}, "
                  f"index_tip={pos[INDEX_TIP]}, thumb_tip={pos[THUMB_TIP]}")

        try:
            q[0] = _thumb_yaw(pos)
            q[1] = _thumb_curl(pos)
            q[2] = _finger_curl(pos, INDEX_MCP, INDEX_PROX, INDEX_DIST, INDEX_TIP)
            q[3] = _finger_curl(pos, MIDDLE_MCP, MIDDLE_PROX, MIDDLE_DIST, MIDDLE_TIP)
            q[4] = _finger_curl(pos, RING_MCP, RING_PROX, RING_DIST, RING_TIP)
            q[5] = _finger_curl(pos, PINKY_MCP, PINKY_PROX, PINKY_DIST, PINKY_TIP)
        except Exception as e:
            if self._call_count % 200 == 0:
                print(f"[HandTrackSolver-{self.side}] error: {e}")

        self._call_count += 1
        return q
