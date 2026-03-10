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

# Finger curl uses normalized tip-to-base distance ratio.
# OPEN_RATIO: ratio at/above which the finger is fully open (accounts for natural curvature).
# CLOSED_RATIO: ratio at/below which the finger is fully closed.
FINGER_OPEN_RATIO = 0.92
FINGER_CLOSED_RATIO = 0.30

# Thumb curl (pitch) distance-ratio thresholds.
# Pico data: open ≈ 0.99, curled ≈ 0.86. The usable range is narrow,
# so we map aggressively: full pitch reached at moderate curl (0.91).
THUMB_OPEN_RATIO = 0.97
THUMB_CLOSED_RATIO = 0.91

# Thumb yaw: driven by the max of two signals (palm proximity OR fingertip proximity).
# Palm-based: thumb-tip-to-palm, normalized by hand span.
# Pico data: spread ≈ 3.8-4.1, toward palm ≈ 2.3-2.6.
THUMB_YAW_PALM_OPEN = 3.8
THUMB_YAW_PALM_CLOSE = 2.3

# Pinch-based: minimum thumb-tip-to-fingertip distance, normalized by hand span.
# Spread ≈ 1.0-1.5, pinching ≈ 0.05-0.15.
THUMB_YAW_PINCH_FAR = 1.0
THUMB_YAW_PINCH_NEAR = 0.1


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _chain_length(pos: np.ndarray, *indices: int) -> float:
    """Sum of segment lengths along a chain of joint indices."""
    total = 0.0
    for i in range(len(indices) - 1):
        total += _dist(pos[indices[i]], pos[indices[i + 1]])
    return total


def _finger_curl(pos: np.ndarray, mcp: int, prox: int, inter: int, dist: int, tip: int) -> float:
    """
    Compute finger curl from the ratio of straight-line tip-to-MCP distance vs total chain length.

    When finger is STRAIGHT: tip-to-MCP ≈ total chain length → ratio ≈ 1.0 → curl = 0.
    When finger is CURLED into a fist: tip folds back → ratio drops → curl increases.
    """
    total_len = _chain_length(pos, mcp, prox, inter, dist, tip)
    if total_len < 1e-6:
        return 0.0
    direct = _dist(pos[mcp], pos[tip])
    ratio = direct / total_len

    t = np.clip((FINGER_OPEN_RATIO - ratio) / (FINGER_OPEN_RATIO - FINGER_CLOSED_RATIO), 0.0, 1.0)
    return float(FINGER_MIN + t * (FINGER_MAX - FINGER_MIN))


def _pinch_t(pos: np.ndarray, hand_span: float) -> float:
    """Normalized pinch signal: 0 = spread, 1 = thumb touching a fingertip."""
    if hand_span < 1e-6:
        return 0.0
    min_finger_dist = min(
        _dist(pos[THUMB_TIP], pos[INDEX_TIP]),
        _dist(pos[THUMB_TIP], pos[MIDDLE_TIP]),
        _dist(pos[THUMB_TIP], pos[RING_TIP]),
        _dist(pos[THUMB_TIP], pos[PINKY_TIP]),
    ) / hand_span
    return float(np.clip(
        (THUMB_YAW_PINCH_FAR - min_finger_dist) / (THUMB_YAW_PINCH_FAR - THUMB_YAW_PINCH_NEAR),
        0.0, 1.0,
    ))


def _thumb_curl(pos: np.ndarray) -> float:
    """
    Thumb pitch/curl from the max of two signals:
      1) Chain ratio: thumb joints physically curling
      2) Pinch proximity: thumb tip near a fingertip (pinch doesn't curl the chain much)
    """
    total_len = _chain_length(pos, THUMB_MCP, THUMB_PROX, THUMB_DIST, THUMB_TIP)
    if total_len < 1e-6:
        return float(THUMB_PITCH_MIN)
    direct = _dist(pos[THUMB_MCP], pos[THUMB_TIP])
    ratio = direct / total_len

    t_chain = np.clip((THUMB_OPEN_RATIO - ratio) / (THUMB_OPEN_RATIO - THUMB_CLOSED_RATIO), 0.0, 1.0)

    hand_span = _dist(pos[WRIST], pos[MIDDLE_MCP])
    t_pinch = _pinch_t(pos, hand_span)

    t = max(float(t_chain), t_pinch)
    return float(THUMB_PITCH_MIN + t * (THUMB_PITCH_MAX - THUMB_PITCH_MIN))


def _thumb_yaw(pos: np.ndarray) -> float:
    """
    Thumb opposition from the max of two signals:
      1) Palm proximity: thumb tip near palm center → curling into palm
      2) Pinch proximity: thumb tip near any fingertip → pinch gesture

    Either gesture drives yaw up; whichever is stronger wins.
    """
    hand_span = _dist(pos[WRIST], pos[MIDDLE_MCP])
    if hand_span < 1e-6:
        return float(THUMB_YAW_MIN)

    # Signal 1: palm proximity (curling toward palm)
    tip_to_palm = _dist(pos[THUMB_TIP], pos[PALM]) / hand_span
    t_palm = np.clip(
        (THUMB_YAW_PALM_OPEN - tip_to_palm) / (THUMB_YAW_PALM_OPEN - THUMB_YAW_PALM_CLOSE),
        0.0, 1.0,
    )

    # Signal 2: pinch proximity (thumb tip near closest fingertip)
    t_pinch = _pinch_t(pos, hand_span)

    t = max(t_palm, t_pinch)
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

        try:
            q[0] = _thumb_yaw(pos)
            q[1] = _thumb_curl(pos)
            q[2] = _finger_curl(pos, INDEX_MCP, INDEX_PROX, INDEX_INTER, INDEX_DIST, INDEX_TIP)
            q[3] = _finger_curl(pos, MIDDLE_MCP, MIDDLE_PROX, MIDDLE_INTER, MIDDLE_DIST, MIDDLE_TIP)
            q[4] = _finger_curl(pos, RING_MCP, RING_PROX, RING_INTER, RING_DIST, RING_TIP)
            q[5] = _finger_curl(pos, PINKY_MCP, PINKY_PROX, PINKY_INTER, PINKY_DIST, PINKY_TIP)
        except Exception as e:
            if self._call_count % 200 == 0:
                print(f"[HandTrackSolver-{self.side}] error: {e}")

        if self._call_count % 100 == 0:
            hand_span = _dist(pos[WRIST], pos[MIDDLE_MCP])
            hs = max(hand_span, 1e-6)
            thumb_chain = _chain_length(pos, THUMB_MCP, THUMB_PROX, THUMB_DIST, THUMB_TIP)
            thumb_ratio = _dist(pos[THUMB_MCP], pos[THUMB_TIP]) / max(thumb_chain, 1e-6)
            tip_palm_n = _dist(pos[THUMB_TIP], pos[PALM]) / hs
            pinch_n = min(
                _dist(pos[THUMB_TIP], pos[INDEX_TIP]),
                _dist(pos[THUMB_TIP], pos[MIDDLE_TIP]),
                _dist(pos[THUMB_TIP], pos[RING_TIP]),
                _dist(pos[THUMB_TIP], pos[PINKY_TIP]),
            ) / hs
            print(
                f"[HT-{self.side}] q={np.array2string(q, precision=3, suppress_small=True)} | "
                f"t_ratio={thumb_ratio:.3f} palm={tip_palm_n:.2f} pinch={pinch_n:.2f}"
            )

        self._call_count += 1
        return q
