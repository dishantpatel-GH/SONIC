"""DexPilot-based hand retargeting solver for Inspire DFQ hands with PICO OpenXR tracking.

Uses the DexPilot optimization framework (from dex_retargeting) to retarget PICO 26-joint
OpenXR hand tracking data to Inspire DFQ 6-DOF hand joint commands. This provides more
accurate finger mapping than the simple geometric heuristic solver by:

  - Using forward kinematics with the actual Inspire hand URDF model
  - Non-linear optimization (NLopt SLSQP) to match fingertip positions
  - Contact-aware weighting for stable grasps (DexPilot paper)
  - Low-pass filtering for smooth motion
  - Mimic joint handling for coupled intermediate joints

Output format (same as G1InspireHandTrackingSolver for drop-in compatibility):
  [0] thumb_yaw     [-0.1, 1.3]
  [1] thumb_pitch   [-0.1, 0.6]
  [2] index         [ 0.0, 1.7]
  [3] middle        [ 0.0, 1.7]
  [4] ring          [ 0.0, 1.7]
  [5] pinky         [ 0.0, 1.7]
  [6] padding       (always 0)

PICO OpenXR 26-joint layout:
  0=Palm, 1=Wrist,
  2=ThumbMetacarpal, 3=ThumbProximal, 4=ThumbDistal, 5=ThumbTip,
  6=IndexMetacarpal, 7=IndexProximal, 8=IndexIntermediate, 9=IndexDistal, 10=IndexTip,
  11=MiddleMetacarpal, 12=MiddleProximal, 13=MiddleIntermediate, 14=MiddleDistal, 15=MiddleTip,
  16=RingMetacarpal, 17=RingProximal, 18=RingIntermediate, 19=RingDistal, 20=RingTip,
  21=PinkyMetacarpal, 22=PinkyProximal, 23=PinkyIntermediate, 24=PinkyDistal, 25=PinkyTip
"""

from pathlib import Path

import numpy as np
import yaml

from dex_retargeting import RetargetingConfig

from gear_sonic.utils.teleop.solver.solver import Solver

# Rotation from hand-local frame [fwd, lat, norm] to Unitree hand URDF frame.
# Hand-local axes (computed from joint positions):
#   fwd  = wrist → middle proximal (along fingers)
#   lat  = index proximal → ring proximal (across palm, radial → ulnar)
#   norm = cross(fwd, lat) = palm → back of hand (dorsal)
# Unitree URDF axes:
#   x = dorsal (palm → back)  = norm
#   y = proximal (finger → wrist) = -fwd
#   z = cross(x,y) = -lat
_R_HAND_TO_UNITREE = np.array(
    [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64
)

# PICO joint indices for building hand frame
_WRIST_IDX = 1
_INDEX_PROXIMAL_IDX = 7
_MIDDLE_PROXIMAL_IDX = 12
_RING_PROXIMAL_IDX = 17

# Inspire DFQ joint output order for our pipeline
_OUTPUT_JOINT_NAMES_LEFT = [
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_ring_proximal_joint",
    "L_pinky_proximal_joint",
]

_OUTPUT_JOINT_NAMES_RIGHT = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
]

# Default config path (relative to gear_sonic package root)
_DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "assets" / "inspire_hand" / "inspire_hand_pico.yml"


def _build_hand_frame_rotation(pos: np.ndarray, is_left: bool) -> np.ndarray:
    """Build rotation matrix from world frame to Unitree hand URDF frame using joint positions.

    Computes a hand-local coordinate frame from wrist, index, middle, and ring proximal
    joint positions, then maps it to the Unitree hand URDF convention. This is more robust
    than using the wrist quaternion since it's independent of PICO's quaternion convention.

    The lateral direction is flipped for left vs right hand so that cross(fwd, lat) always
    produces the dorsal (palm→back) direction, matching the Unitree URDF x-axis convention.

    Args:
        pos: (26, 3) joint positions in OpenXR world frame.
        is_left: True for left hand, False for right hand.

    Returns:
        R: (3, 3) rotation matrix that transforms world-frame vectors to Unitree URDF frame.
    """
    wrist = pos[_WRIST_IDX]
    middle_prox = pos[_MIDDLE_PROXIMAL_IDX]
    index_prox = pos[_INDEX_PROXIMAL_IDX]
    ring_prox = pos[_RING_PROXIMAL_IDX]

    # Forward: wrist → middle proximal (along finger direction)
    v_fwd = middle_prox - wrist
    v_fwd = v_fwd / (np.linalg.norm(v_fwd) + 1e-8)

    # Lateral direction depends on handedness so that cross(fwd, lat) = dorsal.
    # Left hand: ring - index (ulnar) → cross(fwd, ulnar) = dorsal ✓
    # Right hand: index - ring (radial) → cross(fwd, radial) = dorsal ✓
    if is_left:
        v_lat_raw = ring_prox - index_prox
    else:
        v_lat_raw = index_prox - ring_prox
    # Orthogonalize against forward direction
    v_lat = v_lat_raw - np.dot(v_lat_raw, v_fwd) * v_fwd
    v_lat = v_lat / (np.linalg.norm(v_lat) + 1e-8)

    # Normal: palm → back (dorsal direction, via cross product)
    v_norm = np.cross(v_fwd, v_lat)

    # R_world_to_hand: rows are hand-local axes expressed in world frame
    # Transforms world vectors into [fwd, lat, norm] space
    R_world_to_hand = np.vstack([v_fwd, v_lat, v_norm])  # (3, 3)

    # Combined: world → hand-local → Unitree URDF
    return _R_HAND_TO_UNITREE @ R_world_to_hand


class G1InspireDexPilotSolver(Solver):
    """Maps 26-joint PICO OpenXR hand tracking to 7-DOF Inspire hand commands via DexPilot."""

    def __init__(self, side: str, config_path: str = None) -> None:
        self.side = "L" if side.lower() == "left" else "R"
        self.is_left = self.side == "L"
        self._call_count = 0

        config_path = Path(config_path) if config_path else _DEFAULT_CONFIG
        assets_dir = config_path.parent.parent  # .../assets/

        # Load YAML config
        with config_path.open("r") as f:
            cfg = yaml.safe_load(f)

        side_key = "left" if self.is_left else "right"
        side_cfg = cfg[side_key]

        # Set URDF base directory and build retargeting
        RetargetingConfig.set_default_urdf_dir(str(assets_dir))
        retargeting_config = RetargetingConfig.from_dict(side_cfg)
        self.retargeting = retargeting_config.build()

        # Cache the human landmark indices for computing reference vectors
        self.indices = self.retargeting.optimizer.target_link_human_indices  # (2, 15)

        # Build index mapping: retargeting robot_qpos → our 6-joint output order
        output_names = _OUTPUT_JOINT_NAMES_LEFT if self.is_left else _OUTPUT_JOINT_NAMES_RIGHT
        retargeting_joint_names = self.retargeting.joint_names  # pinocchio DOF order
        self._output_indices = [retargeting_joint_names.index(n) for n in output_names]

        print(f"[DexPilot-{self.side}] Initialized with config: {config_path.name}")
        print(f"[DexPilot-{self.side}] Retargeting DOF: {len(retargeting_joint_names)}, "
              f"output joints: {output_names}")

    def register_robot(self, robot):
        pass

    def __call__(self, hand_tracking_state: np.ndarray) -> np.ndarray:
        """
        Args:
            hand_tracking_state: shape (26, 7) from PICO OpenXR, each row [x, y, z, qx, qy, qz, qw].
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
                print(f"[DexPilot-{self.side}] unexpected shape {hand_tracking_state.shape}")
            self._call_count += 1
            return q

        # Extract 3D positions (ignore quaternion orientation columns)
        pos = hand_tracking_state[:, :3]  # (26, 3) in OpenXR world frame

        try:
            # Build rotation from world frame to Unitree hand URDF frame using joint positions.
            # This computes a hand-local frame from wrist/knuckle positions (robust, no quaternion
            # convention dependency) then maps to the Unitree URDF convention.
            R = _build_hand_frame_rotation(pos, self.is_left)

            # Compute reference vectors in world frame, then rotate to Unitree hand URDF frame
            ref_value_world = pos[self.indices[1, :]] - pos[self.indices[0, :]]  # (15, 3)
            ref_value = (R @ ref_value_world.T).T  # (15, 3) in Unitree hand URDF frame

            # Run DexPilot retargeting optimization
            robot_qpos = self.retargeting.retarget(ref_value)  # full robot DOF

            # Extract our 6 output joints in the correct order
            for i, idx in enumerate(self._output_indices):
                q[i] = robot_qpos[idx]

        except Exception as e:
            if self._call_count % 200 == 0:
                print(f"[DexPilot-{self.side}] retargeting error: {e}")

        if self._call_count % 100 == 0:
            # Log joint values and reference vector magnitudes for diagnostics
            ref_norms = np.linalg.norm(ref_value_world, axis=1) if 'ref_value_world' in dir() else None
            extra = ""
            if ref_norms is not None:
                extra = f"  ref_mag=[{ref_norms.min():.4f}, {ref_norms.max():.4f}]"
            print(
                f"[DexPilot-{self.side}] q={np.array2string(q, precision=3, suppress_small=True)}"
                f"{extra}"
            )

        self._call_count += 1
        return q
