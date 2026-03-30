"""Inspire FTP hand controller for SONIC teleop pipeline.

Receives 7-element pipeline-format hand joint arrays from the existing SONIC
solvers (DexPilot / geometric / IK), converts them to the FTP DDS protocol
([0-1000] integer scale), and publishes to the Inspire FTP hand DDS topics.

Requires:
  - ``unitree_sdk2py`` (DDS transport)
  - ``inspire_sdkpy``  (FTP message definitions)
  - ``Headless_driver_double.py`` running on the robot (ModbusTCP <-> DDS bridge)
"""

import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# DDS topic names (must match Headless_driver_double.py)
# ---------------------------------------------------------------------------
kTopicInspireCtrlLeft = "rt/inspire_hand/ctrl/l"
kTopicInspireCtrlRight = "rt/inspire_hand/ctrl/r"
kTopicInspireStateLeft = "rt/inspire_hand/state/l"
kTopicInspireStateRight = "rt/inspire_hand/state/r"

# ---------------------------------------------------------------------------
# Pipeline joint limits (same as C++ inspire_hands.hpp)
#   Pipeline order: [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
# ---------------------------------------------------------------------------
PIPE_MAX = np.array([1.3, 0.6, 1.7, 1.7, 1.7, 1.7])
PIPE_MIN = np.array([-0.1, -0.1, 0.0, 0.0, 0.0, 0.0])

# Pipeline index -> FTP hardware motor index (reversed order)
#   Pipeline: [0:thumb_yaw, 1:thumb_pitch, 2:index, 3:middle, 4:ring, 5:pinky]
#   Hardware: [0:pinky,     1:ring,        2:middle, 3:index,  4:thumb_pitch, 5:thumb_yaw]
PIPELINE_TO_HW = [5, 4, 3, 2, 1, 0]

MOTORS_PER_HAND = 6


def _pipeline_to_ftp_cmd(joints_7: np.ndarray) -> list:
    """Convert a 7-element pipeline joint array to a 6-element FTP command list.

    Steps:
      1. Take first 6 elements (ignore padding at index 6).
      2. Normalize each to [0, 1]  (0 = closed, 1 = open).
      3. Reorder from pipeline to FTP hardware order.
      4. Scale to integer [0, 1000].

    Returns:
        List of 6 ints in [0, 1000] in FTP hardware motor order.
    """
    q = np.asarray(joints_7[:6], dtype=np.float64)
    ranges = PIPE_MAX - PIPE_MIN
    normalized = np.clip((PIPE_MAX - q) / ranges, 0.0, 1.0)
    # Reorder to hardware
    hw = normalized[PIPELINE_TO_HW]
    return [int(np.clip(v * 1000, 0, 1000)) for v in hw]


class InspireFTPController:
    """Send hand commands to Inspire FTP hands via DDS.

    Usage::

        ctrl = InspireFTPController()
        # In the teleop loop, after computing hand joints with existing solvers:
        ctrl.send(left_hand_joints_7, right_hand_joints_7)

    The controller also subscribes to hand state feedback from the FTP hands
    so that state can be read back (e.g. for logging / data collection).
    """

    def __init__(self):
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )

        # Import only the submodules we need (inspire_dds + inspire_hand_defaut).
        # inspire_sdkpy/__init__.py eagerly imports pymodbus and Qt which may not
        # be installed, so we register a minimal stub package first to let the
        # submodule relative imports work without triggering __init__.py.
        import importlib
        import sys
        import types

        if "inspire_sdkpy" not in sys.modules:
            # Find the package directory on sys.path
            import importlib.util

            spec = importlib.util.find_spec("inspire_sdkpy")
            if spec is None or spec.submodule_search_locations is None:
                raise ImportError(
                    "inspire_sdkpy is required for FTP hand control. "
                    "Make sure the inspire_hand_sdk package is installed or on PYTHONPATH."
                )
            pkg = types.ModuleType("inspire_sdkpy")
            pkg.__path__ = list(spec.submodule_search_locations)
            pkg.__package__ = "inspire_sdkpy"
            sys.modules["inspire_sdkpy"] = pkg

        try:
            inspire_dds = importlib.import_module("inspire_sdkpy.inspire_dds")
            inspire_hand_defaut = importlib.import_module("inspire_sdkpy.inspire_hand_defaut")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "inspire_sdkpy is required for FTP hand control. "
                "Make sure the inspire_hand_sdk package is installed or on PYTHONPATH."
            ) from exc

        self._inspire_dds = inspire_dds
        self._inspire_hand_defaut = inspire_hand_defaut

        # Initialise DDS domain (no interface arg = auto-detect, same as xr_teleoperate).
        # In SONIC the C++ deploy runs DDS in a separate process, so this Python
        # process needs its own DDS domain for the Inspire FTP topics.
        try:
            ChannelFactoryInitialize(0)
            print("[InspireFTP] DDS domain initialised")
        except Exception as e:
            print(f"[InspireFTP] DDS init note: {e} (may already be initialised)")

        # Publishers
        self._pub_left = ChannelPublisher(kTopicInspireCtrlLeft, inspire_dds.inspire_hand_ctrl)
        self._pub_left.Init()
        self._pub_right = ChannelPublisher(kTopicInspireCtrlRight, inspire_dds.inspire_hand_ctrl)
        self._pub_right.Init()

        # Subscribers (state feedback)
        self._sub_left = ChannelSubscriber(kTopicInspireStateLeft, inspire_dds.inspire_hand_state)
        self._sub_left.Init()
        self._sub_right = ChannelSubscriber(kTopicInspireStateRight, inspire_dds.inspire_hand_state)
        self._sub_right.Init()

        # Latest state (normalised to [0, 1], 6 per hand)
        self._left_state = np.zeros(MOTORS_PER_HAND)
        self._right_state = np.zeros(MOTORS_PER_HAND)
        self._state_lock = threading.Lock()

        # Start state subscription thread
        self._running = True
        self._state_thread = threading.Thread(target=self._subscribe_state, daemon=True)
        self._state_thread.start()

        # Wait briefly for state confirmation
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._state_lock:
                if np.any(self._left_state) or np.any(self._right_state):
                    break
            time.sleep(0.01)

        with self._state_lock:
            got_state = np.any(self._left_state) or np.any(self._right_state)
        if got_state:
            print("[InspireFTP] DDS state confirmed — hands connected")
        else:
            print("[InspireFTP] WARNING: No state received — is Headless_driver_double.py running?")

        self._log_counter = 0
        print("[InspireFTP] Controller initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, left_joints_7: np.ndarray, right_joints_7: np.ndarray):
        """Convert pipeline joint arrays and publish FTP commands."""
        if self._log_counter == 0:
            print(f"[InspireFTP] First send() call — L_pipeline={left_joints_7[:6]}  R_pipeline={right_joints_7[:6]}")
        left_cmd = _pipeline_to_ftp_cmd(left_joints_7)
        right_cmd = _pipeline_to_ftp_cmd(right_joints_7)

        msg_l = self._inspire_hand_defaut.get_inspire_hand_ctrl()
        msg_l.angle_set = left_cmd
        msg_l.mode = 0b0001
        self._pub_left.Write(msg_l)

        msg_r = self._inspire_hand_defaut.get_inspire_hand_ctrl()
        msg_r.angle_set = right_cmd
        msg_r.mode = 0b0001
        self._pub_right.Write(msg_r)

        self._log_counter += 1
        if self._log_counter % 500 == 1:
            print(f"[InspireFTP] Cmd L={left_cmd} R={right_cmd}")

    def get_state(self):
        """Return latest hand state as (left_6, right_6) normalised [0,1]."""
        with self._state_lock:
            return self._left_state.copy(), self._right_state.copy()

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _subscribe_state(self):
        while self._running:
            left_msg = self._sub_left.Read()
            if left_msg is not None and hasattr(left_msg, "angle_act") and len(left_msg.angle_act) == MOTORS_PER_HAND:
                with self._state_lock:
                    for i in range(MOTORS_PER_HAND):
                        self._left_state[i] = left_msg.angle_act[i] / 1000.0

            right_msg = self._sub_right.Read()
            if right_msg is not None and hasattr(right_msg, "angle_act") and len(right_msg.angle_act) == MOTORS_PER_HAND:
                with self._state_lock:
                    for i in range(MOTORS_PER_HAND):
                        self._right_state[i] = right_msg.angle_act[i] / 1000.0

            time.sleep(0.002)
