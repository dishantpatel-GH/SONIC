"""Inspire FTP hand DDS reader for data collection.

Subscribes to FTP hand state and tactile DDS topics, converts state to
pipeline-format radians, and provides a polling API for the data exporter.

Requires ``unitree_sdk2py`` and ``inspire_sdkpy`` (same as Headless_driver).
"""

import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# DDS topic names (must match Headless_driver_double.py)
# ---------------------------------------------------------------------------
kTopicInspireStateLeft = "rt/inspire_hand/state/l"
kTopicInspireStateRight = "rt/inspire_hand/state/r"
kTopicInspireTouchLeft = "rt/inspire_hand/touch/l"
kTopicInspireTouchRight = "rt/inspire_hand/touch/r"
kTopicInspireCtrlLeft = "rt/inspire_hand/ctrl/l"
kTopicInspireCtrlRight = "rt/inspire_hand/ctrl/r"

# ---------------------------------------------------------------------------
# Pipeline joint limits (same as inspire_hands.hpp / inspire_ftp_controller.py)
#   Pipeline order: [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
# ---------------------------------------------------------------------------
PIPE_MAX = np.array([1.3, 0.6, 1.7, 1.7, 1.7, 1.7])
PIPE_MIN = np.array([-0.1, -0.1, 0.0, 0.0, 0.0, 0.0])
PIPE_RANGE = PIPE_MAX - PIPE_MIN

# Hardware → pipeline reorder
# Hardware: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
# Pipeline: [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
HW_TO_PIPELINE = [5, 4, 3, 2, 1, 0]

MOTORS_PER_HAND = 6

# Tactile sensor fields (per hand)
TOUCH_FIELDS = [
    ("fingerone_tip_touch", 9),
    ("fingerone_top_touch", 96),
    ("fingerone_palm_touch", 80),
    ("fingertwo_tip_touch", 9),
    ("fingertwo_top_touch", 96),
    ("fingertwo_palm_touch", 80),
    ("fingerthree_tip_touch", 9),
    ("fingerthree_top_touch", 96),
    ("fingerthree_palm_touch", 80),
    ("fingerfour_tip_touch", 9),
    ("fingerfour_top_touch", 96),
    ("fingerfour_palm_touch", 80),
    ("fingerfive_tip_touch", 9),
    ("fingerfive_top_touch", 96),
    ("fingerfive_middle_touch", 9),
    ("fingerfive_palm_touch", 96),
    ("palm_touch", 112),
]
TOUCH_FIELD_NAMES = [name for name, _ in TOUCH_FIELDS]
TACTILE_DIM = sum(size for _, size in TOUCH_FIELDS)  # 1062 per hand


def _hw_state_to_pipeline_radians(hw_normalized_6: np.ndarray) -> np.ndarray:
    """Convert 6-element normalised [0,1] hardware state to 7-element pipeline radians.

    Hardware state: angle_act / 1000 → [0,1] in hardware motor order.
    Pipeline: radians in [thumb_yaw, thumb_pitch, index, middle, ring, pinky, pad].
    """
    hw = np.asarray(hw_normalized_6, dtype=np.float64)
    pipeline_6 = hw[HW_TO_PIPELINE]
    radians = PIPE_MAX - pipeline_6 * PIPE_RANGE
    return np.append(radians, 0.0)  # 7-element with padding


def _touch_msg_to_flat_array(touch_msg) -> np.ndarray:
    """Flatten all tactile fields from a DDS touch message into a 1-D array."""
    parts = []
    for field_name, expected_size in TOUCH_FIELDS:
        vals = list(getattr(touch_msg, field_name, [0] * expected_size))
        if len(vals) < expected_size:
            vals.extend([0] * (expected_size - len(vals)))
        parts.extend(vals[:expected_size])
    return np.array(parts, dtype=np.float32)


class InspireFTPReader:
    """Read Inspire FTP hand state and tactile data from DDS for data collection."""

    def __init__(self):
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )

        # Import inspire_dds submodule without triggering inspire_sdkpy/__init__.py
        # (which eagerly imports pymodbus/Qt that may not be installed).
        import importlib
        import sys
        import types

        if "inspire_sdkpy" not in sys.modules:
            import importlib.util

            spec = importlib.util.find_spec("inspire_sdkpy")
            if spec is None or spec.submodule_search_locations is None:
                raise ImportError(
                    "inspire_sdkpy is required for FTP hand data collection. "
                    "Make sure the inspire_hand_sdk package is installed or on PYTHONPATH."
                )
            pkg = types.ModuleType("inspire_sdkpy")
            pkg.__path__ = list(spec.submodule_search_locations)
            pkg.__package__ = "inspire_sdkpy"
            sys.modules["inspire_sdkpy"] = pkg

        try:
            inspire_dds = importlib.import_module("inspire_sdkpy.inspire_dds")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "inspire_sdkpy is required for FTP hand data collection. "
                "Make sure the inspire_hand_sdk package is installed or on PYTHONPATH."
            ) from exc

        self._inspire_dds = inspire_dds

        try:
            ChannelFactoryInitialize(0, "")
        except Exception:
            pass

        # State subscribers
        self._sub_state_left = ChannelSubscriber(kTopicInspireStateLeft, inspire_dds.inspire_hand_state)
        self._sub_state_left.Init()
        self._sub_state_right = ChannelSubscriber(kTopicInspireStateRight, inspire_dds.inspire_hand_state)
        self._sub_state_right.Init()

        # Ctrl subscribers (to capture commanded action)
        self._sub_ctrl_left = ChannelSubscriber(kTopicInspireCtrlLeft, inspire_dds.inspire_hand_ctrl)
        self._sub_ctrl_left.Init()
        self._sub_ctrl_right = ChannelSubscriber(kTopicInspireCtrlRight, inspire_dds.inspire_hand_ctrl)
        self._sub_ctrl_right.Init()

        # Touch subscribers
        self._sub_touch_left = ChannelSubscriber(kTopicInspireTouchLeft, inspire_dds.inspire_hand_touch)
        self._sub_touch_left.Init()
        self._sub_touch_right = ChannelSubscriber(kTopicInspireTouchRight, inspire_dds.inspire_hand_touch)
        self._sub_touch_right.Init()

        # Latest data (protected by lock)
        self._lock = threading.Lock()
        self._left_state_hw = np.zeros(MOTORS_PER_HAND)  # normalised [0,1]
        self._right_state_hw = np.zeros(MOTORS_PER_HAND)
        self._left_action_hw = np.zeros(MOTORS_PER_HAND)
        self._right_action_hw = np.zeros(MOTORS_PER_HAND)
        self._left_tactile = np.zeros(TACTILE_DIM, dtype=np.float32)
        self._right_tactile = np.zeros(TACTILE_DIM, dtype=np.float32)

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

        # Wait for state
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._lock:
                if np.any(self._left_state_hw) or np.any(self._right_state_hw):
                    break
            time.sleep(0.01)

        with self._lock:
            got = np.any(self._left_state_hw) or np.any(self._right_state_hw)
        if got:
            print("[InspireFTPReader] DDS state confirmed")
        else:
            print("[InspireFTPReader] WARNING: No state — is Headless_driver_double.py running?")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_hand_state(self):
        """Return (left_q_7, right_q_7) in pipeline radians (7-element with padding)."""
        with self._lock:
            left = _hw_state_to_pipeline_radians(self._left_state_hw)
            right = _hw_state_to_pipeline_radians(self._right_state_hw)
        return left, right

    def get_hand_action(self):
        """Return (left_q_7, right_q_7) last commanded action in pipeline radians."""
        with self._lock:
            left = _hw_state_to_pipeline_radians(self._left_action_hw)
            right = _hw_state_to_pipeline_radians(self._right_action_hw)
        return left, right

    def get_tactile(self):
        """Return (left_tactile, right_tactile) as flat float32 arrays of dim TACTILE_DIM."""
        with self._lock:
            return self._left_tactile.copy(), self._right_tactile.copy()

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _poll_loop(self):
        while self._running:
            # State
            msg = self._sub_state_left.Read()
            if msg is not None and hasattr(msg, "angle_act") and len(msg.angle_act) == MOTORS_PER_HAND:
                with self._lock:
                    for i in range(MOTORS_PER_HAND):
                        self._left_state_hw[i] = msg.angle_act[i] / 1000.0

            msg = self._sub_state_right.Read()
            if msg is not None and hasattr(msg, "angle_act") and len(msg.angle_act) == MOTORS_PER_HAND:
                with self._lock:
                    for i in range(MOTORS_PER_HAND):
                        self._right_state_hw[i] = msg.angle_act[i] / 1000.0

            # Ctrl (action)
            msg = self._sub_ctrl_left.Read()
            if msg is not None and hasattr(msg, "angle_set") and len(msg.angle_set) == MOTORS_PER_HAND:
                with self._lock:
                    for i in range(MOTORS_PER_HAND):
                        self._left_action_hw[i] = msg.angle_set[i] / 1000.0

            msg = self._sub_ctrl_right.Read()
            if msg is not None and hasattr(msg, "angle_set") and len(msg.angle_set) == MOTORS_PER_HAND:
                with self._lock:
                    for i in range(MOTORS_PER_HAND):
                        self._right_action_hw[i] = msg.angle_set[i] / 1000.0

            # Tactile
            msg = self._sub_touch_left.Read()
            if msg is not None:
                with self._lock:
                    self._left_tactile = _touch_msg_to_flat_array(msg)

            msg = self._sub_touch_right.Read()
            if msg is not None:
                with self._lock:
                    self._right_tactile = _touch_msg_to_flat_array(msg)

            time.sleep(0.002)
