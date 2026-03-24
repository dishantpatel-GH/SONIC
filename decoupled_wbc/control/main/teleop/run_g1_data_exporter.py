import select
import sys
import termios
import tty
from collections import deque
from datetime import datetime
import threading
import time

import numpy as np
import rclpy
import tyro

from decoupled_wbc.control.main.constants import ROBOT_CONFIG_TOPIC, STATE_TOPIC_NAME
from decoupled_wbc.control.main.teleop.configs.configs import DataExporterConfig
from decoupled_wbc.control.robot_model.instantiation import g1
from decoupled_wbc.control.sensor.composed_camera import ComposedCameraClientSensor
from decoupled_wbc.control.utils.episode_state import EpisodeState
from decoupled_wbc.control.utils.ros_utils import ROSMsgSubscriber


class StdinKeyboardReader:
    """Non-blocking stdin keyboard reader using raw terminal mode."""

    def __init__(self):
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def read_msg(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def close(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
from decoupled_wbc.control.utils.telemetry import Telemetry
from decoupled_wbc.control.utils.text_to_speech import TextToSpeech
from decoupled_wbc.data.constants import BUCKET_BASE_PATH
from decoupled_wbc.data.exporter import DataCollectionInfo, Gr00tDataExporter
from decoupled_wbc.data.utils import get_dataset_features, get_modality_config


class TimeDeltaException(Exception):
    def __init__(self, failure_count: int, reset_timeout_sec: float):
        """
        Exception raised when the time delta between two messages exceeds
        a threshold for a consecutive number of times
        """
        self.failure_count = failure_count
        self.reset_timeout_sec = reset_timeout_sec
        self.message = f"{self.failure_count} failures in {self.reset_timeout_sec} seconds"
        super().__init__(self.message)


class TimingThresholdMonitor:
    def __init__(self, max_failures=3, reset_timeout_sec=5, time_delta=0.2, raise_exception=False):
        """
        Monitor the time diff (between two messages) and optionally raise an exception
        if there is a consistent violations
        """
        self.max_failures = max_failures
        self.reset_timeout_sec = reset_timeout_sec
        self.failure_count = 0
        self.last_failure_time = 0
        self.time_delta = time_delta
        self.raise_exception = raise_exception

    def reset(self):
        self.failure_count = 0
        self.last_failure_time = 0

    def log_time_delta(self, time_delta_sec: float):
        time_delta = abs(time_delta_sec)
        if time_delta > self.time_delta:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()

        if self.is_threshold_exceeded():
            print(
                f"Time delta exception: {self.failure_count} failures in {self.reset_timeout_sec} seconds"
                f", time delta: {time_delta}"
            )
            if self.raise_exception:
                raise TimeDeltaException(self.failure_count, self.reset_timeout_sec)

    def is_threshold_exceeded(self):
        if self.failure_count >= self.max_failures:
            return True
        if time.monotonic() - self.last_failure_time > self.reset_timeout_sec:
            self.reset()
        return False


class Gr00tDataCollector:
    def __init__(
        self,
        node,
        camera_host: str,
        camera_port: int,
        state_topic_name: str,
        data_exporter: Gr00tDataExporter,
        text_to_speech=None,
        frequency=20,
        state_act_msg_frequency=50,
    ):

        self.text_to_speech = text_to_speech
        self.frequency = frequency
        self.data_exporter = data_exporter

        self.node = node

        thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        thread.start()
        time.sleep(0.5)

        self._episode_state = EpisodeState()
        self._keyboard_listener = StdinKeyboardReader()
        print("[INFO] Keyboard controls: 'c' = start/stop recording, 'x' = discard episode, Ctrl+C = quit")
        self._state_subscriber = ROSMsgSubscriber(state_topic_name, node=self.node)
        self._image_subscriber = ComposedCameraClientSensor(server_ip=camera_host, port=camera_port)
        self.rate = self.node.create_rate(self.frequency)

        self.obs_act_buffer = deque(maxlen=100)
        self.latest_image_msg = None
        self.latest_proprio_msg = None

        self.state_polling_rate = 1 / state_act_msg_frequency
        self.last_state_poll_time = time.monotonic()

        self.telemetry = Telemetry(window_size=100)
        self.timing_threshold_monitor = TimingThresholdMonitor()

        print(f"Recording to {self.data_exporter.meta.root}")

    @property
    def current_episode_index(self):
        return self.data_exporter.episode_buffer["episode_index"]

    def _print_and_say(self, message: str, say: bool = True):
        """Helper to use TextToSpeech print_and_say or fallback to print."""
        if self.text_to_speech is not None:
            self.text_to_speech.print_and_say(message, say)
        else:
            print(message)

    def _check_keyboard_input(self):
        key = self._keyboard_listener.read_msg()
        if key == "c":
            self._episode_state.change_state()
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self._print_and_say(f"Started recording {self.current_episode_index}")
            elif self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
                self._print_and_say("Stopping recording, preparing to save")
            elif self._episode_state.get_state() == self._episode_state.IDLE:
                self._print_and_say("Saved episode and back to idle state")
        elif key == "x":
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self.data_exporter.save_episode_as_discarded()
                self._episode_state.reset_state()
                self._print_and_say("Discarded episode")

    def _add_data_frame(self):
        t_start = time.monotonic()

        if self.latest_proprio_msg is None or self.latest_image_msg is None:
            self._print_and_say(
                f"Waiting for message. "
                f"Avail msg: proprio {self.latest_proprio_msg is not None} | "
                f"image {self.latest_image_msg is not None}",
                say=False,
            )
            return False

        # Debug: print proprio message keys once
        if not hasattr(self, '_debug_printed'):
            print(f"[DEBUG] proprio msg keys: {list(self.latest_proprio_msg.keys())}")
            self._debug_printed = True

        if self._episode_state.get_state() == self._episode_state.RECORDING:

            proprio = self.latest_proprio_msg

            # Compute image-proprio time delta
            # C++ deploy uses ros_timestamp; image uses camera_timestamps
            max_time_delta = 0
            proprio_time = proprio.get("ros_timestamp", time.time())
            for _, image_time in self.latest_image_msg["timestamps"].items():
                time_delta = abs(image_time - proprio_time)
                max_time_delta = max(max_time_delta, time_delta)

            self.timing_threshold_monitor.log_time_delta(max_time_delta)
            if (self.timing_threshold_monitor.failure_count + 1) % 100 == 0:
                self._print_and_say("Image state delta too high, please discard data")

            # Build full joint state matching URDF order.
            # C++ deploy sends 6 actuated DOF per hand (proximal joints only).
            # URDF has 12 per hand (proximal + coupled intermediate/distal).
            # Expand: duplicate each proximal value for its coupled joint.
            # URDF hand order: thumb_yaw, thumb_pitch, thumb_inter, thumb_distal,
            #   index_prox, index_inter, middle_prox, middle_inter,
            #   ring_prox, ring_inter, pinky_prox, pinky_inter
            # Deploy hand order: thumb_yaw, thumb_pitch, index, middle, ring, pinky
            def expand_hand_6_to_12(hand_6):
                """Expand 6 actuated hand DOFs to 12 URDF DOFs (proximal + coupled)."""
                h = np.array(hand_6, dtype=np.float64)
                if len(h) < 6:
                    h = np.zeros(6, dtype=np.float64)
                return np.array([
                    h[0], h[1], h[1], h[1],  # thumb: yaw, pitch, inter(=pitch), distal(=pitch)
                    h[2], h[2],               # index: prox, inter(=prox)
                    h[3], h[3],               # middle: prox, inter(=prox)
                    h[4], h[4],               # ring: prox, inter(=prox)
                    h[5], h[5],               # pinky: prox, inter(=prox)
                ], dtype=np.float64)

            body_q = np.array(proprio.get("body_q", []), dtype=np.float64)
            left_hand_q = expand_hand_6_to_12(proprio.get("left_hand_q", []))
            right_hand_q = expand_hand_6_to_12(proprio.get("right_hand_q", []))
            full_q = np.concatenate([body_q, left_hand_q, right_hand_q])

            last_action = np.array(proprio.get("last_action", []), dtype=np.float64)
            last_lh_action = expand_hand_6_to_12(proprio.get("last_left_hand_action", []))
            last_rh_action = expand_hand_6_to_12(proprio.get("last_right_hand_action", []))
            full_action = np.concatenate([last_action, last_lh_action, last_rh_action])

            # eef_state placeholder (C++ deploy doesn't provide wrist poses directly)
            eef_state = np.zeros(14, dtype=np.float64)

            frame_data = {
                "observation.state": full_q,
                "observation.eef_state": eef_state,
                "action": full_action,
                "action.eef": eef_state,
                "observation.img_state_delta": np.array([max_time_delta], dtype=np.float32),
                "teleop.navigate_command": np.zeros(3, dtype=np.float64),
                "teleop.base_height_command": np.array([0.74], dtype=np.float64),
            }

            # Add images based on dataset features (skip missing optional cameras)
            images = self.latest_image_msg["images"]
            for feature_name, feature_info in self.data_exporter.features.items():
                if feature_info.get("dtype") in ["image", "video"]:
                    image_key = feature_name.split(".")[-1]
                    if image_key in images:
                        frame_data[feature_name] = images[image_key]
                    else:
                        # Fill missing camera with black frame
                        h, w = feature_info["shape"][0], feature_info["shape"][1]
                        frame_data[feature_name] = np.zeros((h, w, 3), dtype=np.uint8)

            self.data_exporter.add_frame(frame_data)

        t_end = time.monotonic()
        if t_end - t_start > (1 / self.frequency):
            print(f"DataExporter Missed: {t_end - t_start} sec")

        if self._episode_state.get_state() == self._episode_state.NEED_TO_SAVE:
            self.data_exporter.save_episode()
            self.timing_threshold_monitor.reset()
            self._print_and_say("Finished saving episode")
            self._episode_state.change_state()

        return True

    def save_and_cleanup(self):
        try:
            self._print_and_say("saving episode done")
            # save on going episode if any
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode()
            self._print_and_say(f"Recording complete: {self.data_exporter.meta.root}", say=False)
        except Exception as e:
            self._print_and_say(f"Error saving episode: {e}")

        self._keyboard_listener.close()
        self.node.destroy_node()
        rclpy.shutdown()
        self._print_and_say("Shutting down data exporter...", say=False)

    def run(self):
        try:
            while rclpy.ok():
                t_start = time.monotonic()
                with self.telemetry.timer("total_loop"):
                    # 1. poll proprio msg
                    with self.telemetry.timer("poll_state"):
                        msg = self._state_subscriber.get_msg()
                        if msg is not None:
                            self.latest_proprio_msg = msg

                    # 2. poll image msg
                    with self.telemetry.timer("poll_image"):
                        msg = self._image_subscriber.read()
                        if msg is not None:
                            self.latest_image_msg = msg

                    # 3. check keyboard input
                    with self.telemetry.timer("check_keyboard"):
                        self._check_keyboard_input()

                    # 4. add frame
                    with self.telemetry.timer("add_frame"):
                        self._add_data_frame()

                    end_time = time.monotonic()

                self.rate.sleep()

                # Log timing information if we missed our target frequency
                if (end_time - t_start) > (1 / self.frequency):
                    self.telemetry.log_timing_info(
                        context="Data Exporter Loop Missed", threshold=0.001
                    )

        except KeyboardInterrupt:
            print("Data exporter terminated by user")
            # The user will trigger a keyboard interrupt if there's something wrong,
            # so we flag the ongoing episode as discarded
            buffer_size = self.data_exporter.episode_buffer.get("size", 0)
            if buffer_size > 0:
                self.data_exporter.save_episode_as_discarded()

        finally:
            self.save_and_cleanup()


def main(config: DataExporterConfig):

    rclpy.init(args=None)
    node = rclpy.create_node("data_exporter")

    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    g1_rm = g1.instantiate_g1_robot_model(
        waist_location=waist_location, high_elbow_pose=config.high_elbow_pose
    )

    dataset_features = get_dataset_features(g1_rm, config.add_stereo_camera)
    modality_config = get_modality_config(g1_rm, config.add_stereo_camera)

    text_to_speech = TextToSpeech() if config.text_to_speech else None

    # Only set DataCollectionInfo if we're creating a new dataset
    # When adding to existing dataset, DataCollectionInfo will be ignored
    if config.robot_id is not None:
        data_collection_info = DataCollectionInfo(
            teleoperator_username=config.teleoperator_username,
            support_operator_username=config.support_operator_username,
            robot_type="g1",
            robot_id=config.robot_id,
            lower_body_policy=config.lower_body_policy,
            wbc_model_path=config.wbc_model_path,
        )
    else:
        # Use default DataCollectionInfo when adding to existing dataset
        # This will be ignored if the dataset already exists
        data_collection_info = DataCollectionInfo()

    # The C++ deploy publishes robot_config as a topic, not a service.
    # Skip the service call and use an empty config dict for metadata.
    robot_config = {}
    print("[INFO] Skipping robot_config service (deploy publishes as topic). Using empty config.")

    data_exporter = Gr00tDataExporter.create(
        save_root=f"{config.root_output_dir}/{config.dataset_name}",
        fps=config.data_collection_frequency,
        features=dataset_features,
        modality_config=modality_config,
        task=config.task_prompt,
        upload_bucket_path=BUCKET_BASE_PATH,
        data_collection_info=data_collection_info,
        script_config=robot_config,
    )

    data_collector = Gr00tDataCollector(
        node=node,
        frequency=config.data_collection_frequency,
        data_exporter=data_exporter,
        state_topic_name=STATE_TOPIC_NAME,
        camera_host=config.camera_host,
        camera_port=config.camera_port,
        text_to_speech=text_to_speech,
    )
    data_collector.run()


if __name__ == "__main__":
    config = tyro.cli(DataExporterConfig)
    config.task_prompt = input("Enter the task prompt: ").strip().lower()
    add_to_existing_dataset = input("Add to existing dataset? (y/n): ").strip().lower()

    if add_to_existing_dataset == "y":
        config.dataset_name = input("Enter the dataset name: ").strip().lower()
        # When adding to existing dataset, we don't need robot_id or operator usernames
        # as they should already be set in the existing dataset
    elif add_to_existing_dataset == "n":
        # robot_id = input("Enter the robot ID: ").strip().lower()
        # if robot_id not in G1_ROBOT_IDS:
        #     raise ValueError(f"Invalid robot ID: {robot_id}. Available robot IDs: {G1_ROBOT_IDS}")
        config.robot_id = "sim"
        config.dataset_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-G1-{config.robot_id}"

        # Only ask for operator usernames when creating a new dataset
        # print("Available teleoperator usernames:")
        # for i, username in enumerate(OPERATOR_USERNAMES):
        #     print(f"{i}: {username}")
        # teleop_idx = int(input("Select teleoperator username index: "))
        # config.teleoperator_username = OPERATOR_USERNAMES[teleop_idx]
        config.teleoperator_username = "NEW_USER"

        # print("\nAvailable support operator usernames:")
        # for i, username in enumerate(OPERATOR_USERNAMES):
        #     print(f"{i}: {username}")
        # support_idx = int(input("Select support operator username index: "))
        # config.support_operator_username = OPERATOR_USERNAMES[support_idx]
        config.support_operator_username = "NEW_USER"

    main(config)
