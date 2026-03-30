from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal, Optional

import yaml

import decoupled_wbc
from decoupled_wbc.control.main.config_template import ArgsConfig as ArgsConfigTemplate
from decoupled_wbc.control.policy.wbc_policy_factory import WBC_VERSIONS
from decoupled_wbc.control.utils.network_utils import resolve_interface


def override_wbc_config(
    wbc_config: dict, config: "BaseConfig", missed_keys_only: bool = False
) -> dict:
    """Override WBC YAML values with dataclass values.

    Args:
        wbc_config: The loaded WBC YAML configuration dictionary
        config: The BaseConfig dataclass instance with override values
        missed_keys_only: If True, only add keys that don't exist in wbc_config.
                          If False, validate all keys exist and override all.

    Returns:
        Updated wbc_config dictionary with overridden values

    Raises:
        KeyError: If any required keys are missing from the WBC YAML configuration
                  (only when missed_keys_only=False)
    """
    # Override yaml values with dataclass values
    key_to_value = {
        "INTERFACE": config.interface,
        "ENV_TYPE": config.env_type,
        "VERSION": config.wbc_version,
        "SIMULATOR": config.simulator,
        "SIMULATE_DT": 1 / float(config.sim_frequency),
        "ENABLE_OFFSCREEN": config.enable_offscreen,
        "ENABLE_ONSCREEN": config.enable_onscreen,
        "model_path": config.wbc_model_path,
        "enable_waist": config.enable_waist,
        "with_hands": config.with_hands,
        # --- Add HAND_TYPE here so G1Env can read it ---
        "HAND_TYPE": config.hand_type, 
        # -----------------------------------------------
        "verbose": config.verbose,
        "verbose_timing": config.verbose_timing,
        "upper_body_max_joint_speed": config.upper_body_joint_speed,
        "keyboard_dispatcher_type": config.keyboard_dispatcher_type,
        "enable_gravity_compensation": config.enable_gravity_compensation,
        "gravity_compensation_joints": config.gravity_compensation_joints,
        "high_elbow_pose": config.high_elbow_pose,
    }

    if missed_keys_only:
        # Only add keys that don't exist in wbc_config
        for key in key_to_value:
            if key not in wbc_config:
                wbc_config[key] = key_to_value[key]
    else:
        # Set all keys (overwrite existing)
        for key in key_to_value:
            wbc_config[key] = key_to_value[key]

    # g1 kp, kd, sim2real gap
    if config.env_type == "real":
        # update waist pitch damping, index 14
        wbc_config["MOTOR_KD"][14] = wbc_config["MOTOR_KD"][14] - 10

    return wbc_config


@dataclass
class BaseConfig(ArgsConfigTemplate):
    """Base config inherited by all G1 control loops"""

    # WBC Configuration
    wbc_version: Literal[tuple(WBC_VERSIONS)] = "gear_wbc"
    """Version of the whole body controller."""

    wbc_model_path: str = (
        "policy/GR00T-WholeBodyControl-Balance.onnx," "policy/GR00T-WholeBodyControl-Walk.onnx"
    )
    """Path to WBC model file (relative to gr00t_wbc/sim2mujoco/resources/robots/g1)"""
    """gear_wbc model path: policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx"""

    wbc_policy_class: str = "G1DecoupledWholeBodyPolicy"
    """Whole body policy class."""

    # System Configuration
    interface: str = "sim"
    """Interface to use for the control loop. [sim, real, lo, enxe8ea6a9c4e09]"""

    simulator: str = "mujoco"
    """Simulator to use."""

    sim_sync_mode: bool = False
    """Whether to run the control loop in sync mode."""

    control_frequency: int = 50
    """Frequency of the control loop."""

    sim_frequency: int = 200
    """Frequency of the simulation loop."""

    # Robot Configuration
    enable_waist: bool = False
    """Whether to include waist joints in IK."""

    with_hands: bool = True
    """Enable hand functionality. When False, robot operates without hands."""

    # --- Add hand_type field ---
    hand_type: Literal["dex3", "inspire"] = "dex3"
    """Type of hand to use. Options: 'dex3' (default), 'inspire'."""
    # ---------------------------

    high_elbow_pose: bool = False
    """Enable high elbow pose configuration for default joint positions."""

    verbose: bool = True
    """Whether to print verbose output."""

    # Additional common fields
    enable_offscreen: bool = False
    """Whether to enable offscreen rendering."""

    enable_onscreen: bool = True
    """Whether to enable onscreen rendering."""

    upper_body_joint_speed: float = 1000
    """Upper body joint speed."""

    env_name: str = "default"
    """Environment name."""

    ik_indicator: bool = False
    """Whether to draw IK indicators."""

    verbose_timing: bool = False
    """Enable verbose timing output every iteration."""

    keyboard_dispatcher_type: str = "raw"
    """Keyboard dispatcher to use. [raw, ros]"""

    # Gravity Compensation Configuration
    enable_gravity_compensation: bool = False
    """Enable gravity compensation using pinocchio dynamics."""

    gravity_compensation_joints: Optional[list[str]] = None
    """Joint groups to apply gravity compensation to (e.g., ['arms', 'left_arm', 'right_arm'])."""
    
    # Teleop/Device Configuration
    body_control_device: str = "dummy"
    """Device to use for body control. Options: dummy, vive, iphone, leapmotion, joycon."""

    hand_control_device: Optional[str] = "dummy"
    """Device to use for hand control. Options: None, manus, joycon, iphone."""

    body_streamer_ip: str = "10.112.210.229"
    """IP address for body streamer (vive only)."""

    body_streamer_keyword: str = "knee"
    """Body streamer keyword (vive only)."""

    enable_visualization: bool = False
    """Whether to enable visualization."""

    enable_real_device: bool = True
    """Whether to enable real device."""

    teleop_frequency: int = 20
    """Teleoperation frequency (Hz)."""

    teleop_replay_path: Optional[str] = None
    """Path to teleop replay data."""

    # Hand Tracking Server Configuration (for pico_hand_tracking device)
    hand_tracking_server_host: str = "localhost"
    """Host where the Pico hand tracking server is running (for pico_hand_tracking device)"""
    hand_tracking_server_port: int = 5557
    """Port where the Pico hand tracking server is listening (for pico_hand_tracking device)"""

    # Deployment/Camera Configuration
    robot_ip: str = "192.168.123.164"
    """Robot IP address"""
    
    # Data collection settings
    data_collection: bool = True
    """Enable data collection"""

    data_collection_frequency: int = 20
    """Data collection frequency (Hz)"""

    root_output_dir: str = "outputs"
    """Root output directory"""

    # Policy settings
    enable_upper_body_operation: bool = True
    """Enable upper body operation"""

    upper_body_operation_mode: Literal["teleop", "inference"] = "teleop"
    """Upper body operation mode"""

    def __post_init__(self):
        # Resolve interface (handles sim/real shortcuts, platform differences, and error handling)
        self.interface, self.env_type = resolve_interface(self.interface)

    def load_wbc_yaml(self) -> dict:
        """Load and merge wbc yaml with dataclass overrides"""
        # Get the base path to gr00t_wbc and convert to Path object
        package_path = Path(os.path.dirname(gr00t_wbc.__file__))

        if self.wbc_version == "gear_wbc":
            config_path = str(package_path / "control/main/teleop/configs/g1_29dof_gear_wbc.yaml")
        else:
            raise ValueError(
                f"Invalid wbc_version: {self.wbc_version}, please use one of: " f"gear_wbc"
            )

        with open(config_path) as file:
            wbc_config = yaml.load(file, Loader=yaml.FullLoader)

        # Override yaml values with dataclass values
        wbc_config = override_wbc_config(wbc_config, self)

        return wbc_config

# ... (Rest of the classes: ControlLoopConfig, TeleopConfig, etc. remain unchanged) ...
@dataclass
class ControlLoopConfig(BaseConfig):
    """Config for running the G1 control loop."""
    pass


@dataclass
class TeleopConfig(BaseConfig):
    """Config for running the G1 teleop policy loop."""
    robot: Literal["g1"] = "g1"
    lerobot_replay_path: Optional[str] = None
    body_streamer_ip: str = "10.110.67.24"
    body_streamer_keyword: str = "foot"
    teleop_frequency: float = 20
    binary_hand_ik: bool = True
    hand_tracking_server_host: str = "localhost"
    """Host where the Pico hand tracking server is running (for pico_hand_tracking device)"""
    hand_tracking_server_port: int = 5557
    """Port where the Pico hand tracking server is listening (for pico_hand_tracking device)"""

@dataclass
class ComposedCameraClientConfig:
    """Config for running the composed camera client."""
    camera_port: int = 5555
    camera_host: str = "localhost"
    fps: float = 20.0

@dataclass
class DataExporterConfig(BaseConfig, ComposedCameraClientConfig):
    """Config for running the G1 data exporter."""
    dataset_name: Optional[str] = None
    task_prompt: str = "demo"
    state_dim: int = 43
    action_dim: int = 43
    teleoperator_username: Optional[str] = None
    support_operator_username: Optional[str] = None
    robot_id: Optional[str] = None
    lower_body_policy: Optional[str] = None
    img_stream_viewer: bool = False
    text_to_speech: bool = True
    add_stereo_camera: bool = True
    enable_ftp_hands: bool = False
    """Enable Inspire FTP hand data collection (finger joints + tactile).
    Reads hand state from FTP DDS topics instead of C++ deploy.
    Requires Headless_driver_double.py running on the robot."""

@dataclass
class SyncSimDataCollectionConfig(ControlLoopConfig, TeleopConfig):
    """Args Config for running the data collection loop."""
    robot: str = "G1"
    task_name: str = "GroundOnly"
    body_control_device: str = "dummy"
    hand_control_device: Optional[str] = "dummy"
    remove_existing_dir: bool = False
    hardcode_teleop_cmd: bool = False
    ik_indicator: bool = False
    enable_onscreen: bool = True
    save_img_obs: bool = False
    success_hold_steps: int = 50
    renderer: Literal["mjviewer", "mujoco", "rerun"] = "mjviewer"
    replay_data_path: str | None = None
    replay_speed: float = 2.5
    ci_test: bool = False
    ci_test_mode: Literal["unit", "pre_merge"] = "pre_merge"
    manual_control: bool = False

@dataclass
class SyncSimPlaybackConfig(SyncSimDataCollectionConfig):
    """Configuration class for playback script arguments."""
    enable_real_device: bool = False
    dataset: str | None = None
    use_actions: bool = False
    use_wbc_goals: bool = False
    use_teleop_cmd: bool = False
    save_video: bool = False
    save_lerobot: bool = False
    video_path: str | None = None
    num_episodes: int = 1
    intervention: bool = False
    ci_test: bool = False

    def validate_args(self):
        if self.use_teleop_cmd and not self.use_actions:
            raise ValueError("--use-teleop-cmd requires --use-actions to be set")
        if self.use_teleop_cmd and self.use_wbc_goals:
            raise ValueError("--use-teleop-cmd and --use-wbc-goals are mutually exclusive")
        if (self.use_teleop_cmd or self.use_wbc_goals) and not self.use_actions:
            raise ValueError(
                "You are using --use-teleop-cmd or --use-wbc-goals but not --use-actions. "
                "This will not play back actions whether via teleop or wbc goals. "
                "Instead, it'll play back states only."
            )
        if self.save_img_obs and not self.save_lerobot:
            raise ValueError("--save-img-obs is only supported with --save-lerobot")
        if self.intervention and not self.save_video:
            raise ValueError("--intervention requires --save-video to be enabled for visualization")

@dataclass
class WebcamRecorderConfig(BaseConfig):
    """Config for running the webcam recorder."""
    output_dir: str = "logs_experiment"
    device_id: int = 0
    fps: int = 30
    duration: Optional[int] = None

@dataclass
class SimLoopConfig(BaseConfig):
    """Config for running the simulation loop."""
    mp_start_method: str = "spawn"
    enable_image_publish: bool = False
    camera_port: int = 5555
    verbose: bool = False

@dataclass
class DeploymentConfig(BaseConfig, ComposedCameraClientConfig):
    """G1 Robot Deployment Configuration"""
    camera_publish_rate: float = 30.0
    view_camera: bool = True
    enable_webcam_recording: bool = True
    webcam_output_dir: str = "logs_experiment"
    skip_img_transform: bool = False
    sim_in_single_process: bool = False
    image_publish: bool = False
    # Real robot camera server (composed_camera) resolution; used when start_camera_server runs
    camera_width: int = 640
    camera_height: int = 480
    # Data collection: include wrist cameras in LeRobot dataset (add_stereo_camera)
    add_stereo_camera: bool = True
    # Optional replay/dummy camera sources; if set, start_camera_sensor runs instead of real camera server
    egoview_replay_dummy: Optional[str] = None
    head_replay_dummy: Optional[str] = None