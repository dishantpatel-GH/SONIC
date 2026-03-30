import cv2
import zmq
import time
import struct
import pickle
from collections import deque
import numpy as np
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
import logging

try:
    import logging_mp
    logger_mp = logging_mp.getLogger(__name__)
except (ImportError, AttributeError):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger_mp = logging.getLogger(__name__)

try:
    from teleop.image_server.depth_visualization_3ddp import depth_to_visualization
except ImportError:
    try:
        from decoupled_wbc.control.sensor.depth_visualization_3ddp import depth_to_visualization
    except ImportError:
        try:
            from depth_visualization_3ddp import depth_to_visualization
        except ImportError:
            depth_to_visualization = None


def _depth_to_display(depth: np.ndarray, near_mm: int, far_mm: int, style: str = "3ddp") -> np.ndarray:
    """Use 3D-Diffusion-Policy style (3ddp) or fallback heatmap."""
    if depth_to_visualization is None:
        raise RuntimeError("depth_visualization_3ddp not available (only needed for ImageServer, not client)")
    return depth_to_visualization(depth, style=style, near_mm=near_mm, far_mm=far_mm)


class RealSenseCamera(object):
    def __init__(self, img_shape, fps, serial_number=None, enable_depth=False) -> None:
        """
        img_shape: [height, width]
        serial_number: serial number
        """
        self.img_shape = img_shape
        self.fps = fps
        self.serial_number = serial_number
        self.enable_depth = enable_depth

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.init_realsense()

    def init_realsense(self):

        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number is not None:
            config.enable_device(self.serial_number)

        # Check USB type and adjust FPS for USB 2.x (limited bandwidth)
        # USB 2.x can't handle 30fps with both color+depth at 640x480
        rs_fps = self.fps
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                if dev.get_info(rs.camera_info.serial_number) == self.serial_number:
                    usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
                    if usb_type.startswith("2"):
                        rs_fps = 15  # Use 15fps for USB 2.x
                        logger_mp.warning(f'[RealSense] USB {usb_type} detected, reducing to {rs_fps}fps')
                    break
        except:
            pass

        config.enable_stream(rs.stream.color, self.img_shape[1], self.img_shape[0], rs.format.bgr8, rs_fps)

        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.img_shape[1], self.img_shape[0], rs.format.z16, rs_fps)

        logger_mp.info(f'[RealSense] Starting pipeline: {self.img_shape[1]}x{self.img_shape[0]} @ {rs_fps}fps')
        profile = self.pipeline.start(config)
        self._device = profile.get_device()
        if self._device is None:
            logger_mp.error('[Image Server] pipe_profile.get_device() is None .')
        if self.enable_depth:
            assert self._device is not None
            depth_sensor = self._device.first_depth_sensor()
            self.g_depth_scale = depth_sensor.get_depth_scale()

        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # Depth post-processing: spatial (smooth) -> temporal (persist) -> hole_fill
        self.spatial_filter = rs.spatial_filter() if self.enable_depth else None
        self.temporal_filter = rs.temporal_filter() if self.enable_depth else None
        self.hole_filling_filter = rs.hole_filling_filter() if self.enable_depth else None
        if self.enable_depth:
            try:
                # Moderate smoothing: reduces noise/speckle while keeping edges
                self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
                self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
                self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
                # Temporal: smooth over frames to reduce flicker
                self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
                self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
                # Hole fill: 2=nearest_around
                try:
                    self.hole_filling_filter.set_option(rs.option.holes_fill, 2)
                except Exception:
                    pass
                logger_mp.info('[RealSense] Depth filters: spatial + temporal + hole_fill')
            except Exception as e:
                logger_mp.warning(f'[RealSense] Filter setup: {e}')
        
        # Warm-up: discard first frames to stabilize the camera
        logger_mp.info(f'[RealSense {self.serial_number}] Warming up...')
        for i in range(10):
            try:
                self.pipeline.wait_for_frames(timeout_ms=2000)
                logger_mp.info(f'[RealSense] Warm-up frame {i} OK')
            except Exception as e:
                logger_mp.warning(f'[RealSense] Warm-up frame {i} failed: {e}')
        logger_mp.info(f'[RealSense {self.serial_number}] Ready')

    def get_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except Exception as e:
            logger_mp.warning(f'[RealSense] Frame timeout: {e}')
            return None, None
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if self.enable_depth:
            depth_frame = aligned_frames.get_depth_frame()
            # Post-process: spatial (smooth) -> temporal (reduce flicker) -> hole_fill
            if depth_frame and self.spatial_filter:
                depth_frame = self.spatial_filter.process(depth_frame)
            if depth_frame and self.temporal_filter:
                depth_frame = self.temporal_filter.process(depth_frame)
            if depth_frame and self.hole_filling_filter:
                depth_frame = self.hole_filling_filter.process(depth_frame)

        if not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()) if self.enable_depth and depth_frame else None
        return color_image, depth_image

    def release(self):
        self.pipeline.stop()


class OpenCVCamera():
    def __init__(self, device_id, img_shape, fps):
        """
        decive_id: /dev/video* or *
        img_shape: [height, width]
        """
        self.id = device_id
        self.fps = fps
        self.img_shape = img_shape
        self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Test if the camera can read frames
        if not self._can_read_frame():
            logger_mp.error(f"[Image Server] Camera {self.id} Error: Failed to initialize the camera or read frames. Exiting...")
            self.release()

    def _can_read_frame(self):
        success, _ = self.cap.read()
        return success

    def release(self):
        self.cap.release()

    def get_frame(self):
        ret, color_image = self.cap.read()
        if not ret:
            return None
        return color_image


class ImageServer:
    def __init__(self, config, port = 5555, Unit_Test = False):
        """
        config example1:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'opencv',                                     # opencv or realsense
            'head_camera_image_shape': [480, 1280],                           # Head camera resolution  [height, width]
            'head_camera_id_numbers': [0],                                    # '/dev/video0' (opencv)
            'wrist_camera_type': 'realsense', 
            'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            'wrist_camera_id_numbers': ["218622271789", "241222076627"],      # realsense camera's serial number
        }

        config example2:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'realsense',                                  # opencv or realsense
            'head_camera_image_shape': [480, 640],                            # Head camera resolution  [height, width]
            'head_camera_id_numbers': ["218622271739"],                       # realsense camera's serial number
            'wrist_camera_type': 'opencv', 
            'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            'wrist_camera_id_numbers': [0,1],                                 # '/dev/video0' and '/dev/video1' (opencv)
        }

        If you are not using the wrist camera, you can comment out its configuration, like this below:
        config:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'opencv',                                     # opencv or realsense
            'head_camera_image_shape': [480, 1280],                           # Head camera resolution  [height, width]
            'head_camera_id_numbers': [0],                                    # '/dev/video0' (opencv)
            #'wrist_camera_type': 'realsense', 
            #'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            #'wrist_camera_id_numbers': ["218622271789", "241222076627"],      # serial number (realsense)
        }
        """
        logger_mp.info(config)
        self.fps = config.get('fps', 30)
        self.head_camera_type = config.get('head_camera_type', 'opencv')
        self.head_image_shape = config.get('head_camera_image_shape', [480, 640])      # (height, width)
        self.head_camera_id_numbers = config.get('head_camera_id_numbers', [0])

        self.wrist_camera_type = config.get('wrist_camera_type', None)
        self.wrist_image_shape = config.get('wrist_camera_image_shape', [480, 640])    # (height, width)
        self.wrist_camera_id_numbers = config.get('wrist_camera_id_numbers', None)

        self.port = port
        self.Unit_Test = Unit_Test
        self.depth_near_mm = config.get('depth_near_mm', 250)
        self.depth_far_mm = config.get('depth_far_mm', 4000)
        self.depth_style = config.get('depth_style', '3ddp')  # 3ddp, turbo, jet

        # Initialize head cameras
        self.head_cameras = []
        self.enable_depth = config.get('enable_depth', False)
        if self.head_camera_type == 'opencv':
            for device_id in self.head_camera_id_numbers:
                camera = OpenCVCamera(device_id=device_id, img_shape=self.head_image_shape, fps=self.fps)
                self.head_cameras.append(camera)
        elif self.head_camera_type == 'realsense':
            for serial_number in self.head_camera_id_numbers:
                camera = RealSenseCamera(img_shape=self.head_image_shape, fps=self.fps, serial_number=serial_number, enable_depth=self.enable_depth)
                self.head_cameras.append(camera)
        else:
            logger_mp.warning(f"[Image Server] Unsupported head_camera_type: {self.head_camera_type}")

        # Initialize wrist cameras if provided
        self.wrist_cameras = []
        if self.wrist_camera_type and self.wrist_camera_id_numbers:
            if self.wrist_camera_type == 'opencv':
                for device_id in self.wrist_camera_id_numbers:
                    camera = OpenCVCamera(device_id=device_id, img_shape=self.wrist_image_shape, fps=self.fps)
                    self.wrist_cameras.append(camera)
            elif self.wrist_camera_type == 'realsense':
                for serial_number in self.wrist_camera_id_numbers:
                    camera = RealSenseCamera(img_shape=self.wrist_image_shape, fps=self.fps, serial_number=serial_number)
                    self.wrist_cameras.append(camera)
            else:
                logger_mp.warning(f"[Image Server] Unsupported wrist_camera_type: {self.wrist_camera_type}")

        # Set ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        if self.Unit_Test:
            self._init_performance_metrics()

        for cam in self.head_cameras:
            if isinstance(cam, OpenCVCamera):
                logger_mp.info(f"[Image Server] Head camera {cam.id} resolution: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            elif isinstance(cam, RealSenseCamera):
                logger_mp.info(f"[Image Server] Head camera {cam.serial_number} resolution: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                logger_mp.warning("[Image Server] Unknown camera type in head_cameras.")

        for cam in self.wrist_cameras:
            if isinstance(cam, OpenCVCamera):
                logger_mp.info(f"[Image Server] Wrist camera {cam.id} resolution: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            elif isinstance(cam, RealSenseCamera):
                logger_mp.info(f"[Image Server] Wrist camera {cam.serial_number} resolution: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                logger_mp.warning("[Image Server] Unknown camera type in wrist_cameras.")

        logger_mp.info("[Image Server] Image server has started, waiting for client connections...")



    def _init_performance_metrics(self):
        self.frame_count = 0  # Total frames sent
        self.time_window = 1.0  # Time window for FPS calculation (in seconds)
        self.frame_times = deque()  # Timestamps of frames sent within the time window
        self.start_time = time.time()  # Start time of the streaming

    def _update_performance_metrics(self, current_time):
        # Add current time to frame times deque
        self.frame_times.append(current_time)
        # Remove timestamps outside the time window
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        # Increment frame count
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            logger_mp.info(f"[Image Server] Real-time FPS: {real_time_fps:.2f}, Total frames sent: {self.frame_count}, Elapsed time: {elapsed_time:.2f} sec")

    def _close(self):
        for cam in self.head_cameras:
            cam.release()
        for cam in self.wrist_cameras:
            cam.release()
        self.socket.close()
        self.context.term()
        logger_mp.info("[Image Server] The server has been closed.")

    def send_process(self):
        try:
            while True:
                head_frames = []
                raw_depth_image = None  # Store raw 16-bit depth for recording
                
                for cam in self.head_cameras:
                    if self.head_camera_type == 'opencv':
                        color_image = cam.get_frame()
                        if color_image is None:
                            logger_mp.error("[Image Server] Head camera frame read is error.")
                            break
                        head_frames.append(color_image)
                    elif self.head_camera_type == 'realsense':
                        result = cam.get_frame()
                        if result is None or result[0] is None:
                            logger_mp.error("[Image Server] Head camera frame read is error.")
                            break
                        color_image, depth_image = result
                        # Order: RGB first, then Depth (colorized for display)
                        head_frames.append(color_image)
                        if depth_image is not None:
                            raw_depth_image = depth_image.copy()  # Keep raw 16-bit depth
                            depth_colormap = _depth_to_display(
                                depth_image,
                                near_mm=self.depth_near_mm,
                                far_mm=self.depth_far_mm,
                                style=self.depth_style,
                            )
                            head_frames.append(depth_colormap)
                if not head_frames:
                    break
                head_color = cv2.hconcat(head_frames)
                
                if self.wrist_cameras:
                    wrist_frames = []
                    wrist_ok = True
                    for cam in self.wrist_cameras:
                        if self.wrist_camera_type == 'opencv':
                            color_image = cam.get_frame()
                            if color_image is None:
                                logger_mp.warning("[Image Server] Wrist camera frame read failed, skipping wrist.")
                                wrist_ok = False
                                break
                        elif self.wrist_camera_type == 'realsense':
                            color_image, depth_iamge = cam.get_frame()
                            if color_image is None:
                                logger_mp.warning("[Image Server] Wrist camera frame read failed, skipping wrist.")
                                wrist_ok = False
                                break
                        wrist_frames.append(color_image)
                    
                    if wrist_ok and wrist_frames:
                        wrist_color = cv2.hconcat(wrist_frames)
                        # Concatenate head and wrist frames
                        full_color = cv2.hconcat([head_color, wrist_color])
                    else:
                        # No wrist, just use head
                        full_color = head_color
                else:
                    full_color = head_color

                ret, buffer = cv2.imencode('.jpg', full_color)
                if not ret:
                    logger_mp.error("[Image Server] Frame imencode is failed.")
                    continue

                jpg_bytes = buffer.tobytes()

                # Pack message with raw depth data for recording
                # Format: pickle({'image': jpg_bytes, 'depth_raw': raw_depth_image})
                message_data = {
                    'image': jpg_bytes,
                    'depth_raw': raw_depth_image,  # 16-bit numpy array or None
                }
                message = pickle.dumps(message_data)

                self.socket.send(message)

                if self.Unit_Test:
                    current_time = time.time()
                    self._update_performance_metrics(current_time)
                    self._print_performance_metrics(current_time)

        except KeyboardInterrupt:
            logger_mp.warning("[Image Server] Interrupted by user.")
        finally:
            self._close()


class ComposedCameraClientSensor:
    """ZMQ client that receives camera frames from ImageServer.

    Handles two server formats:
      - New format: pickle({'camera_frames': {name: jpg_bytes}, 'camera_timestamps': {...}})
      - Old format: pickle({'image': jpg_bytes, 'depth_raw': ...})
        The 'image' key contains a single concatenated JPEG [head | wrist].
        With --no-depth --wrist 1: width = head_w + wrist_w (e.g. 640+640=1280).

    Returns data in the format expected by the data exporter:
        {"images": {"ego_view": np.ndarray, ...}, "timestamps": {"ego_view": float, ...}}
    """

    def __init__(self, server_ip="localhost", port=5555, head_width=640):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{server_ip}:{port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._latest_message = None
        self._head_width = head_width
        logger_mp.info(f"[CameraClient] Connected to tcp://{server_ip}:{port}")

    def read(self):
        """Non-blocking receive of latest camera frame."""
        latest_raw = None
        while True:
            try:
                latest_raw = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

        if latest_raw is None:
            return self._latest_message

        try:
            data = pickle.loads(latest_raw)
        except Exception:
            return self._latest_message

        if not isinstance(data, dict):
            return self._latest_message

        frame_time = time.time()

        # --- New format: server sends per-camera JPEGs ---
        camera_frames_jpg = data.get('camera_frames', {})
        if camera_frames_jpg:
            camera_timestamps = data.get('camera_timestamps', {})
            images = {}
            for name, jpg_bytes in camera_frames_jpg.items():
                np_buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
                img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                if img is not None:
                    images[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if images:
                self._latest_message = {"images": images, "timestamps": camera_timestamps}
            return self._latest_message

        # --- Old format: single concatenated JPEG [head | wrist] ---
        jpg_bytes = data.get('image')
        if jpg_bytes is None:
            return self._latest_message

        np_buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
        full_img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if full_img is None:
            return self._latest_message

        h, w = full_img.shape[:2]
        images = {}
        timestamps = {}

        if w > self._head_width:
            # Concatenated: [ego_view | right_wrist]
            images["ego_view"] = cv2.cvtColor(full_img[:, :self._head_width], cv2.COLOR_BGR2RGB)
            images["right_wrist"] = cv2.cvtColor(full_img[:, self._head_width:], cv2.COLOR_BGR2RGB)
        else:
            # Head only (no wrist or width matches head exactly)
            images["ego_view"] = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

        for name in images:
            timestamps[name] = frame_time

        self._latest_message = {"images": images, "timestamps": timestamps}
        return self._latest_message

    def close(self):
        self._socket.close()
        self._context.term()

    def fps(self):
        return 0.0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wrist', type=int, nargs='+', default=[], help='Wrist camera ports (empty or -1 to disable)')
    parser.add_argument('--no-depth', action='store_true', help='Disable depth')
    parser.add_argument('--no-wrist', action='store_true', help='Disable wrist cameras')
    parser.add_argument('--depth-near', type=int, default=250, help='Depth near plane (mm)')
    parser.add_argument('--depth-far', type=int, default=4000, help='Depth far plane (mm)')
    parser.add_argument('--depth-style', type=str, default='turbo', choices=['3ddp', 'turbo', 'jet'],
                        help='3ddp=3D-Diffusion-Policy style (u,v,depth as RGB), turbo/jet=colormap')
    parser.add_argument('--resolution', type=str, default='640x480', help='Width x height')
    args = parser.parse_args()
    
    # Auto-detect RealSense
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("ERROR: No RealSense camera found!")
        exit(1)
    serial = devices[0].get_info(rs.camera_info.serial_number)
    print(f"[AUTO] RealSense: {serial}")
    
    # Handle wrist camera config
    wrist_ports = args.wrist
    if args.no_wrist or wrist_ports == [-1]:
        wrist_ports = []
    
    # Parse resolution
    try:
        w, h = map(int, args.resolution.lower().split('x'))
        img_shape = [h, w]  # [height, width]
    except Exception:
        img_shape = [480, 640]
        logger_mp.warning(f'[CONFIG] Invalid --resolution "{args.resolution}", using 640x480')
    
    config = {
        'fps': 30,
        'head_camera_type': 'realsense',
        'head_camera_image_shape': img_shape,
        'head_camera_id_numbers': [serial],
        'enable_depth': not args.no_depth,
        'depth_near_mm': args.depth_near,
        'depth_far_mm': args.depth_far,
        'depth_style': args.depth_style,
    }
    
    # Only add wrist config if wrist cameras specified
    if wrist_ports:
        config['wrist_camera_type'] = 'opencv'
        config['wrist_camera_image_shape'] = img_shape
        config['wrist_camera_id_numbers'] = wrist_ports
    
    print(f"[CONFIG] Resolution: {config['head_camera_image_shape'][1]}x{config['head_camera_image_shape'][0]}")
    print(f"[CONFIG] Head: RealSense RGB + {'Depth' if config['enable_depth'] else 'No Depth'}")
    if config['enable_depth']:
        print(f"[CONFIG] Depth: {config['depth_near_mm']}-{config['depth_far_mm']}mm, style={config['depth_style']}")
    print(f"[CONFIG] Wrist: {wrist_ports if wrist_ports else 'DISABLED'}")
    
    server = ImageServer(config, Unit_Test=False)
    server.send_process()
