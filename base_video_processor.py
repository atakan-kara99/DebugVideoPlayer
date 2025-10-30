import sys
import cv2
import time
import threading
import pyautogui
import numpy as np
import numpy.typing as npt
from abc import ABC
from queue import Queue
from copy import deepcopy
from collections import deque
from typing import Any, Callable, Dict, List, Tuple, Union

from utils import Utils
from debug_overlay import DebugOverlay
from user_input_handler import UserInputHandler


# Type alias for window configuration dictionary.
WindowConfig = Dict[str, Any]


class BaseVideoProcessor(ABC):
    """
    Abstract base class for processing video frames with playback controls,
    multiple window support, and overlay information (playback speed, processing time, frame ID).
    """

    def __init__(self, num_frames: int, overlay: bool) -> None:
        """
        Initialize the video processor with default playback settings and overlay configuration.

        Args:
            num_frames (int): Number of frames to be buffered for processing.
            overlay (bool): Flag if the overlay should be turned on or off.
        """
        # Playback settings.
        self.delay: int = 1                      # Delay between frames in milliseconds.
        self.delay_step: int = 2                 # Step factor for adjusting playback delay.
        self.paused: bool = False                # Playback pause flag.
        self.rewind: bool = False                # Rewind playback flag.
        self.seconds_to_skip: int = 3  # Seconds to skip when seeking.
        # Total number of frames to skip, computed based on frame rate.
        self.skip_frames: int = self.frame_rate * self.seconds_to_skip
        self.overlay = overlay
        if overlay:
            # Initialize overlay for displaying playback hints and other information.
            self.playback_overlay: DebugOverlay = DebugOverlay(self.seconds_to_skip)
        # Initialize user input handler.
        self.user_input_handler: UserInputHandler = UserInputHandler(self)
        # Get screen dimensions using pyautogui.
        self.screen_width, self.screen_height = pyautogui.size()
        # Compute geometric mean for zoom calculations.
        self.geometric_mean: float = Utils.geometric_mean(self.screen_width, self.screen_height)
        self.zoom_radius: int = int(self.geometric_mean / 10)
        self.zoom_factor: int = 3

        # Playback timing and speed.
        self.playback_start: Union[float, None] = None
        self.playback_speed: float = 0

        # Frame buffering.
        self.num_frames: int = num_frames
        self.frame_buffer: deque[npt.NDArray] = deque(maxlen=num_frames)
        # List of registered window configurations.
        self.windows: List[WindowConfig] = []

        # System-specific flag.
        self.is_macOS: bool = sys.platform == "darwin"

    def process_video_single(self) -> None:
        """
        Main loop for processing video frames in a single-threaded manner.

        Continuously reads frames, processes them using registered window callbacks,
        and displays the results. Raises a RuntimeError if no windows have been registered.
        """
        if not self.windows:
            raise RuntimeError(
                "No windows registered. Please register at least one window using "
                "register_window before calling process_video_single."
            )
        try:
            while True:
                frame = self.read()  # Read the next frame.
                self.frame_buffer.append(frame)
                # Process and display frames for each registered window.
                for window in self.windows:
                    processed_frame = self._process_frame(window)
                    try:
                        self.display_frame(window, processed_frame)
                    except Exception as e:
                        print(e)
        finally:
            self.close()

    def process_video_multi(self) -> None:
        """
        Main loop for processing video frames using multi-threading.

        Frames are read in the main thread and dispatched to worker threads for processing.
        Processed frames are then collected and displayed. Raises a RuntimeError if no windows
        have been registered.
        """
        if not self.windows:
            raise RuntimeError(
                "No windows registered. Please register at least one window using "
                "register_window before calling process_video_multi."
            )

        # Create queues for exchanging frames between the main thread and worker threads.
        frame_queues: Dict[Tuple[str, ...], Queue] = {tuple(window['names']): Queue() for window in self.windows}
        result_queues: Dict[Tuple[str, ...], Queue] = {tuple(window['names']): Queue() for window in self.windows}

        # Start a worker thread for each registered window.
        threads: List[threading.Thread] = []
        for window in self.windows:
            window_key = tuple(window['names'])
            thread = threading.Thread(
                target=self._process_window_multi,
                args=(window, frame_queues[window_key], result_queues[window_key])
            )
            thread.daemon = True  # Ensure threads exit when main program exits.
            thread.start()
            threads.append(thread)

        try:
            while True:
                frame = self.read()  # Read the next frame.
                self.frame_buffer.append(frame)
                # Dispatch the frame to all worker threads.
                for window in self.windows:
                    window_key = tuple(window['names'])
                    frame_queues[window_key].put(frame)
                # Collect processed frames from each worker thread and display them.
                for window in self.windows:
                    window_key = tuple(window['names'])
                    processed_frame = result_queues[window_key].get()
                    if processed_frame is None:
                        return
                    try:
                        self.display_frame(window, processed_frame)
                    except Exception as e:
                        print(e)
        finally:
            # Signal all worker threads to terminate.
            for window in self.windows:
                window_key = tuple(window['names'])
                frame_queues[window_key].put(None)
            for t in threads:
                t.join()
            self.close()

    def _process_window_multi(
        self, window: WindowConfig, frame_queue: Queue, result_queue: Queue
    ) -> None:
        """
        Worker thread function for processing frames for a specific window.

        Continuously retrieves frames from the frame_queue, processes them using the window's
        callback, and puts the processed frames into the result_queue. Terminates when a None frame
        is received.

        Args:
            window (WindowConfig): The configuration dictionary for the window.
            frame_queue (Queue): Queue from which raw frames are retrieved.
            result_queue (Queue): Queue into which processed frames are placed.
        """
        try:
            while True:
                frame = frame_queue.get()
                if frame is None:  # Termination signal.
                    break
                processed_frame = self._process_frame(window)
                result_queue.put(processed_frame)
        finally:
            # Ensure a termination signal is sent to the main thread.
            result_queue.put(None)

    def _process_frame(self, window: WindowConfig) -> npt.NDArray:
        """
        Process the latest frame from the buffer using the specified window's callback.

        Makes a deep copy of the frame buffer (if full) to ensure thread safety during processing.
        Measures the processing time and updates the window configuration accordingly.

        Args:
            window (WindowConfig): A dictionary containing window settings and a processing callback.

        Returns:
            npt.NDArray: The processed frame.
        """
        current_frame = self.frame_buffer[-1]
        if len(self.frame_buffer) == self.num_frames:
            frames_to_process: Union[npt.NDArray, List[npt.NDArray]] = deepcopy(self.frame_buffer)
            if self.num_frames == 1:
                frames_to_process = frames_to_process[0]
            else:
                frames_to_process = list(frames_to_process)
            # Measure processing time for the callback.
            start_time = time.time()
            processed_frame = window['callback'](frames_to_process)
            window['processing_time'] = time.time() - start_time
            return processed_frame
        return current_frame

    def register_window(
        self, window_names: Union[str, List[str], Tuple[str, ...]],
        frame_callback: Callable[[Union[npt.NDArray, List[npt.NDArray]]], npt.NDArray],
        width: Union[int, None] = None, height: Union[int, None] = None
    ) -> None:
        """
        Register a window for displaying video frames with a given processing callback.

        Creates a new window (or windows) using OpenCV, sets the window size, and assigns
        a mouse callback if the platform is not macOS.

        Args:
            window_names (Union[str, List[str], Tuple[str, ...]]): Name or names of the window(s).
            frame_callback (Callable): Function to process the frame(s) before display.
            width (int, optional): Width of the window. Defaults to half the screen width.
            height (int, optional): Height of the window. Defaults to half the screen height.
        """
        if not isinstance(window_names, (list, tuple)):
            window_names = (window_names,)
        # Register the window configuration.
        self.windows.append({
            'names': window_names,
            'callback': frame_callback,
            'processing_time': 0.0,
        })
        # Determine default window size if not provided.
        width = width or self.screen_width // 2
        height = height or self.screen_height // 2
        for window_name in window_names:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, width, height)
            # Set mouse callback if not running on macOS.
            if not self.is_macOS:
                cv2.setMouseCallback(window_name, self.user_input_handler.handle_mouse_event)

    def register_key(self, key: str, callback: Callable[[], None]) -> None:
        """
        Register a key press with an associated callback function.

        Args:
            key (str): The key character to register.
            callback (Callable): The function to call when the key is pressed.
        """
        self.user_input_handler.register_key(key, callback)

    def display_frame(self, window: WindowConfig, frames: Union[npt.NDArray, List[npt.NDArray]]) -> None:
        """
        Draw overlays and display the processed frame(s) in the specified window(s).

        Applies zoom (if active), sets up overlay information, and displays the frame(s)
        using OpenCV's imshow.

        Args:
            window (WindowConfig): The window configuration dictionary.
            frames (Union[npt.NDArray, List[npt.NDArray]]): The processed frame(s) to display.
        """
        for i, window_name in enumerate(window['names']):
            # Select frame if multiple frames are returned.
            frame = frames[i] if isinstance(frames, (list, tuple)) else frames
            if self.overlay:
                # Apply zoom effect if the user is dragging the mouse.
                if self.user_input_handler.dragging:
                    frame = self._zoom_circle(frame)
                # Update overlay parameters with the current frame.
                self.playback_overlay.set_frame(frame)
                # Draw various overlays.
                self.playback_overlay.draw_playback_hint()
                self.playback_overlay.draw_process_duration(window['processing_time'])
                self.playback_overlay.draw_playback_speed(self.playback_speed)
                self.playback_overlay.draw_frame_id(self.current_frame, self.total_frames)
                # Draw mouse info overlay if enabled.
                if self.playback_overlay.display_mouse_info:
                    mouse_x = self.user_input_handler.mouse_x
                    mouse_y = self.user_input_handler.mouse_y
                    pixel_color = Utils.xy_to_hsv(mouse_x, mouse_y, frame)
                    self.playback_overlay.draw_mouse_info(mouse_x, mouse_y, pixel_color)
                frame = self.playback_overlay.frame
            # Show the final overlayed frame.
            cv2.imshow(window_name, frame)

    def _toggle_rewind(self) -> None:
        """
        Toggle the rewind state and update the overlay hint accordingly.
        """
        self.rewind = not self.rewind
        if self.overlay:
            self.playback_overlay.set_hint("REWIND" if self.rewind else "RESUME")

    def _adjust_delay(self, decrease: bool) -> None:
        """
        Adjust the frame delay to control playback speed.

        Args:
            decrease (bool): If True, decrease the delay (speed up playback);
                             otherwise, increase the delay (slow down playback).
        """
        if decrease:
            self.delay = max(1, int(self.delay / self.delay_step))
        else:
            # Clamp delay to a safe range.
            self.delay = min(max(self.delay * self.delay_step, -(2**31)), 2**31 - 1)
        if self.overlay:
            self.playback_overlay.set_hint("SLOWER" if decrease else "FASTER")

    def _toggle_pause(self) -> None:
        """
        Toggle the paused state of playback and update the overlay hint.
        """
        self.paused = not self.paused
        hint = "PAUSE" if self.paused else ("REWIND" if self.rewind else "RESUME")
        if self.overlay:
            self.playback_overlay.set_hint(hint)

    def _skip_video_frames(
        self, forward: bool = True, single: bool = False, percentage: Union[None, float] = None
    ) -> None:
        """
        Skip a number of frames in the video either forward or backward.

        Args:
            forward (bool, optional): If True, skip forward; otherwise, skip backward. Defaults to True.
            single (bool, optional): If True, perform a single-frame skip. Defaults to False.
            percentage (float, optional): If provided, skip to the frame corresponding to this percentage
                                          of the total video length.
        """
        if single:
            frame_offset = self.frame_offset_forward if forward else self.frame_offset_backward
        else:
            frame_offset = self.skip_frames if forward else -self.skip_frames

        if percentage is not None:
            target_frame = int(self.total_frames * percentage)
            frames_skipped = ((target_frame - self.current_frame) / self.total_frames) * 100
            forward = frames_skipped > 0
        else:
            target_frame = max(0, min(self.total_frames - 1, self.current_frame + frame_offset))

        hint = "S_FORWARD" if forward and single else "S_BACKWARD" if single else "FORWARD" if forward else "BACKWARD"
        if percentage is not None:
            hint += "_" + str(abs(int(frames_skipped)))  # Append percentage value.
        if self.overlay:
            self.playback_overlay.set_hint(hint)
        self.set_frame_pos(target_frame)

    def _update_playback_speed(self) -> None:
        """
        Update the playback speed based on elapsed time between frames.
        """
        if self.paused:
            self.playback_speed = 0
        elif self.playback_start:
            elapsed_time = time.time() - self.playback_start
            fps = 1 / elapsed_time if elapsed_time > 0 else 0
            self.playback_speed = fps / self.frame_rate
        self.playback_start = time.time()

    def _pixel_info(self, turn_on: bool) -> None:
        """
        Toggle the display of mouse-related pixel information.

        Args:
            turn_on (bool): If True, enable mouse pixel info display; otherwise, disable it.
        """
        if turn_on:
            self.playback_overlay.set_hint("MOUSE")
        else:
            self.playback_overlay.display_mouse_info = False

    def _zoom_circle(self, frame: npt.NDArray) -> npt.NDArray:
        """
        Apply a circular zoom effect centered around the current mouse pointer.

        The method extracts a circular region under the mouse pointer, enlarges it,
        and blends it back into the original frame with a thin black border.

        Args:
            frame (npt.NDArray): The original video frame.

        Returns:
            npt.NDArray: The video frame with the zoomed circular region.
        """
        h, w = frame.shape[:2]
        radius = self.zoom_radius
        center_x = self.user_input_handler.mouse_x
        center_y = self.user_input_handler.mouse_y

        # Create a binary mask for the circular region.
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        # Define the bounding box for the circular region.
        x1 = max(center_x - radius, 0)
        y1 = max(center_y - radius, 0)
        x2 = min(center_x + radius, w)
        y2 = min(center_y + radius, h)

        # Extract the region of interest (ROI) and corresponding mask.
        cropped_frame = frame[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]

        # Compute padding amounts to handle edge cases.
        top_pad = max(0, radius - center_y)
        bottom_pad = max(0, (center_y + radius) - h)
        left_pad = max(0, radius - center_x)
        right_pad = max(0, (center_x + radius) - w)

        padded_frame = cv2.copyMakeBorder(
            cropped_frame, top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_CONSTANT, value=0
        )

        # Zoom the padded region.
        factor = self.zoom_factor
        zoomed_padded_frame = cv2.resize(
            padded_frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR
        )

        # Crop the zoomed image back to the size of the original ROI.
        zoomed_h, zoomed_w = zoomed_padded_frame.shape[:2]
        center_zoom_x = zoomed_w // 2
        center_zoom_y = zoomed_h // 2
        new_x1 = center_zoom_x - (x2 - x1) // 2
        new_y1 = center_zoom_y - (y2 - y1) // 2
        new_x2 = new_x1 + (x2 - x1)
        new_y2 = new_y1 + (y2 - y1)
        zoomed_cropped_frame = zoomed_padded_frame[new_y1:new_y2, new_x1:new_x2]

        # Replace the ROI in the original frame with the zoomed version.
        is_color = len(frame.shape) == 3
        if is_color:
            for c in range(frame.shape[2]):
                frame[y1:y2, x1:x2, c] = np.where(
                    cropped_mask > 0,
                    zoomed_cropped_frame[:, :, c],
                    frame[y1:y2, x1:x2, c]
                )
        else:
            frame[y1:y2, x1:x2] = np.where(cropped_mask > 0, zoomed_cropped_frame, frame[y1:y2, x1:x2])

        # Draw a thin black border around the circular region.
        border_color = (0, 0, 0) if is_color else 0
        cv2.circle(frame, (center_x, center_y), radius, border_color, 2)

        return frame

    def set_frame_pos(self, target_frame: int) -> None:
        """
        Set the current frame position to a specified target frame index.

        Args:
            target_frame (int): The target frame index to jump to.
        """
        self.current_frame = target_frame
