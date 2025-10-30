import cv2
import time
import numpy as np
import numpy.typing as npt
from typing import Tuple

class DebugOverlay:
    """
    Handles visual overlays on video frames such as playback hints, speed indicators,
    processing durations, frame IDs, and mouse information.
    """

    def __init__(self, seconds_to_skip: int) -> None:
        """
        Initialize the DebugOverlay with default settings.

        Args:
            seconds_to_skip (Union[int, float]): The number of seconds to skip when displaying skip arrows.
        """
        self.font: int = cv2.FONT_HERSHEY_SIMPLEX  # Font style for text overlays.
        self.color: Tuple[int, int, int] = (0, 0, 255)  # Red color for text and icons.

        self.hint_timer: float = 0.0  # Time at which the current hint was set.
        self.hint_text: str = ""  # Current hint text to display.
        self.seconds_to_skip: int = seconds_to_skip

        self.display_mouse_info: bool = False

        # Predefined display durations (in seconds) for various hints.
        self.display_times: dict[str, float] = {
            "PAUSE": 0.5, "RESUME": 0.5, "REWIND": 0.5,
            "BACKWARD": 0.5, "FORWARD": 0.5,
            "SLOWER": 0.25, "FASTER": 0.25,
            "S_BACKWARD": 0.25, "S_FORWARD": 0.25,
        }

    def set_frame(self, frame: npt.NDArray) -> None:
        """
        Set the current frame for overlay operations and compute layout coordinates.

        This method converts grayscale frames to BGR, calculates key coordinates (center,
        margins, offsets), and stores the frame for subsequent drawing.

        Args:
            frame (npt.NDArray): The current video frame. Can be grayscale or color.
        """
        # Convert grayscale or single-channel images to BGR.
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        height, width = frame.shape[:2]
        self.center_x: int = width // 2
        self.center_y: int = height // 2
        self.corner_margin_x: int = int(0.03 * width)
        self.corner_margin_y: int = int(0.04 * height)
        self.triangle_offset_x: int = int(0.015 * width)
        self.triangle_offset_y: int = int(0.035 * height)
        self.resume_bar_width: int = int(0.015 * width)
        self.resume_bar_height: int = int(0.075 * height)
        self.text_scale: float = 0.015 * min(height, width) / 10
        self.text_thickness: int = max(1, int(self.text_scale * 3))
        self.frame_height: int = height
        self.frame_width: int = width
        self.frame: npt.NDArray = frame

    def set_hint(self, hint_text: str) -> None:
        """
        Set the hint text to display and start the hint timer.

        If the hint is "MOUSE", enable the display of mouse information.

        Args:
            hint_text (str): The hint text to be displayed.
        """
        self.hint_text = hint_text
        if hint_text == "MOUSE":
            self.display_mouse_info = True
        else:
            self.hint_timer = time.time()  # Start or reset the hint timer.

    def check_hint_timer(self) -> bool:
        """
        Check whether the display duration for the current hint has elapsed.

        Returns:
            bool: True if the hint display time is over; False otherwise.
        """
        # Get the display duration for the current hint; default to 1 second if not specified.
        display_time: float = self.display_times.get(self.hint_text, 1)
        return time.time() - self.hint_timer > display_time

    def draw_playback_hint(self) -> None:
        """
        Draw playback-related hints (e.g., "PAUSE", "RESUME", "REWIND", or skip arrows)
        on the current frame.
        """
        ht: str = self.hint_text
        if not ht or self.check_hint_timer():
            return  # No active hint or the hint timer has expired.

        if ht in {"PAUSE", "RESUME", "REWIND"}:
            self._draw_playback_icon()
        elif "FORWARD" in ht or "BACKWARD" in ht:
            is_single: bool = ht.startswith("S_")
            message: str = ""
            if not is_single:
                # Extract skip amount from hint text if available.
                if '_' in ht:
                    frames_skipped = ht.split('_')[-1]
                    message += f"{frames_skipped}%"
                else:
                    message += f"{self.seconds_to_skip} sec"
            direction: int = 1 if "FORWARD" in ht else -1
            x_position: int = (7 * self.frame_width) // 8 if direction == 1 else self.frame_width // 8
            self._draw_skip_icon(x_position, self.center_y, direction, is_single, message)

    def draw_playback_speed(self, playback_speed: float) -> None:
        """
        Draw the current playback speed along with speed adjustment indicators.

        The speed is displayed as text in the top-left corner. If a speed adjustment hint
        is active, corresponding triangles are drawn as indicators.

        Args:
            playback_speed (float): The current playback speed multiplier.
        """
        text: str = f"{playback_speed:.1f}x"
        self._draw_text_corner(text, "top-left")

        if self.check_hint_timer():
            return

        # Calculate text size and position for drawing triangles.
        text_size = cv2.getTextSize(text, self.font, self.text_scale, self.text_thickness)[0]
        text_start_x: int = self.corner_margin_x + text_size[0]
        text_start_y: int = self.corner_margin_y - text_size[1] // 2

        # Update triangle dimensions relative to text size.
        self.triangle_offset_x = text_start_x // 12
        self.triangle_offset_y = text_start_y

        if self.hint_text == "SLOWER":
            x: int = text_start_x + self.triangle_offset_x
            y: int = text_start_y + text_size[1]
            self._draw_triangle(x, y, self.triangle_offset_x, "right")
        elif self.hint_text == "FASTER":
            x: int = self.triangle_offset_x
            y: int = text_start_y + text_size[1]
            self._draw_triangle(x, y, self.triangle_offset_x, "left")

    def draw_process_duration(self, processing_time: float) -> None:
        """
        Display the frame processing time in milliseconds at the top-right corner.

        Args:
            processing_time (float): Processing time in seconds.
        """
        processing_time_ms: int = int(processing_time * 1000)
        text: str = f"{processing_time_ms} ms"
        self._draw_text_corner(text, "top-right")

    def draw_frame_id(self, current_frame: int, total_frames: int) -> None:
        """
        Display the current frame number and total frame count at the bottom-right corner.

        Args:
            current_frame (int): The current frame index.
            total_frames (int): The total number of frames.
        """
        text: str = f"{current_frame}/{total_frames}"
        self._draw_text_corner(text, "bottom-right")

    def draw_mouse_info(self, x: int, y: int, hsv: Tuple[int, int, int]) -> None:
        """
        Draw the mouse coordinates and HSV values at the bottom-left corner of the frame.

        Args:
            x (int): The x-coordinate of the mouse.
            y (int): The y-coordinate of the mouse.
            hsv (Tuple[int, int, int]): The HSV values at the mouse position.
        """
        if not self.display_mouse_info:
            return

        coordinates_text: str = f"X, Y: {x}, {y}"
        hsv_text: str = f"H, S, V: {hsv[0]}, {hsv[1]}, {hsv[2]}"
        text_size = cv2.getTextSize(hsv_text, self.font, self.text_scale, self.text_thickness)[0]
        self._draw_text_corner(coordinates_text, "bottom-left", y_offset=-int(text_size[1] * 1.75))
        self._draw_text_corner(hsv_text, "bottom-left")

    def _draw_playback_icon(self) -> None:
        """
        Draw playback icons (pause, resume, or rewind) at the center of the frame.
        """
        ht: str = self.hint_text
        c_x: int = self.center_x
        c_y: int = self.center_y

        if ht == "REWIND":
            # Draw two left-facing triangles for rewind.
            self._draw_triangle(c_x, c_y, x_offset=int(self.triangle_offset_x / 2), direction='left')
            self._draw_triangle(c_x, c_y, x_offset=-int(self.triangle_offset_x * 1.5), direction='left')
        elif ht == "PAUSE":
            # Draw two vertical bars for pause.
            bar_width: int = self.resume_bar_width
            bar_height: int = self.resume_bar_height
            gap: int = int(self.resume_bar_width / 3)
            for offset in [-gap - bar_width, gap]:
                cv2.rectangle(self.frame,
                              (c_x + offset, c_y - bar_height // 2),
                              (c_x + offset + bar_width, c_y + bar_height // 2),
                              self.color, -1)
        elif ht == "RESUME":
            # Draw a right-facing triangle for resume.
            self._draw_triangle(c_x, c_y, x_offset=0, direction='right')

    def _draw_skip_icon(self, x: int, y: int, direction: int, is_single: bool, message: str) -> None:
        """
        Draw skip icons (triangles) to indicate forward or backward skipping.

        Args:
            x (int): Base x-coordinate for the icon.
            y (int): Base y-coordinate for the icon.
            direction (int): 1 for forward, -1 for backward.
            is_single (bool): If True, draw a single triangle; otherwise, multiple triangles.
            message (str): Additional text to display (e.g., skip percentage or seconds).
        """
        triangle_gap: int = self.resume_bar_height
        offsets: list[int] = [0] if is_single else [-triangle_gap, 0, triangle_gap]

        for offset in offsets:
            self._draw_triangle(x, y, x_offset=offset, direction='right' if direction > 0 else 'left')

        if not is_single:
            text_size = cv2.getTextSize(message, self.font, self.text_scale, self.text_thickness)[0]
            text_x: int = x - text_size[0] // 2
            text_x += self.triangle_offset_x // 2 if direction < 0 else -self.triangle_offset_x // 2
            text_y: int = y + int(self.triangle_offset_y * 2.5)
            cv2.putText(self.frame, message, (text_x, text_y),
                        self.font, self.text_scale, self.color, self.text_thickness)

    def _draw_text_corner(self, text: str, position: str, x_offset: int = 0, y_offset: int = 0) -> None:
        """
        Draw text at a specified corner of the frame with optional offsets.

        Args:
            text (str): The text to display.
            position (str): One of 'top-right', 'top-left', 'bottom-right', 'bottom-left'.
            x_offset (int, optional): Horizontal offset. Defaults to 0.
            y_offset (int, optional): Vertical offset. Defaults to 0.

        Raises:
            ValueError: If the position is invalid.
        """
        text_size = cv2.getTextSize(text, self.font, self.text_scale, self.text_thickness)[0]
        c_m_x: int = self.corner_margin_x
        c_m_y: int = self.corner_margin_y

        if position == "top-right":
            text_x = self.frame_width - text_size[0] - c_m_x + x_offset
            text_y = c_m_y + text_size[1] + y_offset
        elif position == "top-left":
            text_x = c_m_x + x_offset
            text_y = c_m_y + text_size[1] + y_offset
        elif position == "bottom-right":
            text_x = self.frame_width - text_size[0] - c_m_x + x_offset
            text_y = self.frame_height - c_m_y + y_offset
        elif position == "bottom-left":
            text_x = c_m_x + x_offset
            text_y = self.frame_height - c_m_y + y_offset
        else:
            raise ValueError("Invalid position. Choose from 'top-right', 'top-left', 'bottom-right', 'bottom-left'.")

        cv2.putText(self.frame, text, (text_x, text_y), self.font, self.text_scale, self.color, self.text_thickness)

    def _draw_triangle(self, x: int, y: int, x_offset: int, direction: str) -> None:
        """
        Draw a triangle at a given position with an offset and in the specified direction.

        Args:
            x (int): Base x-coordinate for the triangle.
            y (int): Base y-coordinate for the triangle.
            x_offset (int): Horizontal offset for adjusting triangle placement.
            direction (str): 'left' or 'right', indicating the triangle's orientation.
        """
        tri_off_x: int = self.triangle_offset_x
        tri_off_y: int = self.triangle_offset_y

        if direction == 'left':
            points = np.array([
                [x + x_offset + tri_off_x, y - tri_off_y],
                [x + x_offset + tri_off_x, y + tri_off_y],
                [x + x_offset - tri_off_x, y]
            ])
        elif direction == 'right':
            points = np.array([
                [x + x_offset - tri_off_x, y - tri_off_y],
                [x + x_offset - tri_off_x, y + tri_off_y],
                [x + x_offset + tri_off_x, y]
            ])
        cv2.fillPoly(self.frame, [points], self.color)
