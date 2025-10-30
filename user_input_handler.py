import cv2
from typing import Optional, Callable, Dict, Any


class UserInputHandler:
    """
    Handles keyboard and mouse input for controlling video playback.

    This class maps specific key presses and mouse events to corresponding
    actions on a video processor (e.g., pausing, adjusting speed, skipping frames,
    or displaying pixel information).
    """

    def __init__(self, video_processor) -> None:
        """
        Initialize the user input handler with a video processor.

        Args:
            video_processor (DecordVideoProcessor): An instance of the video processor
                that provides methods for controlling video playback.
        """
        self.vp = video_processor
        self.dragging: bool = False
        self.mouse_x: Optional[int] = None
        self.mouse_y: Optional[int] = None

        # Map key codes to their corresponding callback functions.
        self.controls: Dict[int, Callable[[], None]] = {
            ord('q'): self.vp.close,  # Quit the application.
            ord('d'): lambda: self.vp._adjust_delay(decrease=True),  # Increase playback speed.
            ord('s'): lambda: self.vp._adjust_delay(decrease=False),  # Decrease playback speed.
            ord(' '): self.vp._toggle_pause,  # Toggle pause/resume.
            ord('k'): lambda: self.vp._skip_video_frames(forward=False, single=self.vp.paused),  # Skip backward.
            ord('l'): lambda: self.vp._skip_video_frames(forward=True, single=self.vp.paused),  # Skip forward.
            ord('p'): self.vp._toggle_rewind  # Rewind.
        }

    def register_key(self, key: str, callback: Callable[[], None]) -> None:
        """
        Register a new key press with its associated callback function.

        Args:
            key (str): A single-character string representing the key.
            callback (Callable[[], None]): A function to be called when the key is pressed.

        Raises:
            ValueError: If the key is not a single character or if the callback is not callable.
        """
        if not (isinstance(key, str) and len(key) == 1):
            raise ValueError("Key must be a single character.")
        if not callable(callback):
            raise ValueError("Callback must be callable.")
        self.controls[ord(key)] = callback

    def handle_key_press(self) -> None:
        """
        Capture and process a key press to control video playback.

        The function checks for registered key actions. If no registered key is pressed,
        it also checks for numeric keys (0-9) to skip to a specific percentage of the video.
        """
        key = cv2.waitKey(self.vp.delay) & 0xFF
        action = self.controls.get(key)
        if action:
            action()  # Execute the corresponding action.
        else:
            # Check for numeric keys (0-9) to perform a percentage-based skip.
            if ord('0') <= key <= ord('9'):
                number_pressed = key - ord('0')
                percentage = number_pressed / 10
                self.vp._skip_video_frames(percentage=percentage)

    def handle_mouse_event(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """
        Handle mouse events for interactive video display.

        This callback function handles:
          - Left mouse button press: starts dragging and displays pixel info.
          - Mouse movement: updates the current mouse coordinates when dragging.
          - Left mouse button release: stops dragging and hides pixel info.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_LBUTTONDOWN).
            x (int): The x-coordinate of the mouse event.
            y (int): The y-coordinate of the mouse event.
            flags (int): Any relevant flags passed by OpenCV.
            param (Any): Additional parameters (unused).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start dragging: record initial mouse position and enable pixel info display.
            self.dragging = True
            self.mouse_x = x
            self.mouse_y = y
            self.vp._pixel_info(True)
        elif event == cv2.EVENT_MOUSEMOVE:
            # Update mouse coordinates if dragging is active.
            if self.dragging:
                self.mouse_x = x
                self.mouse_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            # End dragging: reset dragging state and disable pixel info display.
            self.dragging = False
            self.vp._pixel_info(False)
