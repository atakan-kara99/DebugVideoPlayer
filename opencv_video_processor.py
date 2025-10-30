import cv2
import numpy.typing as npt
from typing import Any

from base_video_processor import BaseVideoProcessor


class OpenCVVideoProcessor(BaseVideoProcessor):
    """
    Processes video frames using OpenCV with playback controls, multiple window support,
    and on-screen overlays for playback speed, processing time, and frame ID.
    """

    def __init__(self, video_path: str = 'cv/resources/test_011.mp4', num_frames: int = 1, overlay: bool = True) -> None:
        """
        Initialize the OpenCV video processor with the specified video source and playback settings.

        Args:
            video_path (str): Path to the video file. Defaults to 'cv/resources/test_011.mp4'.
            num_frames (int): Number of frames to process concurrently. Defaults to 1.
            overlay (bool): Flag if the overlay should be turned on or off.

        Raises:
            IOError: If the video source cannot be opened.
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Error: Unable to open video source {video_path}")

        # Retrieve video properties.
        self.frame_rate: int = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_changed: bool = False  # Indicates a manual frame change request.
        self.frame_offset_forward: int = 0
        self.frame_offset_backward: int = -2

        # Initialize additional state in the base class.
        super().__init__(num_frames, overlay)

    def _process_frame(self, window: Any) -> npt.NDArray:
        """
        Process a single frame using the provided window's callback.

        Resets the frame_changed flag if needed before delegating processing to the base class.

        Args:
            window (Any): The window object or identifier for display purposes.

        Returns:
            npt.NDArray: The processed video frame.
        """
        if self.frame_changed:
            # Reset the manual frame change flag.
            self.frame_changed = False
        return super()._process_frame(window)

    def read(self) -> npt.NDArray:
        """
        Read the next frame from the video source while applying playback adjustments.

        This method updates the playback speed, handles user input, and adjusts the frame
        position based on pause or rewind states before reading the next frame.

        Returns:
            npt.NDArray: The next video frame.

        Raises:
            Exception: If the end of the video is reached or if there is a capture error.
        """
        # Update playback speed and process any user input.
        self._update_playback_speed()
        self.user_input_handler.handle_key_press()

        # Use the CAP_PROP_POS_FRAMES property for current frame position.
        frame_pos: int = cv2.CAP_PROP_POS_FRAMES

        # Adjust frame position if no manual change has occurred.
        if not self.frame_changed:
            if self.paused:
                # When paused, rewind to the previous frame.
                self.cap.set(frame_pos, self.current_frame - 1)
            elif self.rewind:
                # When rewinding, go back two frames.
                self.cap.set(frame_pos, self.current_frame - 2)

        # Read the next frame.
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("End of video or capture error.")

        # Update the current frame index.
        self.current_frame = int(self.cap.get(frame_pos))
        return frame

    def close(self) -> None:
        """
        Release video resources and close all OpenCV windows.

        Raises:
            Exception: Signals the end of video or a capture error.
        """
        self.cap.release()
        cv2.destroyAllWindows()
        raise Exception("End of video or capture error.")

    def set_frame_pos(self, target_frame: int) -> None:
        """
        Set the current frame position to a specific frame index.

        This method adjusts the video capture to the desired target frame and marks that a
        manual frame change has occurred.

        Args:
            target_frame (int): The desired frame index to jump to.
        """
        frame_pos: int = cv2.CAP_PROP_POS_FRAMES
        self.cap.set(frame_pos, target_frame)
        self.frame_changed = True
