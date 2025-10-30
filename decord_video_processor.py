import cv2
import numpy.typing as npt
from typing import Any

from base_video_processor import BaseVideoProcessor


class DecordVideoProcessor(BaseVideoProcessor):
    """
    Processes video frames using Decord with playback controls, multiple window support,
    and on-screen overlays (e.g., playback speed, processing time, and frame ID).
    """

    def __init__(self, video_path: str = 'cv/resources/test_011.mp4', num_frames: int = 1, overlay: bool = True) -> None:
        """
        Initialize the Decord video processor with a video source and default playback settings.

        Args:
            video_path (str): Path to the video file. Defaults to 'cv/resources/test_011.mp4'.
            num_frames (int): Number of frames to process concurrently. Defaults to 1.
            overlay (bool): Flag if the overlay should be turned on or off.

        Raises:
            IOError: If the video source cannot be opened.
        """
        from decord import VideoReader, cpu  # type: ignore

        # Initialize the Decord VideoReader with CPU context.
        try:
            self.video_reader = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            raise IOError(f"Error: Unable to open video source {video_path}. {str(e)}")

        # Retrieve video properties.
        self.frame_rate: int = int(self.video_reader.get_avg_fps())
        self.total_frames: int = len(self.video_reader)
        self.current_frame: int = 0

        # Offsets for frame navigation.
        self.frame_offset_forward: int = 1
        self.frame_offset_backward: int = -1

        # Initialize the base video processor.
        super().__init__(num_frames, overlay)

    def _process_frame(self, window: Any) -> npt.NDArray:
        """
        Process a frame using the callback provided by the window.

        Args:
            window (Any): A window object or identifier used for processing.

        Returns:
            npt.NDArray: The processed frame as a NumPy array.
        """
        return super()._process_frame(window)

    def read(self) -> npt.NDArray:
        """
        Read the next frame from the video source while handling playback adjustments.

        This method updates the current frame index based on the playback state (e.g., paused, rewinding),
        retrieves the frame from Decord, converts it from RGB (Decord's format) to BGR (OpenCV's format),
        and returns the processed frame.

        Returns:
            npt.NDArray: The next video frame in BGR format.

        Raises:
            Exception: If the video cannot be read or if the end of the video is reached.
        """
        # Update playback speed and process user input.
        self._update_playback_speed()
        self.user_input_handler.handle_key_press()

        # Adjust current frame index based on playback state.
        if not self.paused:
            if self.rewind:
                # When rewinding, ensure the index doesn't drop below zero.
                self.current_frame = max(0, self.current_frame - 1)
            else:
                self.current_frame += 1

        # Retrieve the frame using Decord.
        frame = self.video_reader[self.current_frame].asnumpy()
        # Convert frame from RGB (Decord's output) to BGR (OpenCV's format).
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def close(self) -> None:
        """
        Release video resources and close all OpenCV windows.

        Raises:
            Exception: Signals the end of video or a capture error.
        """
        cv2.destroyAllWindows()
        raise Exception("End of video or capture error.")

    def set_frame_pos(self, target_frame: int) -> None:
        """
        Set the current frame position to the specified target frame index.

        Args:
            target_frame (int): The target frame index to jump to.
        """
        self.current_frame = target_frame
