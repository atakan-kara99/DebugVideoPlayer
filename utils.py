import cv2
import numpy as np


class Utils:
    """
    A collection of utility functions for image processing tasks.

    This class provides methods to compute geometric means, convert between color spaces,
    generate custom grayscale images, compute histograms, crop and warp frames, and convert
    pixel coordinates to relative percentages.
    """

    @staticmethod
    def geometric_mean(height: int, width: int) -> float:
        """
        Calculate the geometric mean of two values.

        Args:
            height (int): The first value.
            width (int): The second value.

        Returns:
            float: The geometric mean of height and width.
        """
        return np.sqrt(height * width)

    @staticmethod
    def xy_to_hsv(x: int, y: int, frame: np.ndarray) -> np.ndarray:
        """
        Convert the BGR (or grayscale) pixel value at (x, y) in a frame to HSV.

        If the frame is grayscale, the pixel value is replicated across BGR channels.

        Args:
            x (int): The x-coordinate of the pixel.
            y (int): The y-coordinate of the pixel.
            frame (np.ndarray): The input image frame (BGR or grayscale).

        Returns:
            np.ndarray: The HSV value of the pixel as a 1D array of 3 elements.
        """
        if len(frame.shape) == 2:  # Grayscale image.
            gray: int = frame[y, x]
            bgr: np.ndarray = np.array([gray, gray, gray])
        else:  # Color image.
            bgr = frame[y, x]
        return Utils.bgr_to_hsv(bgr)

    @staticmethod
    def bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
        """
        Convert a single BGR pixel value to HSV.

        Args:
            bgr (np.ndarray): A 1D array or list of 3 elements representing a BGR pixel.

        Returns:
            np.ndarray: A 1D array representing the HSV value of the input pixel.
        """
        # Create a 1x1 image from the BGR pixel.
        bgr_pixel: np.ndarray = np.uint8([[bgr]])
        hsv_pixel: np.ndarray = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
        return hsv_pixel[0, 0]
