# Debug Video Player

A powerful tool for video frame processing, playback control, and debugging. It offers overlays, custom frame processing, and interactive mouse-based controls, making it ideal for development and testing.

---

## Features

- **Playback Controls**:  
  Pause, resume, rewind, adjust speed, and skip frames.

- **Mouse Interaction**:  
  Inspect pixel coordinates and HSV values by clicking and dragging on the video.

- **Multiple Window Support**:  
  Display video in customizable, resizable windows.

- **Overlays**:  
  Show playback speed, frame ID, processing time, and helpful hints.

- **Custom Frame Processing**:  
  Apply custom transformations to single or consecutive video frames.

- **Multithreading Support**:  
  Process video frames efficiently in parallel.

---

## Overlay Elements

The Debug Video Player includes an **Information Overlay** that displays real-time data:

1. **Playback Speed** (Top Left):  
   Displays the current playback speed (e.g., `1.5x`).

2. **Processing Time** (Top Right):  
   Shows the frame processing time in milliseconds (e.g., `23.45 ms`).

3. **Pixel Information** (Bottom Left):  
   On clicking, displays the pixel coordinates and HSV values (e.g., `X, Y: 120, 80, HSV: 255, 128, 64`).

4. **Frame ID** (Bottom Right):  
   Indicates the current frame number (e.g., `1234/12345`).

---

## Controls

### Mouse Controls

- **Left Click & Drag**:  
  Displays pixel coordinates and HSV values under the mouse pointer and zooms into that area.  
  *(Note: Not available on macOS.)*

### Keyboard Controls

- **`q`**: Quit the player.
- **`d`**: Increase playback speed.
- **`s`**: Decrease playback speed.
- **Space (` `)**: Pause/Resume playback.
- **`p`**: Rewind playback.
- **`k`**: Skip a single frame if paused; otherwise, skip 3 seconds backward.
- **`l`**: Skip a single frame if paused; otherwise, skip 3 seconds forward.
- **`0`–`9`**: Jump to a specific percentage of the video (e.g., pressing 5 jumps to 50%).

---

## Dependencies

- **OpenCV**
- **PyAutoGUI**
- **NumPy**
- **Decord**

---

## Usage

### 1. Initialize the Processor

Create a `DecordVideoProcessor` or `OpenCVVideoProcessor` instance with your video file.

```python
from debug_player import DecordVideoProcessor, OpenCVVideoProcessor

# Specify the video file path.
video_path = 'path/to/video.mp4'

# Initialize a processor using Decord.
processor = DecordVideoProcessor(video_path=video_path)

# Alternatively, initialize using OpenCV.
processor2 = OpenCVVideoProcessor(video_path=video_path)

# Optionally, specify how many frames to process at once.
processor3 = DecordVideoProcessor(video_path=video_path, num_frames=2)
```

### 2. Register Windows with Custom Frame Processing

Each window can apply custom processing to frames before displaying them. Window sizes can also be specified. By default, the screen is divided into four chunks.

#### Single-Frame Processing

When `num_frames` is set to 1 (default), you receive a single frame.

```python
import cv2

def custom_processing(frame):
    # Example: Convert frame to grayscale.
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Register a window with default size.
processor.register_window("Main Window", frame_callback=custom_processing)

# Register a window with custom dimensions.
processor.register_window("Resized Window", frame_callback=custom_processing, width=800, height=600)
```

#### Multi-Frame Processing (List of Frames)

When `num_frames` is greater than 1, the callback receives a list of frames.

```python
def custom_processing2(frames):
    # Example: Compute the absolute difference between the first and last frame.
    return cv2.absdiff(frames[0], frames[-1])

# Register a window that processes multiple frames.
processor2.register_window("Multiple Frames Processed Window", frame_callback=custom_processing2)
```

#### Multiple Windows from a Single Callback

You can register multiple windows at once if your callback returns multiple frames.

```python
def custom_processing3(frames):
    frame1, frame2 = frames
    # Example: Compute the absolute difference and return all frames.
    diff = cv2.absdiff(frame1, frame2)
    return frame1, diff, frame2

# Register multiple windows simultaneously.
processor2.register_window(("Frame 1", "Diff", "Frame 2"), frame_callback=custom_processing3)
```

### 3. Register Keys with Custom Callbacks

Customize key bindings by associating keys with callback functions.

```python
def custom_key():
    print("Custom key pressed!")

# Register a custom key (e.g., 'm') and its callback.
processor.register_key(key='m', callback=custom_key)
```

### 4. Run the Video Player

Start processing and playback using either single-threaded or multi-threaded modes.

```python
# Start single-threaded video processing.
processor.process_video_single()

# Or start multi-threaded video processing.
processor.process_video_multi()
```

---

Enjoy debugging your videos effortlessly with the Debug Video Player!
