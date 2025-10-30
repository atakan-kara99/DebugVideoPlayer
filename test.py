from decord_video_processor import DecordVideoProcessor


def run(frame):
    return frame

if __name__ == "__main__":
    # Create and configure the video processor.
    video_processor = DecordVideoProcessor('test_009_1Tor.mp4')
    video_processor.register_window("Test", run)
    video_processor.process_video_multi()
