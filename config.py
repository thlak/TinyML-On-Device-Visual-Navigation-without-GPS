"""Camera and dataset settings used by capture.py.

Keep its capture dimensions aligned with the corresponding settings in infer.py.
"""

# Edit to your real targets. Keep "no_target" last.
CLASSES = ["pad_A", "pad_B", "pad_C", "pad_D", "pad_E", "no_target"]

DATA_ROOT = "data"          # data/day_A/<class>/, data/day_B/<class>/
CAPTURE_SIZE = (640, 480)   # (w, h), 4:3 VGA; must match runtime input dimensions
SENSOR_MODE = (2304, 1296)  # full field of view, binned (the 1536x864 mode is cropped)
JPEG_QUALITY = 95


def camera_config(cam):
    """Same config for dataset capture and runtime.

    Uses a *video* configuration on purpose: still and video configs can pick
    different sensor modes, which changes field of view and sharpness.
    NOTE: picamera2's "RGB888" delivers pixels in BGR order. cv2.imwrite expects
    BGR, so saved JPEGs are correct. At runtime, convert BGR->RGB before
    preprocessing, because Keras loads training images as RGB.
    """
    return cam.create_video_configuration(
        main={"size": CAPTURE_SIZE, "format": "RGB888"},
        sensor={"output_size": SENSOR_MODE},
        controls={"AfMode": 2},  # Camera Module 3: continuous autofocus (2 = Continuous)
        buffer_count=4,
    )
