"""Live inference on the Pi: camera -> TFLite model -> dashboard.

Needs:
  aether_pi.py   (your classmate's connection file, same folder)
  *.tflite       (MobileNetV2 classifier)
  *.json         ({"classes": [...], "img": 128, ...}) in the same folder as the model

Install:  pip install ai-edge-litert numpy pillow
Run:      python infer.py                            # first *.tflite in the current folder
          python infer.py model/128/model_128_int8.tflite # explicit model path
"""
import argparse
import glob
import io
import json
import os
import shutil
import threading
import time
from collections import Counter, deque

import numpy as np
from PIL import Image
from picamera2 import Picamera2

from aether_pi import FPS, Aether

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

BACKGROUND = "no_target"    # name of the "no landing spot" class
PREPROCESS = "raw"          # model has its own Rescaling layer -> feed 0..255 ("01" / "11" otherwise)
THRESHOLD = 0.85            # min confidence to count as "target seen" (full frame)
TILE_THRESHOLD = 0.95       # stricter for zoomed tiles: they cause most false positives
MARGIN = 0.5                # top-1 must beat top-2 by this much
VOTES, WINDOW = 4, 5        # same pad in >= 4 of the last 5 results before we trust it
RESULT_HZ = 10              # results per second sent to the dashboard
TILES = True                # also check 3x3 zoomed-in tiles -> detects pads from further away
CAPTURE = (640, 480)        # camera resolution used for inference (video stream is sent at 320x240)
SAVE_DETECTIONS = False     # True: save every detected frame to detections/ (hard negatives)
COLOR_CHECK = False         # off: hand-set hue ranges rejected real red/green pads in a new room (day D)
COLOR_DEBUG = False         # print the colour counts for every detection
# Pad colour as hue range in PIL HSV units (0-255). None = no check (black/white pad).
PAD_HUE = {"pad_A": "red", "pad_B": "blue", "pad_C": "yellow", "pad_D": "green", "pad_E": None}
HUES = {"red": [(0, 10), (235, 256)], "yellow": [(28, 50)], "green": [(60, 120)], "blue": [(135, 185)]}


def dequant(x, detail):
    """Tensor -> float32, undoing int8/uint8 quantization if the tensor has it."""
    x = x.astype(np.float32)
    if detail["dtype"] != np.float32:
        scale, zero = detail["quantization"]
        x = (x - zero) * scale
    return x


class Model:
    def __init__(self, model_path=None):
        if model_path is None:
            found = sorted(glob.glob("*.tflite"))
            if not found:
                raise SystemExit("No .tflite in the current folder - pass a model path.")
            model_path = found[0]
        self.path = model_path
        folder = os.path.dirname(self.path) or "."
        configs = sorted(glob.glob(os.path.join(folder, "*.json")))
        if not configs:
            raise SystemExit(f"No .json config next to {self.path}")
        with open(configs[0]) as f:
            self.labels = json.load(f)["classes"]

        self.it = Interpreter(model_path=self.path, num_threads=4)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        # 2 outputs: feature map (4x4x1280 at 128 px, 3x3x1280 at 96 px) and [1, N] class scores
        outs = self.it.get_output_details()
        self.out = next(d for d in outs if len(d["shape"]) == 2)
        self.feat_out = next(d for d in outs if len(d["shape"]) == 4)
        self.h, self.w = self.inp["shape"][1:3]
        self.pads = [i for i, l in enumerate(self.labels) if l != BACKGROUND]
        self.W = self._dense_weights(folder)  # (channels, classes) or None
        self.info = f"{os.path.basename(self.path)} · {self.inp['dtype'].__name__} · {self.w}×{self.h}"
        print(f"Model: {self.info} | classes {self.labels} | "
              f"localisation: {'CAM' if self.W is not None else 'tile box'}")

    def _dense_weights(self, folder):
        """Final Dense weights for CAM: cam_weights.npy from the model's folder, else from the .tflite."""
        npy = os.path.join(folder, "cam_weights.npy")
        if os.path.exists(npy):
            return np.load(npy).astype(np.float32)  # (C, N), float, saved from Colab
        C, N = self.feat_out["shape"][-1], len(self.labels)
        for d in self.it.get_tensor_details():
            if tuple(d["shape"]) == (N, C):  # TFLite stores FullyConnected as (out, in)
                try:
                    w = self.it.get_tensor(d["index"]).astype(np.float32)
                except ValueError:
                    return None
                q = d["quantization_parameters"]
                if d["dtype"] != np.float32 and len(q["scales"]):  # int8 weights -> float
                    w = (w - q["zero_points"].reshape(-1, 1)) * q["scales"].reshape(-1, 1)
                return w.T
        return None

    def predict(self, rgb):
        """rgb: HxWx3 uint8 -> probabilities (float). Feature map stored in self.last_feat."""
        x = np.asarray(Image.fromarray(rgb).resize((self.w, self.h)), np.float32)
        if PREPROCESS == "01":
            x = x / 255.0
        elif PREPROCESS == "11":
            x = x / 127.5 - 1.0
        if self.inp["dtype"] != np.float32:  # int8 / uint8 quantized input
            scale, zero = self.inp["quantization"]
            info = np.iinfo(self.inp["dtype"])
            x = np.clip(np.round(x / scale + zero), info.min, info.max)
        self.it.set_tensor(self.inp["index"], x[None].astype(self.inp["dtype"]))
        self.it.invoke()
        self.last_feat = dequant(self.it.get_tensor(self.feat_out["index"])[0], self.feat_out)
        y = dequant(self.it.get_tensor(self.out["index"])[0], self.out)
        if y.min() < 0 or y.sum() > 1.01:  # logits -> softmax
            y = np.exp(y - y.max()); y /= y.sum()
        return y


def locate_cam(model, feat, cls):
    """Class activation map on the feature map -> box (0..1) around the hot cells."""
    cam = np.maximum(feat @ model.W[:, cls], 0)
    if cam.max() <= 0:
        return None
    gh, gw = cam.shape
    ys, xs = np.nonzero(cam >= 0.5 * cam.max())
    return {"x": float(xs.min() / gw), "y": float(ys.min() / gh),
            "w": float((xs.max() - xs.min() + 1) / gw), "h": float((ys.max() - ys.min() + 1) / gh)}


REGIONS = [(0.0, 0.0, 1.0)]  # (x0, y0, size) in frame fractions
if TILES:
    REGIONS += [(gx * 0.25, gy * 0.25, 0.5) for gy in range(3) for gx in range(3)]


def analyse(model, rgb):
    """Run the model on the full frame and (if TILES) on 3x3 half-size zoomed tiles.
    Returns (label, conf, second, box, from_tile) of the best pad candidate, box in full-frame 0..1 coords."""
    H, W = rgb.shape[:2]
    best = None
    for x0, y0, size in REGIONS:
        crop = rgb[int(y0 * H):int((y0 + size) * H), int(x0 * W):int((x0 + size) * W)]
        p = model.predict(crop)
        cls = max(model.pads, key=lambda i: p[i])
        if best is None or p[cls] > best[1]:
            best = (cls, float(p[cls]), float(np.delete(p, cls).max()), (x0, y0, size), model.last_feat)

    cls, conf, second, (x0, y0, size), feat = best
    box = locate_cam(model, feat, cls) if model.W is not None else None
    if box:  # map tile-relative box -> full frame
        box = {"x": x0 + box["x"] * size, "y": y0 + box["y"] * size,
               "w": box["w"] * size, "h": box["h"] * size}
    else:
        box = {"x": x0, "y": y0, "w": size, "h": size}
    return model.labels[cls], conf, second, box, size < 1.0


def color_ok(rgb, box, label):
    """True if the pad's own colour dominates the other pad colours inside the box.
    Wood/grey/white pixels are ignored (low saturation or hue outside all ranges)."""
    want = PAD_HUE.get(label)
    if not COLOR_CHECK or want is None:
        return True
    H, W = rgb.shape[:2]
    x0, y0 = int(box["x"] * W), int(box["y"] * H)
    x1, y1 = int((box["x"] + box["w"]) * W), int((box["y"] + box["h"]) * H)
    hsv = np.asarray(Image.fromarray(rgb[y0:y1, x0:x1]).convert("HSV")).reshape(-1, 3).astype(int)
    hsv = hsv[(hsv[:, 1] > 90) & (hsv[:, 2] > 60)]          # saturated, not too dark
    counts = {c: int(sum(((hsv[:, 0] >= a) & (hsv[:, 0] < b)).sum() for a, b in rng)) for c, rng in HUES.items()}
    rival = max(v for c, v in counts.items() if c != want)
    ok = counts[want] >= 50 and counts[want] >= 1.5 * rival
    if COLOR_DEBUG:
        print(f"{label}: want {want}  counts {counts}  -> {'OK' if ok else 'REJECT'}")
    return ok


def command_for(x, y):
    moves = []
    if y < 0.44: moves.append("forward")
    if y > 0.56: moves.append("back")
    if x < 0.44: moves.append("left")
    if x > 0.56: moves.append("right")
    return "-".join(moves) or "land"


def temperature():
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        return round(int(f.read()) / 1000, 1)


def storage():
    d = shutil.disk_usage("/")
    return round(100 * d.used / d.total)


latest = None  # newest camera frame, shared with the inference thread


def inference_loop(model, aether, stop):
    last_command, last_frame = "", None
    history = deque(maxlen=WINDOW)
    if SAVE_DETECTIONS:
        os.makedirs("detections", exist_ok=True)
    while not stop.is_set():
        rgb = latest
        if rgb is None or rgb is last_frame:  # wait for a new camera frame
            time.sleep(0.01); continue
        last_frame = rgb

        t0 = time.time()
        label, conf, second, box, from_tile = analyse(model, rgb)

        # Per-frame decision, then a vote over the last few frames
        frame_ok = conf >= (TILE_THRESHOLD if from_tile else THRESHOLD) and conf - second >= MARGIN
        history.append(label if frame_ok else BACKGROUND)
        voted, n = Counter(history).most_common(1)[0]
        confirmed = frame_ok and voted == label and n >= VOTES and color_ok(rgb, box, label)

        if SAVE_DETECTIONS and frame_ok:
            Image.fromarray(rgb).save(f"detections/{label}_{conf:.2f}_{time.time():.0f}.jpg")

        data = {"command": "search", "target": None}
        if confirmed:
            cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
            data["command"] = command_for(cx, cy)
            data["target"] = {"label": label, "confidence": round(conf, 2), "box": box}
            data["offset"] = {"x": (cx - 0.5) * 1.5, "y": (0.5 - cy) * 1.1}

        data.update(latency=round((time.time() - t0) * 1000), temperature=temperature(),
                    storage=storage(), model=model.info)
        if data["command"] != last_command:
            data["event"] = "Command: " + data["command"]
            last_command = data["command"]
        aether.send(data)
        time.sleep(max(0, 1 / RESULT_HZ - (time.time() - t0)))


def main():
    global latest
    parser = argparse.ArgumentParser(description="Live landing-pad inference on the Pi.")
    parser.add_argument("model", nargs="?", help="path to the .tflite model "
                        "(default: first *.tflite in the current folder); the .json config must sit next to it")
    model = Model(parser.parse_args().model)

    camera = Picamera2()
    camera.configure(camera.create_video_configuration(main={"size": CAPTURE, "format": "XBGR8888"}))
    camera.start()

    aether = Aether()
    stop = threading.Event()
    threading.Thread(target=inference_loop, args=(model, aether, stop), daemon=True).start()
    try:
        while True:
            started = time.time()
            latest = camera.capture_array()[:, :, :3]   # XBGR8888 -> RGB
            jpeg = io.BytesIO()
            Image.fromarray(latest).resize((320, 240)).save(jpeg, "JPEG", quality=70)
            aether.send_frame(jpeg.getvalue())
            time.sleep(max(0, 1 / FPS - (time.time() - started)))
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        stop.set()
        camera.stop()
        aether.close()


if __name__ == "__main__":
    main()