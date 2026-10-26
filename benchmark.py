"""Benchmark model and full-pipeline performance on the Raspberry Pi.

Reports inference latency, pipeline FPS, per-step timing, model size and temperature.

Run on the Pi from ~/aether (next to infer.py and aether_pi.py):
  python benchmark.py --models model/128/landing_final.tflite model/128/model_128_int8.tflite \
                              model/96/landing_final.tflite  model/96/model_96_int8.tflite \
                      --images data/day_D --output benchmark.csv --camera

Measures, per model (after warm-up):
  invoke_ms    one model.invoke() on a fixed input             = pure inference time
  predict_ms   infer.Model.predict() on a full frame            = one inference incl. resize + outputs
  analyse_ms   infer.analyse() exactly as in the live loop      = full frame + 9 tiles + CAM
  per-step     mean ms per frame spent in preprocess / invoke / postprocess / CAM (10 inferences)
  fps          1000 / median analyse_ms  (pipeline capacity; the live loop is capped at RESULT_HZ)
Camera capture (--camera) is timed once; it runs in its own thread in infer.py, so it does not
add to the inference time, but it bounds the frame rate if it is slower than analyse().
"""
import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

import infer as runtime


def ms(start):
    return (time.perf_counter() - start) * 1000


def med_p95(values):
    a = np.asarray(values)
    return round(float(np.median(a)), 2), round(float(np.percentile(a, 95)), 2)


def throttled():
    """Pi throttling flags: 0x0 = never throttled. Non-zero bits mean timings may be inflated."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout
        return out.strip().split("=")[-1]
    except FileNotFoundError:
        return ""


def load_frames(directory, n):
    """Load n real images and resize them to the deployed camera resolution."""
    paths = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise SystemExit(f"no images in {directory}")
    step = max(1, len(paths) // n)
    return [np.asarray(Image.open(p).convert("RGB").resize(runtime.CAPTURE)) for p in paths[::step][:n]]


def is_int8(model):
    return any(d["dtype"] == np.int8 for d in model.it.get_tensor_details())


def timed_frame(model, rgb, t):
    """Same work as infer.analyse(), split into steps. Assumes float I/O and PREPROCESS = 'raw'."""
    H, W = rgb.shape[:2]
    regions = [(0.0, 0.0, 1.0)]
    if runtime.TILES:
        regions += [(gx * 0.25, gy * 0.25, 0.5) for gy in range(3) for gx in range(3)]
    pads = [i for i, l in enumerate(model.labels) if l != runtime.BACKGROUND]
    best = None
    for x0, y0, size in regions:
        s = time.perf_counter()
        crop = rgb[int(y0 * H):int((y0 + size) * H), int(x0 * W):int((x0 + size) * W)]
        x = np.asarray(Image.fromarray(crop).resize((model.w, model.h)), np.float32)[None]
        t["preprocess"] += ms(s)

        s = time.perf_counter()
        model.it.set_tensor(model.inp["index"], x)
        model.it.invoke()
        t["invoke"] += ms(s)

        s = time.perf_counter()
        p = model.it.get_tensor(model.out["index"])[0].astype(np.float32)
        feat = model.it.get_tensor(model.feat_out["index"])[0].astype(np.float32)
        cls = max(pads, key=lambda i: p[i])
        if best is None or p[cls] > best[0]:
            best = (p[cls], cls, feat)
        t["postprocess"] += ms(s)

    s = time.perf_counter()
    model.last_feat = best[2]
    if model.W is not None:
        runtime.locate_cam(model, best[1])
    t["cam"] += ms(s)


def time_camera(n=100):
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"size": runtime.CAPTURE, "format": "XBGR8888"}))
    cam.start()
    time.sleep(2)  # let exposure settle
    for _ in range(10):
        cam.capture_array()
    times = []
    for _ in range(n):
        s = time.perf_counter()
        cam.capture_array()
        times.append(ms(s))
    cam.stop()
    cam.close()
    return med_p95(times)


def bench(path, frames, args):
    model = runtime.Model(path)
    assert model.inp["dtype"] == np.float32 and runtime.PREPROCESS == "raw", "expects float I/O, raw input"

    for i in range(args.warmup):                       # warm-up: caches, XNNPACK setup, CPU clock
        runtime.analyse(model, frames[i % len(frames)])
    temp_start = runtime.temperature()

    x = np.asarray(Image.fromarray(frames[0]).resize((model.w, model.h)), np.float32)[None]
    model.it.set_tensor(model.inp["index"], x)
    invoke = []
    for _ in range(args.runs):
        s = time.perf_counter()
        model.it.invoke()
        invoke.append(ms(s))

    predict = []
    for i in range(args.runs):
        rgb = frames[i % len(frames)]
        s = time.perf_counter()
        model.predict(rgb)
        predict.append(ms(s))

    analyse = []
    for i in range(args.frames):
        rgb = frames[i % len(frames)]
        s = time.perf_counter()
        runtime.analyse(model, rgb)
        analyse.append(ms(s))

    steps = dict.fromkeys(["preprocess", "invoke", "postprocess", "cam"], 0.0)
    for i in range(args.frames):
        timed_frame(model, frames[i % len(frames)], steps)

    inv_med, inv_p95 = med_p95(invoke)
    pre_med, pre_p95 = med_p95(predict)
    ana_med, ana_p95 = med_p95(analyse)
    row = {
        "model": path,
        "img": f"{model.w}x{model.h}",
        "weights": "int8" if is_int8(model) else "float32",
        "size_kb": round(Path(path).stat().st_size / 1024),
        "threads": 4,
        "invoke_ms_median": inv_med, "invoke_ms_p95": inv_p95,
        "predict_ms_median": pre_med, "predict_ms_p95": pre_p95,
        "analyse_ms_median": ana_med, "analyse_ms_p95": ana_p95,
        "fps": round(1000 / ana_med, 1),
        "fps_full_frame_only": round(1000 / pre_med, 1),
        **{f"step_{k}_ms": round(v / args.frames, 2) for k, v in steps.items()},
        "temp_start_c": temp_start,
        "temp_end_c": runtime.temperature(),
        "throttled": throttled(),
    }
    print(f"{path}: invoke {inv_med} ms | full loop {ana_med} ms (p95 {ana_p95}) -> {row['fps']} FPS "
          f"| {row['size_kb']} KB | {row['temp_end_c']} C | throttled {row['throttled']}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--images", default="data/day_D", help="folder with real frames (searched recursively)")
    parser.add_argument("--output", default="benchmark.csv")
    parser.add_argument("--n-images", type=int, default=20, help="distinct frames to cycle through")
    parser.add_argument("--warmup", type=int, default=10, help="warm-up frames (full loop)")
    parser.add_argument("--runs", type=int, default=200, help="timed single inferences")
    parser.add_argument("--frames", type=int, default=100, help="timed full-loop frames")
    parser.add_argument("--cooldown", type=int, default=30, help="seconds between models")
    parser.add_argument("--camera", action="store_true", help="also time camera capture")
    args = parser.parse_args()

    frames = load_frames(args.images, args.n_images)
    print(f"{len(frames)} frames at {runtime.CAPTURE}, TILES={runtime.TILES}")

    rows = []
    for i, path in enumerate(args.models):
        if i:
            time.sleep(args.cooldown)
        rows.append(bench(path, frames, args))

    if args.camera:
        cap_med, cap_p95 = time_camera()
        print(f"camera capture: {cap_med} ms median (p95 {cap_p95}) -> max {1000 / cap_med:.1f} FPS")
        for r in rows:
            r["capture_ms_median"], r["capture_ms_p95"] = cap_med, cap_p95

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()