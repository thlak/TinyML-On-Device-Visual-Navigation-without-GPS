#!/usr/bin/env python3
"""Burst-capture labelled sessions on the Pi camera, with live preview on the dashboard.

One session = one class at one height/lighting. While it runs, the person holding the
rig keeps moving it (rotation, tilt, offset, partly out of frame). Images go to
data/day_<X>/<class>/ and a row per image is appended to data/day_<X>/log.csv.
Sessions are the unit for the train/val split.

Days: A, B = original room. C = new room. D = held-out test day (never used for training).

Examples:
  python3 capture.py --day C --cls pad_A --height 1.0 --lighting window
  python3 capture.py --day C --cls no_target --height 1.5 --lighting lamp --note "wood floor" --repeat
"""
import argparse
import csv
import os
import threading
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2

from config import CLASSES, CAPTURE_SIZE, DATA_ROOT, JPEG_QUALITY, camera_config

LOG_FIELDS = ["file", "day", "class", "session", "idx", "timestamp",
              "height_m", "angle", "lighting", "note", "width", "height_px"]
PREVIEW_SIZE = (320, 240)
PREVIEW_FPS = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--day", required=True, type=str.upper,
                   help="A/B = original room, C (or another letter) = new room")
    p.add_argument("--cls", required=True, choices=CLASSES)
    p.add_argument("--height", type=float, required=True, help="approx rig height, m")
    p.add_argument("--lighting", required=True, choices=["window", "ceiling", "lamp", "evening"])
    p.add_argument("--angle", default="mixed", choices=["down", "oblique", "mixed"])
    p.add_argument("--note", default="", help="e.g. surface type for no_target")
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--interval", type=float, default=2.0, help="seconds between shots")
    p.add_argument("--countdown", type=int, default=3)
    p.add_argument("--no-beep", action="store_true")
    p.add_argument("--no-preview", action="store_true", help="don't stream to the dashboard")
    p.add_argument("--repeat", action="store_true",
                   help="after a session: Enter = another session (same settings), q = quit")
    p.add_argument("--manual", action="store_true",
               help="Enter = take one shot; optional text is appended to the note; q = stop")
    a = p.parse_args()
    if len(a.day) != 1 or not a.day.isalpha():
        p.error("--day must be a single letter")
    return a


class Camera:
    """One thread reads the camera; capture and preview both use the newest frame."""

    def __init__(self, preview):
        self.cam = Picamera2()
        self.cam.configure(camera_config(self.cam))
        self.cam.start()
        self.frame, self.stop = None, threading.Event()
        self.aether = None
        if preview:
            try:
                from aether_pi import Aether
                self.aether = Aether()
            except Exception as e:
                print(f"Preview disabled ({e})")
        threading.Thread(target=self._loop, daemon=True).start()
        while self.frame is None:
            time.sleep(0.05)

    def _loop(self):
        last_preview = 0.0
        while not self.stop.is_set():
            self.frame = self.cam.capture_array("main")  # BGR byte order
            if self.aether and time.monotonic() - last_preview >= 1 / PREVIEW_FPS:
                last_preview = time.monotonic()
                small = cv2.resize(self.frame, PREVIEW_SIZE)
                ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.aether.send_frame(jpeg.tobytes())

    def latest(self):
        return self.frame.copy()

    def close(self):
        self.stop.set()
        time.sleep(0.2)
        self.cam.stop()
        if self.aether:
            self.aether.close()


def run_session(a, cam, out_dir, log_path):
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    bell = "" if a.no_beep else "\a"
    for s in range(a.countdown, 0, -1):
        print(f"Starting in {s}…", flush=True)
        time.sleep(1)

    new_log = not os.path.exists(log_path)
    saved = 0
    try:
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new_log:
                w.writeheader()
            t_next = time.monotonic()
            for i in range(a.count):
                shot_note = a.note
                if a.manual:
                    extra = input(f"[{i + 1}/{a.count}] Enter = shoot, text = note, q = stop: ").strip()
                    if extra.lower() == "q":
                        break
                    shot_note = ";".join(x for x in (a.note, extra) if x)
                    time.sleep(0.5)  # let the rig settle after the keypress
                frame = cam.latest()
                fname = f"{session}_{i:03d}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                w.writerow({
                    "file": f"{a.cls}/{fname}", "day": a.day, "class": a.cls,
                    "session": session, "idx": i,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "height_m": a.height, "angle": a.angle, "lighting": a.lighting,
                    "note": shot_note, "width": frame.shape[1], "height_px": frame.shape[0],
                })
                f.flush()
                saved += 1
                print(f"{bell}[{i + 1}/{a.count}] {fname}", flush=True)
                if not a.manual:
                    t_next += a.interval
                    time.sleep(max(0.0, t_next - time.monotonic()))
    except KeyboardInterrupt:
        print("\nStopped early.")
    print(f"Session {session}: {saved} images -> {out_dir} (expected size {CAPTURE_SIZE})")


def main():
    a = parse_args()
    day_dir = os.path.join(DATA_ROOT, f"day_{a.day}")
    out_dir = os.path.join(day_dir, a.cls)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(day_dir, "log.csv")

    cam = Camera(preview=not a.no_preview)
    time.sleep(2)  # let auto-exposure / white balance settle
    try:
        while True:
            run_session(a, cam, out_dir, log_path)
            if not a.repeat or input("Enter = another session, q = quit: ").strip().lower() == "q":
                break
    finally:
        cam.close()


if __name__ == "__main__":
    main()