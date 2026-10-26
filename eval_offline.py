"""Cache per-image full-frame and best-tile scores for offline gate evaluation."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import infer as runtime

BACKGROUND = runtime.BACKGROUND
TILES = runtime.TILES
Model = runtime.Model
color_ok = runtime.color_ok
locate_cam = runtime.locate_cam

CLASSES = ("pad_A", "pad_B", "pad_C", "pad_D", "pad_E", BACKGROUND)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def candidate(model, rgb, x, y, size):
    height, width = rgb.shape[:2]
    crop = rgb[int(y * height):int((y + size) * height),
               int(x * width):int((x + size) * width)]
    scores = model.predict(crop)
    pad_indices = [i for i, label in enumerate(model.labels) if label != BACKGROUND]
    cls = max(pad_indices, key=lambda i: scores[i])
    second = float(np.delete(scores, cls).max())
    feat = model.last_feat
    box = locate_cam(model, feat, cls) if model.W is not None else None
    if box:
        box = {"x": x + box["x"] * size, "y": y + box["y"] * size,
               "w": box["w"] * size, "h": box["h"] * size}
    else:
        box = {"x": x, "y": y, "w": size, "h": size}
    label = model.labels[cls]
    color_check_enabled = runtime.COLOR_CHECK
    try:
        # Record the colour rule's outcome even when evaluating with that rule disabled.
        runtime.COLOR_CHECK = True
        color_result = bool(color_ok(rgb, box, label))
    finally:
        runtime.COLOR_CHECK = color_check_enabled
    return {
        "label": label,
        "confidence": float(scores[cls]),
        "second": second,
        "color_ok": color_result,
    }


def image_scores(model, rgb):
    full = candidate(model, rgb, 0.0, 0.0, 1.0)
    tiles = [
        candidate(model, rgb, gx * 0.25, gy * 0.25, 0.5)
        for gy in range(3) for gx in range(3)
    ] if TILES else []
    best_tile = max(tiles, key=lambda item: item["confidence"]) if tiles else {}
    return full, best_tile


def load_rows(args, parser):
    if args.directory:
        if args.sessions:
            parser.error("--sessions cannot be used with --directory")
        directory = args.directory
        if not directory.is_dir():
            parser.error(f"not a directory: {directory}")
        log = {}
        if (directory / "log.csv").exists():
            with (directory / "log.csv").open(newline="") as f:
                log = {r["file"]: r for r in csv.DictReader(f)}
        rows = []
        for label in CLASSES:
            class_dir = directory / label
            if not class_dir.is_dir():
                parser.error(f"missing class directory: {class_dir}")
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    rel = image_path.relative_to(directory).as_posix()
                    meta = log.get(rel, {})
                    rows.append({
                        "file": rel,
                        "day": directory.name.removeprefix("day_"),
                        "class": label,
                        "session": meta.get("session", ""),
                        "height_m": meta.get("height_m", ""),
                        "lighting": meta.get("lighting", ""),
                        "note": meta.get("note", ""),
                        "_image_path": image_path,
                    })
    else:
        if not args.day or not args.sessions:
            parser.error("provide --directory or both --day and --sessions")
        day_dir = args.data_root / f"day_{args.day}"
        requested_sessions = set(args.sessions)
        with (day_dir / "log.csv").open(newline="") as log_file:
            rows = [
                {**row, "_image_path": day_dir / row["file"]}
                for row in csv.DictReader(log_file)
                if row["session"] in requested_sessions
            ]
        if not rows:
            parser.error(f"no matching sessions in {day_dir / 'log.csv'}")

    if not rows:
        parser.error("no supported images found")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--directory", type=Path,
                        help="class-folder directory containing pad_A...pad_E and no_target")
    source.add_argument("--day", choices=("A", "B", "C", "E"),
                        help="dataset day; use --directory to explicitly evaluate another day")
    parser.add_argument("--sessions", nargs="+",
                        help="session IDs from data/day_<X>/log.csv (required with --day)")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path,
                        help=".tflite to evaluate (default: whatever infer.py loads)")
    parser.add_argument("--output", type=Path, default=Path("offline_scores.csv"))
    args = parser.parse_args()

    rows = load_rows(args, parser)

    output_fields = [
        "file", "day", "class", "session", "height_m", "lighting", "note",
        "full_label", "full_confidence", "full_second", "full_color_ok",
        "tile_label", "tile_confidence", "tile_second", "tile_color_ok",
    ]
    
    model = Model(args.model) if args.model else Model()
    print(f"Model: {args.model or 'infer.py default'}")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            image_path = row["_image_path"]
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            full, tile = image_scores(model, rgb)
            writer.writerow({
                "file": row["file"],
                "day": row["day"],
                "class": row["class"],
                "session": row["session"],
                "height_m": row.get("height_m", ""),
                "lighting": row.get("lighting", ""),
                "note": row.get("note", ""),
                "full_label": full["label"],
                "full_confidence": full["confidence"],
                "full_second": full["second"],
                "full_color_ok": full["color_ok"],
                "tile_label": tile.get("label", ""),
                "tile_confidence": tile.get("confidence", ""),
                "tile_second": tile.get("second", ""),
                "tile_color_ok": tile.get("color_ok", ""),
            })
            print(f"{row['class']} {row['file']}")
    print(f"Wrote {len(rows)} image scores to {args.output}")


if __name__ == "__main__":
    main()
