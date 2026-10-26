"""Report figure: day-D results of the full on-device pipeline (offline replay).

Usage:  python3 plot_day_d.py scores_D.csv day_d_results.png
Gates match infer.py: highest-confidence region (full frame or best tile),
THRESHOLD 0.85 (full) / 0.95 (tile), MARGIN 0.5, colour check off.
"""
import csv
import sys
from collections import Counter

import matplotlib.pyplot as plt

THR, THR_TILE, MARGIN, BG = 0.85, 0.95, 0.5, "no_target"
PADS = ["pad_A", "pad_B", "pad_C", "pad_D", "pad_E"]
COL = {"correct": "#2e7d32", "wrong": "#c62828", "rejected": "#9e9e9e"}


def decide(r):
    src = "tile" if float(r["tile_confidence"]) > float(r["full_confidence"]) else "full"
    conf, second = float(r[f"{src}_confidence"]), float(r[f"{src}_second"])
    if conf < (THR_TILE if src == "tile" else THR) or conf - second < MARGIN:
        return BG
    return r[f"{src}_label"]


def outcome(r):
    p = decide(r)
    if r["class"] == BG:
        return "correct" if p == BG else "wrong"
    return "correct" if p == r["class"] else ("rejected" if p == BG else "wrong")


def stacked(ax, groups, counts, legend):
    bottom = [0] * len(groups)
    for key, name in legend:
        vals = [counts[g][key] for g in groups]
        bars = ax.bar(groups, vals, bottom=bottom, color=COL[key], label=name, width=0.6)
        for b, v, y in zip(bars, vals, bottom):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, y + v / 2, str(v),
                        ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        bottom = [y + v for y, v in zip(bottom, vals)]
    ax.set_ylabel("Images")
    ax.spines[["top", "right"]].set_visible(False)


def main(src, out):
    rows = list(csv.DictReader(open(src)))
    for r in rows:
        r["out"] = outcome(r)

    heights = sorted({r["height_m"] for r in rows if r["class"] != BG}, key=float)
    by_h = {f"{h} m": Counter(r["out"] for r in rows if r["class"] != BG and r["height_m"] == h)
            for h in heights}
    by_c = {c: Counter(r["out"] for r in rows if r["class"] == c) for c in PADS}
    neg = {
        "empty floor": Counter(r["out"] for r in rows if r["class"] == BG and "floor" in r["note"]),
        "decoys": Counter(r["out"] for r in rows if r["class"] == BG and "decoy" in r["note"]),
    }

    pads = [r for r in rows if r["class"] != BG]
    negs = [r for r in rows if r["class"] == BG]
    ok = sum(r["out"] == "correct" for r in rows)
    print(f"overall {ok}/{len(rows)} = {ok / len(rows):.0%} | "
          f"pad recall {sum(r['out'] == 'correct' for r in pads)}/{len(pads)} | "
          f"wrong place {sum(r['out'] == 'wrong' for r in pads)}/{len(pads)} | "
          f"false positives {sum(r['out'] == 'wrong' for r in negs)}/{len(negs)}")

    pad_legend = [("correct", "Correct place"), ("wrong", "Wrong place"), ("rejected", "Rejected")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 5, 2]})
    stacked(axes[0], list(by_h), by_h, pad_legend)
    axes[0].set_title("(a) Pads by height")
    stacked(axes[1], PADS, by_c, pad_legend)
    axes[1].set_title("(b) Pads by class")
    stacked(axes[2], list(neg), neg, [("correct", "Correctly rejected"), ("wrong", "False positive")])
    axes[2].set_title("(c) no_target")

    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("correct", "wrong", "rejected")]
    fig.legend(handles, ["Correct", "Wrong place / false positive", "Rejected as no_target"],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Day D (unseen day and room), full Pi pipeline: {ok}/{len(rows)} correct "
                 f"({ok / len(rows):.0%})", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scores_D.csv",
         sys.argv[2] if len(sys.argv) > 2 else "day_d_results.png")