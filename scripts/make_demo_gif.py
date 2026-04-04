"""Generate an animated GIF of the Argus dashboard for the README.

Uses the Selenium screenshots captured by test_full_ui.py.
Resizes to a sensible README width, adds a brief caption bar
on each frame so viewers know which page they are looking at,
then stitches everything into assets/demo.gif.

Run:
    python scripts/make_demo_gif.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# Frames: (screenshot filename, display label, hold seconds)
FRAMES = [
    ("page_overview.png",         "Overview — live metrics & system status",    3.5),
    ("page_drift_analysis.png",   "Drift Analysis — PSI & KS-test per feature", 3.5),
    ("page_feature_insights.png", "Feature Insights — drift radar & importance", 3.5),
    ("page_retraining_log.png",   "Retraining Log — automated decision policy",  3.5),
    ("page_live_demo.png",        "Live Demo — real-time prediction API",         3.5),
]

TARGET_WIDTH  = 1100   # px — good balance for GitHub README
CAPTION_H     = 36     # px — caption strip height
BG_DARK       = (11, 17, 32)
BG_CAPTION    = (13, 28, 56)
TEXT_ACCENT   = (79, 142, 247)
TEXT_LIGHT    = (226, 234, 245)
BORDER_COL    = (45, 63, 90)
MS_PER_FRAME  = 100    # PIL duration unit is milliseconds


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _build_frame(png: Path, label: str) -> Image.Image:
    src = Image.open(png).convert("RGB")

    # Scale to target width
    scale = TARGET_WIDTH / src.width
    h = int(src.height * scale)
    src = src.resize((TARGET_WIDTH, h), Image.LANCZOS)

    # Build caption strip
    cap = Image.new("RGB", (TARGET_WIDTH, CAPTION_H), BG_CAPTION)
    draw = ImageDraw.Draw(cap)

    # Top border line
    draw.line([(0, 0), (TARGET_WIDTH, 0)], fill=BORDER_COL, width=1)

    font_dot  = _load_font(13)
    font_text = _load_font(13)

    # Argus label left
    draw.text((12, 10), "ARGUS", font=font_dot, fill=TEXT_ACCENT)

    # Page label centre
    text_w = draw.textlength(label, font=font_text)
    draw.text(((TARGET_WIDTH - text_w) / 2, 10), label, font=font_text, fill=TEXT_LIGHT)

    # Combine caption (top) + screenshot
    frame = Image.new("RGB", (TARGET_WIDTH, CAPTION_H + h), BG_DARK)
    frame.paste(cap, (0, 0))
    frame.paste(src, (0, CAPTION_H))
    return frame


def main() -> None:
    frames: list[Image.Image] = []
    durations: list[int] = []

    for filename, label, hold_s in FRAMES:
        png = ASSETS / filename
        if not png.exists():
            print(f"  SKIP — {filename} not found (run test_full_ui.py first)")
            continue
        print(f"  Building frame: {label}")
        frame = _build_frame(png, label)
        frames.append(frame)
        durations.append(int(hold_s * 1000))

    if not frames:
        raise SystemExit("No screenshots found. Run: pytest tests/test_full_ui.py -m selenium")

    out = ASSETS / "demo.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,              # loop forever
        optimize=False,
    )
    size_mb = out.stat().st_size / 1_048_576
    print(f"\nSaved: {out}  ({size_mb:.1f} MB, {len(frames)} frames)")
    if size_mb > 10:
        print("  Note: GIF > 10 MB — GitHub renders up to ~25 MB but it may load slowly.")
        print("  Consider hosting on Hugging Face and linking from the README instead.")


if __name__ == "__main__":
    main()
