"""Compose the gallery card image (900x640): a real clip frame with the model's recorded boxes.

    uv run python gallery/make_thumbnail.py        # writes gallery/pr/yolox-traffic.png

Not a browser screenshot: indah's cards are normally captured from the running app, and this can be
replaced by one (scripts/capture_thumbnails.py) once the demo is in the gallery.
"""

import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
EX = HERE / "examples/yolox_traffic"
COLORS = {"car": "#3fb6c4", "truck": "#e9a23b", "bus": "#c77dd6", "person": "#7fd36b"}
NAMES = "person,bicycle,car,motorcycle,airplane,bus,train,truck".split(",")


def font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main(frame: int = 103) -> None:
    meta = json.loads((EX / "replay.json").read_text())
    blob = (EX / "frames.bin").read_bytes()
    o, n = meta["frames"][frame]
    img = Image.open(io.BytesIO(blob[o : o + n])).convert("RGB").resize((900, 506), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    for x, y, w, h, c, s in meta["frames_detections"][frame]:
        name = NAMES[c] if c < len(NAMES) else str(c)
        col = COLORS.get(name, "#d0d7de")
        box = (x * 900, y * 506, (x + w) * 900, (y + h) * 506)
        draw.rectangle(box, outline=col, width=3)
        draw.rectangle((box[0], box[1] - 20, box[0] + 11 * len(name) + 44, box[1]), fill=col)
        draw.text((box[0] + 4, box[1] - 18), f"{name} {s:.2f}", fill="#0b1217", font=font(14))
    card = Image.new("RGB", (900, 640), "#14212b")
    card.paste(img, (0, 0))
    d = ImageDraw.Draw(card)
    d.text((24, 526), "YOLOX-Tiny INT8, live or replay", fill="#e5ecf0", font=font(30))
    d.text(
        (24, 574),
        "A 5.6 MB quantized detector on a traffic clip it never saw",
        fill="#90a2ae",
        font=font(20),
    )
    card.save(HERE / "pr/yolox-traffic.png", optimize=True)
    print(HERE / "pr/yolox-traffic.png")


if __name__ == "__main__":
    main()
