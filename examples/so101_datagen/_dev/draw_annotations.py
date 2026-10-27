"""Draw the stored VQA boxes and points on the decoded frames they belong to, as a contact sheet."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lerobot.datasets import LeRobotDataset

root, out = Path(sys.argv[1]), sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 12
ds = LeRobotDataset("local/so101_datagen", root=root)
hf = ds.hf_dataset
tiles = []
for i in range(len(hf)):
    events = hf[i]["language_events"]
    for answer in (e for e in events if e["role"] == "assistant" and e["style"] == "vqa"):
        content = json.loads(answer["content"])
        if "detections" not in content and "point" not in content:
            continue
        item = ds[i]
        image = (item[answer["camera"]].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for det in content.get("detections", []):
            x1, y1, x2, y2 = det["bbox"]
            draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=(255, 0, 0), width=2)
            draw.text((x1 * w + 2, y1 * h + 2), det["label"], fill=(255, 255, 0))
        if "point" in content:
            x, y = content["point"]
            draw.ellipse([x * w - 5, y * h - 5, x * w + 5, y * h + 5], outline=(0, 255, 0), width=3)
            draw.text((x * w + 6, y * h), content["label"], fill=(0, 255, 0))
        tiles.append(np.asarray(img.resize((320, 240))))
        if len(tiles) >= limit:
            break
    if len(tiles) >= limit:
        break
cols = 4
rows = [np.concatenate(tiles[i : i + cols], 1) for i in range(0, len(tiles) - cols + 1, cols)]
Image.fromarray(np.concatenate(rows, 0)).save(out)
print(len(tiles), "tiles")
