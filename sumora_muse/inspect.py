"""Agents with eyes: SAM 3.1 finds, Muse Wonder judges.

Used by the front desk of a visual-inspection business (our judge demo, Drishti
Vision, inspects fabric for small textile mills). A mill owner sends a photo of
a fabric roll on WhatsApp:

1. **SAM 3.1** proposes every candidate defect region for a few open-vocabulary
   prompts. SAM localises well but its labels overlap ("hole" also catches stains).
2. Overlapping boxes are merged.
3. **Muse Wonder** looks at the photo with the numbered boxes and classifies each
   region (hole / stain / tear / not a defect) with a severity. A perception
   model and a reasoning model each do what they are best at.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from .client import MuseClient
from .router import Router

PROMPTS = ("hole", "stain", "tear", "fabric defect")

FINDINGS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "inspection",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["regions", "verdict", "summary_for_customer"],
            "properties": {
                "regions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["box_number", "kind", "severity"],
                        "properties": {
                            "box_number": {"type": "integer"},
                            "kind": {"type": "string", "enum": ["hole", "stain", "tear", "slub", "not_a_defect"]},
                            "severity": {"type": "string", "enum": ["minor", "major", "critical"]},
                        },
                    },
                },
                "verdict": {"type": "string", "enum": ["pass", "rework", "reject"]},
                "summary_for_customer": {"type": "string"},
            },
        },
    },
}


@dataclass
class Finding:
    box: dict[str, int]
    kind: str
    severity: str


@dataclass
class Inspection:
    findings: list[Finding]
    verdict: str
    summary: str
    candidates: int


def _iou(a: dict[str, int], b: dict[str, int]) -> float:
    ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    area = lambda r: max(1, (r["x2"] - r["x1"]) * (r["y2"] - r["y1"]))  # noqa: E731
    return inter / (area(a) + area(b) - inter)


def merge_boxes(boxes: list[dict[str, int]], threshold: float = 0.5) -> list[dict[str, int]]:
    kept: list[dict[str, int]] = []
    for box in sorted(boxes, key=lambda r: (r["x2"] - r["x1"]) * (r["y2"] - r["y1"]), reverse=True):
        if all(_iou(box, k) < threshold for k in kept):
            kept.append(box)
    return sorted(kept, key=lambda r: (r["x1"], r["y1"]))


def annotate(image: bytes, boxes: list[dict[str, int]]) -> bytes:
    """Draw numbered boxes (needs Pillow: pip install 'sumora-muse[vision]')."""
    from PIL import Image, ImageDraw

    img = Image.open(io.BytesIO(image)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for n, b in enumerate(boxes, 1):
        sx, sy = img.width / b.get("w", img.width), img.height / b.get("h", img.height)
        xy = (b["x1"] * sx, b["y1"] * sy, b["x2"] * sx, b["y2"] * sy)
        draw.rectangle(xy, outline=(250, 204, 21), width=4)
        draw.rectangle((xy[0], xy[1] - 26, xy[0] + 30, xy[1]), fill=(250, 204, 21))
        draw.text((xy[0] + 9, xy[1] - 23), str(n), fill=(0, 0, 0))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


def inspect_fabric(client: MuseClient, router: Router, image: bytes) -> Inspection:
    boxes: list[dict[str, int]] = []
    for prompt in PROMPTS:
        boxes += client.segment(image, prompt)
    regions = merge_boxes(boxes)
    if not regions:
        return Inspection([], "pass", "No defects found in this photo.", 0)
    marked = annotate(image, regions)
    listing = "\n".join(
        f"box {n}: x1={b['x1']} y1={b['y1']} x2={b['x2']} y2={b['y2']} (image {b.get('w')}x{b.get('h')})"
        for n, b in enumerate(regions, 1)
    )
    result = router.chat(
        "vision",
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": "You are a textile QC inspector. SAM 3.1 proposed these numbered regions on a "
                                         "fabric photo (yellow boxes). Classify every box; say not_a_defect if it is "
                                         "clean fabric. Then give a verdict and a two-sentence summary for the mill owner.\n" + listing},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(marked).decode()}},
            ],
        }],
        response_format=FINDINGS_SCHEMA,
        reasoning_effort="medium",
    )
    data = result.json()
    findings = [
        Finding(regions[r["box_number"] - 1], r["kind"], r["severity"])
        for r in data["regions"]
        if 1 <= r["box_number"] <= len(regions) and r["kind"] != "not_a_defect"
    ]
    return Inspection(findings, data["verdict"], data["summary_for_customer"], len(regions))
