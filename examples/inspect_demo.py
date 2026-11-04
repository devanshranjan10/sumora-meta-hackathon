"""Agents with eyes: a mill owner sends a fabric photo; SAM 3.1 finds, Muse Wonder judges.

    python examples/inspect_demo.py examples/fabric.jpg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sumora_muse import MuseClient, Router, inspect_fabric  # noqa: E402
from sumora_muse.inspect import annotate  # noqa: E402

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("fabric.jpg"))
image = path.read_bytes()
with MuseClient() as client:
    router = Router(client, on_record=lambda r: print(f"   · {r.role:<7} {r.model:<16} {r.latency_ms:>6} ms"))
    result = inspect_fabric(client, router, image)
print(f"SAM 3.1 candidate regions: {result.candidates} · verdict: {result.verdict.upper()}")
for n, f in enumerate(result.findings, 1):
    print(f"  {n}. {f.kind:<6} {f.severity:<8} box={f.box['x1']},{f.box['y1']}–{f.box['x2']},{f.box['y2']}")
print("\nReply to the mill owner:\n" + result.summary)
out = Path("out")
out.mkdir(exist_ok=True)
(out / "inspection.jpg").write_bytes(annotate(image, [f.box for f in result.findings]))
print(f"\nmarked photo: {out / 'inspection.jpg'}")
