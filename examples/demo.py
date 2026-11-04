"""End-to-end demo of Sumora's Meta layer against the live Meta Model API.

    export MODEL_API_KEY=...   # from dev.meta.ai
    python examples/demo.py "I run a home bakery in Pune. Custom cakes and cardamom buns, orders on WhatsApp."

Steps:
1. Muse Spark (structured output) designs the business's agent team.
2. The Front Desk agent (Muse Spark tool calling) handles a real customer message.
   Booking a slot runs; issuing a refund is held for the owner's approval.
3. Muse Image paints the hero image for the business's website.
4. Muse Voice Transcribe turns an owner voice note into a command (if --voice file.wav is given).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sumora_muse import MuseClient, Router, Tool, plan_team, run_agent  # noqa: E402

CALENDAR: dict[str, str | None] = {"2026-11-06 16:00": None, "2026-11-06 18:00": None}


def check_slots(day: str) -> dict:
    return {"free": [s for s, who in CALENDAR.items() if s.startswith(day) and who is None]}


def book_slot(slot: str, customer: str, item: str) -> dict:
    if slot not in CALENDAR or CALENDAR[slot] is not None:
        return {"ok": False, "reason": "slot not free"}
    CALENDAR[slot] = f"{customer}: {item}"
    return {"ok": True, "slot": slot}


def refund(order_id: str, amount_inr: int) -> dict:  # never runs without the owner
    return {"ok": True}


def move_booking(customer: str, from_slot: str, to_slot: str) -> dict:
    if CALENDAR.get(from_slot, "") is None or not str(CALENDAR.get(from_slot, "")).startswith(customer):
        return {"ok": False, "reason": f"no booking for {customer} at {from_slot}"}
    if to_slot not in CALENDAR or CALENDAR[to_slot] is not None:
        return {"ok": False, "reason": "target slot not free"}
    CALENDAR[to_slot], CALENDAR[from_slot] = CALENDAR[from_slot], None
    return {"ok": True, "moved_to": to_slot}


OUTBOX: list[dict] = []


def send_whatsapp(customer: str, text: str) -> dict:
    OUTBOX.append({"to": customer, "text": text})  # Sumora sends via the WhatsApp Business Platform
    return {"ok": True, "queued": True}


TOOLS = [
    Tool("check_slots", "List free pickup slots for a day (YYYY-MM-DD).",
         {"type": "object", "properties": {"day": {"type": "string"}}, "required": ["day"]}, check_slots),
    Tool("book_slot", "Book a pickup slot for a customer order.",
         {"type": "object", "properties": {"slot": {"type": "string"}, "customer": {"type": "string"},
                                           "item": {"type": "string"}}, "required": ["slot", "customer", "item"]},
         book_slot),
    Tool("refund", "Refund money to a customer for an order.",
         {"type": "object", "properties": {"order_id": {"type": "string"}, "amount_inr": {"type": "integer"}},
          "required": ["order_id", "amount_inr"]}, refund, risky=True),
]

OWNER_TOOLS = TOOLS + [
    Tool("move_booking", "Move an existing customer booking to another slot.",
         {"type": "object", "properties": {"customer": {"type": "string"}, "from_slot": {"type": "string"},
                                           "to_slot": {"type": "string"}}, "required": ["customer", "from_slot", "to_slot"]},
         move_booking),
    Tool("send_whatsapp", "Send a WhatsApp message to an existing customer.",
         {"type": "object", "properties": {"customer": {"type": "string"}, "text": {"type": "string"}},
          "required": ["customer", "text"]}, send_whatsapp),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("business", nargs="?", default="I run a home bakery in Pune. Custom cakes and cardamom buns; customers order on WhatsApp and pick up from my home.")
    ap.add_argument("--voice", type=Path, help="owner voice note (WAV) to transcribe")
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    with MuseClient() as client:
        router = Router(client, on_record=lambda r: print(f"   · {r.role:<8} {r.model:<16} {r.latency_ms:>6} ms  ok={r.ok}"))

        print("1) Muse Spark designs the team")
        team = plan_team(router, args.business)
        print(f"   {team.business_summary}  [{team.timezone}]")
        for a in team.agents:
            print(f"   - {a.name}: {a.job}  (approval: {', '.join(a.needs_approval_for) or 'none'})")

        print("\n2) Front Desk agent handles a customer (Muse Spark tool calling)")
        run = run_agent(
            router,
            instructions=("You are the front desk of the bakery described here: " + team.business_summary +
                          " Today is 2026-11-05. Be brief and warm. Use tools; never invent availability."),
            user_message=("Hi! Priya here. Can I pick up a 1 kg chocolate cake on 6 Nov evening? "
                          "Also my last order #A17 was squashed, please refund ₹450."),
            tools=TOOLS,
        )
        for step in run.tool_log:
            print(f"   tool {step['tool']}({json.dumps(step['args'])}) -> {json.dumps(step['output'])}")
        for req in run.approvals:
            print(f"   ⏸  held for owner approval: {req.tool} {json.dumps(req.arguments)}")
        print(f"   reply: {run.reply}")

        # Priya picks 4 pm; the front desk books it (as it would on her next message).
        book_slot("2026-11-06 16:00", "Priya", "1 kg chocolate cake")

        if not args.no_image:
            print("\n3) Muse Image paints the website hero")
            img = client.generate_image(
                f"warm editorial photograph for the website hero of: {team.business_summary}. Natural light, no text, no logos."
            )[0]
            path = args.out / "hero.webp"
            path.write_bytes(img)
            print(f"   saved {path} ({len(img)//1024} KB)")

        if args.voice:
            print("\n4) Muse Voice Transcribe: owner voice note -> command")
            text = client.transcribe(args.voice.read_bytes())
            print(f"   heard: {text}")
            cmd = run_agent(
                router,
                role="command",
                instructions=("You are the bakery owner's chief of staff. Today is 2026-11-05, so 'tomorrow' is 2026-11-06. "
                              "Slots look like 'YYYY-MM-DD HH:MM'. Act with tools, then confirm in one line. "
                              f"Current bookings: {json.dumps({k: v for k, v in CALENDAR.items() if v})}"),
                user_message=text,
                tools=OWNER_TOOLS,
            )
            for step in cmd.tool_log:
                print(f"   tool {step['tool']}({json.dumps(step['args'], ensure_ascii=False)}) -> {json.dumps(step['output'])}")
            print(f"   reply: {cmd.reply}")

        total = sum(r.prompt_tokens + r.completion_tokens for r in router.records)
        print(f"\nModel calls: {len(router.records)} · tokens: {total} · every call ran on Meta Muse.")


if __name__ == "__main__":
    main()
