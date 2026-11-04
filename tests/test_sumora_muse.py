import base64
import json

import httpx
import pytest

from sumora_muse import MuseClient, MuseError, Router, Tool, plan_team, run_agent


def make_client(handler):
    return MuseClient("test-key", transport=httpx.MockTransport(handler), sleep=lambda s: None)


def completion(content=None, tool_calls=None, finish="stop", model="muse-spark-1.3", reasoning=0):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "completion_tokens_details": {"reasoning_tokens": reasoning}},
    }


def test_auth_header_and_budget_growth_when_reasoning_starves_output():
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body["max_tokens"])
        assert req.headers["authorization"] == "Bearer test-key"
        if body["max_tokens"] < 1000:
            return httpx.Response(200, json=completion(None, finish="length", reasoning=body["max_tokens"]))
        return httpx.Response(200, json=completion("hello"))

    res = make_client(handler).chat([{"role": "user", "content": "hi"}], max_tokens=64)
    assert res.content == "hello"
    assert seen == [64, 256, 1024]


def test_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json=completion("ok"))

    assert make_client(handler).chat([{"role": "user", "content": "x"}]).content == "ok"
    assert calls["n"] == 2


def test_router_falls_back_on_server_error_and_records_models():
    def handler(req):
        model = json.loads(req.content)["model"]
        if model == "muse-spark-1.3":
            return httpx.Response(503, json={"error": "down"})
        return httpx.Response(200, json=completion("from fallback", model=model))

    client = make_client(handler)
    client.max_retries = 0
    router = Router(client)
    res = router.chat("command", [{"role": "user", "content": "x"}])
    assert res.content == "from fallback"
    assert [(r.model, r.ok) for r in router.records] == [("muse-spark-1.3", False), ("muse-spark-1.2", True)]


def test_router_does_not_mask_client_errors():
    client = make_client(lambda req: httpx.Response(400, json={"error": "bad request"}))
    with pytest.raises(MuseError) as exc:
        Router(client).chat("agent", [{"role": "user", "content": "x"}])
    assert exc.value.status == 400


def test_plan_team_parses_structured_output():
    team = {
        "business_summary": "Home bakery in Pune",
        "timezone": "Asia/Kolkata",
        "agents": [
            {"name": "Front Desk", "job": "orders", "channels": ["whatsapp"], "tools": ["book_slot"], "needs_approval_for": ["refund"]},
            {"name": "Site", "job": "website", "channels": ["website"], "tools": [], "needs_approval_for": ["publish"]},
            {"name": "Brief", "job": "daily brief", "channels": ["whatsapp"], "tools": [], "needs_approval_for": []},
        ],
    }

    def handler(req):
        body = json.loads(req.content)
        assert body["response_format"]["type"] == "json_schema"
        return httpx.Response(200, json=completion("```json\n" + json.dumps(team) + "\n```"))

    out = plan_team(Router(make_client(handler)), "I bake cakes at home in Pune")
    assert out.timezone == "Asia/Kolkata"
    assert [a.name for a in out.agents] == ["Front Desk", "Site", "Brief"]


def test_agent_runs_safe_tools_and_holds_risky_ones_for_approval():
    executed = []
    turns = iter([
        completion(tool_calls=[
            {"id": "1", "type": "function", "function": {"name": "book", "arguments": json.dumps({"slot": "16:00"})}},
            {"id": "2", "type": "function", "function": {"name": "refund", "arguments": json.dumps({"amount": 450})}},
        ], finish="tool_calls"),
        completion("Booked 4 pm. The owner will confirm your refund."),
    ])

    def handler(req):
        return httpx.Response(200, json=next(turns))

    tools = [
        Tool("book", "book", {"type": "object", "properties": {}}, lambda slot: executed.append(("book", slot)) or {"ok": True}),
        Tool("refund", "refund", {"type": "object", "properties": {}}, lambda amount: executed.append(("refund", amount)), risky=True),
    ]
    run = run_agent(Router(make_client(handler)), instructions="x", user_message="y", tools=tools)
    assert executed == [("book", "16:00")]
    assert [a.tool for a in run.approvals] == ["refund"]
    assert run.reply.startswith("Booked")


def test_generate_image_decodes_b64():
    png = b"\x89PNG fake"

    def handler(req):
        assert req.url.path.endswith("/images/generations")
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png).decode()}], "output_format": "webp"})

    assert make_client(handler).generate_image("a bakery")[0] == png


def test_transcribe_sends_multipart_request_and_audio():
    def handler(req):
        assert req.url.path.endswith("/asr/transcribe")
        body = req.content
        assert b'name="request"' in body and b"muse-voice-transcribe-1.0" in body
        assert b'name="audio"' in body
        return httpx.Response(200, json={"transcript": " move Priya to 6 pm ", "turns": []})

    assert make_client(handler).transcribe(b"RIFF....WAVE") == "move Priya to 6 pm"


def test_read_menu_photo_sends_image_to_vision_role_and_parses_catalogue():
    from sumora_muse import read_menu_photo

    cat = {"business_name": "Asha's Home Bakery", "items": [
        {"name": "Cardamom buns", "price_inr": 240, "unit": "box of 6", "price_is_starting": False},
        {"name": "Custom theme cake", "price_inr": 1800, "unit": "kg", "price_is_starting": True}],
        "opening_or_pickup_hours": "11am-8pm", "closed_days": ["Monday"], "area": "Baner, Pune", "payment": ["UPI"], "lead_time_days": 2}

    def handler(req):
        body = json.loads(req.content)
        assert body["model"] == "muse-wonder-1.0"
        part = body["messages"][0]["content"][1]
        assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/jpeg;base64,")
        return httpx.Response(200, json=completion(json.dumps(cat), model="muse-wonder-1.0"))

    out = read_menu_photo(Router(make_client(handler)), b"\xff\xd8fakejpeg")
    assert out.summary() == "2 items · ₹240–₹1,800 · 11am-8pm, closed Monday"
