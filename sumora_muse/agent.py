"""A tool-calling agent loop with an owner-approval gate.

Muse Spark decides which tools to call. Tools marked ``risky`` are not
executed: they become an ``ApprovalRequest`` that Sumora sends to the owner
(on WhatsApp or in Command). The agent is told the action is pending, so it
answers the customer honestly instead of claiming it already happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .router import Router


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    risky: bool = False

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@dataclass
class ApprovalRequest:
    tool: str
    arguments: dict[str, Any]
    reason: str


@dataclass
class AgentRun:
    reply: str
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[ApprovalRequest] = field(default_factory=list)


def run_agent(
    router: Router,
    *,
    instructions: str,
    user_message: str,
    tools: list[Tool],
    max_steps: int = 6,
    role: str = "agent",
) -> AgentRun:
    by_name = {t.name: t for t in tools}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_message},
    ]
    run = AgentRun(reply="")
    for _ in range(max_steps):
        result = router.chat(role, messages, tools=[t.spec() for t in tools], reasoning_effort="low")
        if not result.tool_calls:
            run.reply = (result.content or "").strip()
            return run
        messages.append({"role": "assistant", "content": result.content, "tool_calls": result.tool_calls})
        for call in result.tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            tool = by_name.get(name)
            if tool is None:
                output: Any = {"error": f"unknown tool {name}"}
            elif tool.risky:
                req = ApprovalRequest(name, args, reason=f"{name} needs the owner's OK")
                run.approvals.append(req)
                output = {"status": "pending_owner_approval", "note": "Tell the customer the owner will confirm shortly."}
            else:
                try:
                    output = tool.fn(**args)
                except Exception as exc:  # tool bugs must not crash the conversation
                    output = {"error": str(exc)}
            run.tool_log.append({"tool": name, "args": args, "output": output})
            messages.append({"role": "tool", "tool_call_id": call.get("id", name), "content": json.dumps(output)})
    run.reply = "I need a moment; the owner will follow up."
    return run
