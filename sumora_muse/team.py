"""Design a small business's agent team from one sentence, using Muse Spark structured output.

This mirrors Sumora's onboarding: the owner describes the business, and the
planner returns a typed team (front desk, bookings, follow-ups, site, brief),
each agent with the tools it may use and the actions that need owner approval.
"""

from __future__ import annotations

from dataclasses import dataclass

from .router import Router

TEAM_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_team",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["business_summary", "timezone", "agents"],
            "properties": {
                "business_summary": {"type": "string"},
                "timezone": {"type": "string", "description": "IANA zone, e.g. Asia/Kolkata"},
                "agents": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 7,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "job", "channels", "tools", "needs_approval_for"],
                        "properties": {
                            "name": {"type": "string"},
                            "job": {"type": "string"},
                            "channels": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["whatsapp", "website", "email", "instagram", "phone"]},
                            },
                            "tools": {"type": "array", "items": {"type": "string"}},
                            "needs_approval_for": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
        },
    },
}

SYSTEM = (
    "You design the AI staff for a very small business. Pick only agents this business needs. "
    "Never include engineering or enterprise roles for a solo shop. Anything that spends money, "
    "messages a new contact, publishes publicly, or refunds must be listed in needs_approval_for."
)


@dataclass
class Agent:
    name: str
    job: str
    channels: list[str]
    tools: list[str]
    needs_approval_for: list[str]


@dataclass
class Team:
    business_summary: str
    timezone: str
    agents: list[Agent]


def plan_team(router: Router, description: str) -> Team:
    result = router.chat(
        "planner",
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": description}],
        response_format=TEAM_SCHEMA,
        reasoning_effort="medium",
    )
    data = result.json()
    return Team(
        business_summary=data["business_summary"],
        timezone=data["timezone"],
        agents=[Agent(**a) for a in data["agents"]],
    )
