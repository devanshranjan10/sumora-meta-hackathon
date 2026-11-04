"""Owner material -> structured catalogue, with Muse Wonder's image understanding.

A home baker doesn't fill in forms. They photograph the menu taped to the
fridge and send it on WhatsApp. One multimodal call turns that photo into the
typed facts the agents and the website use: items, prices, hours, area,
payment methods and lead time.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from .router import Router

CATALOG_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "catalogue",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["business_name", "items", "opening_or_pickup_hours", "closed_days", "area", "payment", "lead_time_days"],
            "properties": {
                "business_name": {"type": ["string", "null"]},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "price_inr", "unit", "price_is_starting"],
                        "properties": {
                            "name": {"type": "string"},
                            "price_inr": {"type": ["number", "null"]},
                            "unit": {"type": ["string", "null"]},
                            "price_is_starting": {"type": "boolean", "description": "true for 'from ₹X' prices"},
                        },
                    },
                },
                "opening_or_pickup_hours": {"type": ["string", "null"], "description": "e.g. 'Pickup 11am-8pm'"},
                "closed_days": {"type": "array", "items": {"type": "string"}},
                "area": {"type": ["string", "null"]},
                "payment": {"type": "array", "items": {"type": "string"}},
                "lead_time_days": {"type": ["integer", "null"]},
            },
        },
    },
}


@dataclass
class Catalogue:
    business_name: str | None
    items: list[dict]
    opening_or_pickup_hours: str | None
    closed_days: list[str]
    area: str | None
    payment: list[str]
    lead_time_days: int | None

    def summary(self) -> str:
        prices = [i["price_inr"] for i in self.items if i.get("price_inr")]
        span = f"₹{min(prices):,.0f}–₹{max(prices):,.0f}" if prices else "no prices"
        closed = f", closed {', '.join(self.closed_days)}" if self.closed_days else ""
        return f"{len(self.items)} items · {span} · {self.opening_or_pickup_hours or 'hours unknown'}{closed}"


def read_menu_photo(router: Router, image: bytes, mime: str = "image/jpeg") -> Catalogue:
    url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
    result = router.chat(
        "vision",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is a small business owner's menu or price list. Extract it exactly; "
                                             "never invent items or prices. Hours may be written as opening, pickup or delivery "
                                             "hours. Use null only when something is truly not shown."},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        response_format=CATALOG_SCHEMA,
        reasoning_effort="medium",  # "low" dropped the hours line in 1 of 2 live runs
    )
    return Catalogue(**result.json())
