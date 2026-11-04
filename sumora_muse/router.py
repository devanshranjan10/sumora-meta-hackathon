"""Per-role model routing with fallbacks, the way Sumora picks a brain for each job.

Each agent role gets an ordered chain of models. The first one is the primary
(Muse Spark for every role in Sumora); later entries are only tried when the
primary fails with a server-side or rate-limit error. Client errors (4xx other
than 408/409/429) are not retried on another model: they mean the request is wrong.
Every call is recorded so the product can show which model made each decision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .client import ChatResult, MuseClient, MuseError

ROLES = ("interview", "planner", "command", "agent", "brief")

DEFAULT_CHAINS: dict[str, list[str]] = {
    # The planner designs a business's whole agent team: worth the strongest model.
    "planner": ["muse-spark-1.3", "muse-spark-1.2"],
    "interview": ["muse-spark-1.3", "muse-spark-1.2"],
    "command": ["muse-spark-1.3", "muse-spark-1.2"],
    "agent": ["muse-spark-1.3", "muse-spark-1.2"],
    "brief": ["muse-spark-1.3", "muse-spark-1.2"],
}


@dataclass
class RunRecord:
    role: str
    model: str
    ok: bool
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    error: str | None = None


@dataclass
class Router:
    client: MuseClient
    chains: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_CHAINS.items()})
    on_record: Callable[[RunRecord], None] | None = None
    records: list[RunRecord] = field(default_factory=list)

    def chat(self, role: str, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        chain = self.chains.get(role)
        if not chain:
            raise ValueError(f"unknown role {role!r}; known: {sorted(self.chains)}")
        last: MuseError | None = None
        for model in chain:
            started = time.monotonic()
            try:
                result = self.client.chat(messages, model=model, **kwargs)
            except MuseError as exc:
                self._record(RunRecord(role, model, False, _ms(started), error=str(exc)))
                last = exc
                if exc.status is not None and exc.status < 500 and exc.status not in (408, 409, 429):
                    raise
                continue
            usage = result.usage
            self._record(
                RunRecord(
                    role,
                    result.model,
                    True,
                    _ms(started),
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                    result.reasoning_tokens,
                )
            )
            return result
        raise MuseError(f"all models failed for role {role!r}") from last

    def _record(self, rec: RunRecord) -> None:
        self.records.append(rec)
        if self.on_record:
            self.on_record(rec)


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
