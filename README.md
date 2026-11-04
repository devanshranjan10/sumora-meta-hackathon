# Sumora × Meta Muse

**Sumora gives a small business a working team of AI agents from one conversation.** The team runs the website, bookings, WhatsApp and payments. It asks the owner before anything risky. Every agent thinks with **Meta Muse Spark**.

Submitted to the **Meta Global AI Developer Hackathon 2026**: Startups Track, "Build for Your World".

- **Live product (staging):** https://staging-app.trysumora.com
- **Company:** https://trysumora.com

> This repository is the open part of Sumora: the Meta Model API integration layer (model routing, Muse Image, Muse Voice), its tests, and the architecture notes. The full platform (agents, workflows, channels, billing) is a private monorepo.

## How Sumora uses Meta AI

| Meta product | Where it runs in Sumora |
|---|---|
| **Muse Spark 1.3** (Meta Model API) | The default brain for every agent role: the onboarding interview, team planning, owner Command chat, tool-calling agent runs (bookings, follow-ups, site edits) and the daily brief. |
| **Muse Image 1.0** | Hero and cover images for each business's website, generated during the build cascade. |
| **Muse Voice Transcribe 1.0** | Owner WhatsApp voice notes become commands. |
| **WhatsApp Business Platform** | Sumora's sender: owner verification, daily briefs, and approve/decline from chat. |

## Architecture (short)

```
owner ── web (/start, Command)  ─┐
      └─ WhatsApp (Cloud API)  ──┤
                                 ▼
                     FastAPI exec + MCP tools
                                 │  durable runs, schedules, approvals as signals
                                 ▼
                          Temporal workers
                                 │  per-role model routing, fallbacks, cost ledger
                                 ▼
                 Model router ──► Meta Model API (Muse Spark / Image / Voice)
                                 │
                       Postgres (tenant-isolated workspaces)
```

Code lands here as the integration is merged. See `docs/` and `meta/`.
