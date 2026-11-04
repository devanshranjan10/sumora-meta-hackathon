# Sumora architecture (public summary)

## Product flow

1. **`/start` onboarding.**
   - The owner describes the business in one sentence.
   - A Muse Spark interviewer asks only what's missing: services, prices, hours, area, how customers pay.
   - The owner can drop in past material such as a menu or price list. Sumora extracts the facts into the business's knowledge graph.
2. **Team design.** The planner role (Muse Spark, strict JSON-schema output) picks the agents this business needs. For each agent it sets the tools and the actions that need owner approval.
3. **Build cascade.**
   - Agents build a private preview of the business's website from the extracted facts.
   - Muse Image paints the hero.
   - When the owner approves, the business "is open": the site is published, the WhatsApp front desk answers, and bookings and payments go live.
4. **Command.** The owner's single inbox shows:
   - agent activity rows, each with the model that made the decision;
   - approvals;
   - a daily brief.
   The same controls work from WhatsApp: the brief arrives in the morning, the owner approves or declines in chat, and can send commands by text or voice note (Muse Voice Transcribe).

## Runtime

- **Durable agent runs.** Every agent run is a Temporal workflow. Retries, timeouts and schedules (the morning brief, follow-ups) survive restarts.
- **Approvals.** An approval is a workflow signal. A run that hits a risky tool parks until the owner decides, so nothing is lost if the owner answers hours later.
- **Model router.** Each role (interview, planner, command, agent, brief) has a model chain, with Muse Spark 1.3 first. Fallbacks trigger only on 5xx, 408, 409 or 429 errors. Every call writes a run record: model, latency, prompt, completion and reasoning tokens. These records feed the UI ("Runs on Meta Muse Spark") and cost accounting.
- **Governance.**
  - Per-action rules decide what needs approval (always, over an amount, or never).
  - An approval gate sits in front of every tool with real-world effect.
  - Internal APIs are HMAC-signed.
  - Every model decision is recorded in an audit log.
- **Tenancy.** Each business gets an isolated workspace in Postgres (Neon). Agents' tools are scoped to that workspace.
- **Edge.** The web app (Next.js) runs on Cloudflare Workers. The backend (FastAPI, Temporal worker, MCP tool server) runs in Cloudflare Containers.
- **Channels.**
  - WhatsApp Business Platform (Cloud API): webhooks, approved utility templates, Embedded Signup for businesses' own numbers;
  - email through Resend;
  - Razorpay partner payments for India (in progress).

## What stays private

The onboarding interviewer and prompts, Temporal workflows, channel adapters, billing and the data model live in Sumora's private monorepo. This repository holds the Meta Model API layer they all call.
