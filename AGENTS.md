---
title: "Fastino API instructions for agents"
hidden: true
noindex: true
---

# Fastino API instructions for agents

Use the published Fastino documentation and canonical public OpenAPI specification when building an integration.

## Sources of truth

- Start with the documentation index at [https://docs.fastino.ai/llms.txt](https://docs.fastino.ai/llms.txt).
- Read [https://docs.fastino.ai/openapi.json](https://docs.fastino.ai/openapi.json) for customer-facing routes, authentication, request schemas, and response schemas.
- Use only operations present in that OpenAPI specification, except for the RL API described below.
- Use [https://docs.fastino.ai/concepts/models](https://docs.fastino.ai/concepts/models) for documented model IDs and `GET /v1/base-models` for current availability.
- For reinforcement learning and custom post-training, use the
  [RL API](https://docs.fastino.ai/rl-api) and the documented `tinker` Python
  SDK contract. RL API routes are intentionally absent from Fastino's OpenAPI.

## API conventions

- API base URL: `https://api.fastino.ai`
- API route prefix: `/v1`
- API-key environment variable: `FASTINO_API_KEY`
- Supported authentication: `X-API-Key: $FASTINO_API_KEY` or `Authorization: Bearer $FASTINO_API_KEY`
- Never embed API keys in source code, logs, examples, or reports.

## Integration guidance

- For GLiDE decisions, read [GLiDE](https://docs.fastino.ai/concepts/glide) and
  [GLiDE Inference](https://docs.fastino.ai/inference/systemone). Use
  `POST /v1/systemone` with `fastino/GLiDE`, `state`, and typed `questions`.
- Install the [GLiDE Agent Skill](https://docs.fastino.ai/concepts/glide-agent-skill)
  for agent-ready GLiDE instructions and examples.
- For GLiNER extraction and classification, read
  [GLiNER Inference](https://docs.fastino.ai/inference/chat-completions). Use
  `POST /v1/chat/completions` with `model`, `messages`, and `schema`.
- Follow the [Training API](https://docs.fastino.ai/training) for training-job operations.
- For agent-controlled RL loops, Tinker migration, and checkpoint operations,
  follow the [RL API](https://docs.fastino.ai/rl-api). Query
  `ServiceClient.get_server_capabilities()` before selecting an RL model.
- Do not construct RL API HTTP routes directly. Configure the `tinker` SDK with
  `TINKER_BASE_URL=https://api.fastino.ai/tinker` and a
  `TINKER_API_KEY=tml-fast_sk_...` credential.
- Do not invent routes, model IDs, request fields, or response fields.
- If an operation is absent from the public OpenAPI specification and the RL
  API compatibility reference, treat it as unsupported for customer
  integrations.
