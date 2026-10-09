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
- The RL API is a limited-access preview. Do not assume that a team has access;
  verify capabilities before planning or running a workload.

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
- For RL workloads, start with the [RL API overview](https://docs.fastino.ai/rl-api)
  and [quickstart](https://docs.fastino.ai/rl-api/quickstart). Use the focused
  guides for SDK migration, verifiable rewards, and checkpointing.
- Treat every workload as a job under `GET /v1/training/jobs`. Config jobs let
  Fastino run the loop; interactive jobs let the application direct each step.
- Query `ServiceClient.get_server_capabilities()` before selecting an RL model.
- Do not construct RL API HTTP routes directly. Configure the `tinker` SDK with
  `TINKER_BASE_URL=https://api.fastino.ai/v1/training/compat/tinker` and
  `TINKER_CREDENTIAL_CMD` that prints the normal `fast_sk_...` credential.
- Never wrap a Fastino key with `tml-`; `tml-` credentials are rejected.
- The SDK training run ID is the interactive Fastino job ID. Use that same ID
  with the canonical job and checkpoint administration routes.
- Do not invent routes, model IDs, request fields, or response fields.
- For RL workloads, use Fastino's
  [SDK compatibility](https://docs.fastino.ai/rl-api/compatibility) page
  for Fastino-specific support boundaries. The pinned `tinker==0.33.1` package
  defines method signatures; an explicitly unsupported item on Fastino's page
  overrides its availability in the client package. Use
  [runtime behavior and errors](https://docs.fastino.ai/rl-api/runtime) for
  operation, checkpoint, recovery, and error behavior.
