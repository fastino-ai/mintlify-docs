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
- Use only customer-authored HTTP operations present in that OpenAPI specification.
- Use [https://docs.fastino.ai/concepts/models](https://docs.fastino.ai/concepts/models) for documented model IDs and `GET /v1/base-models` for current availability.
- For reinforcement learning and custom post-training, follow the
  [Training API](https://docs.fastino.ai/training) custom-loop guidance and
  access the Training API through the documented Tinker SDK. The SDK's HTTP
  interface stays out of OpenAPI and is not a customer-authored REST API.
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
- Use `/v1/training/jobs` for canonical job operations. Access custom-loop
  model operations through the supported Tinker SDK.
  Auto Train is private access and has no public route. Direct interested
  users to `sales@fastino.ai`.
- For custom loops, start with the
  [Training quickstart](https://docs.fastino.ai/training/quickstart#custom-training-loop).
  Use the focused guides for migration, verifiable rewards, and recovery.
- Treat every workload as a job under `GET /v1/training/jobs`. Config jobs let
  Fastino run the loop; interactive jobs let the application direct each step.
- Query `ServiceClient.get_server_capabilities()` before selecting a model for
  a custom loop.
- Do not construct the SDK's HTTP requests directly. Configure the supported
  Tinker SDK with
  `TINKER_BASE_URL=https://api.fastino.ai/v1/training/compat/tinker` and
  `TINKER_CREDENTIAL_CMD` that prints the normal `fast_sk_...` credential.
- Never wrap a Fastino key with `tml-`; `tml-` credentials are rejected.
- The SDK training run ID is the interactive Fastino job ID. Use that same ID
  with documented canonical job reads.
- Do not invent routes, model IDs, request fields, or response fields.
- Use [Migrate from Tinker](https://docs.fastino.ai/training/migrate-from-tinker#supported-client-behavior)
  for client support boundaries. The pinned `tinker==0.33.1` package defines
  method signatures, but Fastino's documented unsupported list takes
  precedence. Use [Training loop](https://docs.fastino.ai/training/training-loop)
  and [Checkpoints and recovery](https://docs.fastino.ai/training/checkpoints)
  for operation and recovery behavior.
