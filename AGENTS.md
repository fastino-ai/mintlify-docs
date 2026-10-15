---
title: "Fastino API instructions for agents"
hidden: true
noindex: true
---

# Fastino API instructions for agents

Use the published Fastino documentation and canonical public OpenAPI specification when building an integration.

## Sources of truth

- Start with the documentation index at [https://docs.fastino.ai/llms.txt](https://docs.fastino.ai/llms.txt).
- Use [https://docs.fastino.ai/llms-full.txt](https://docs.fastino.ai/llms-full.txt)
  when the complete documentation corpus is needed.
- Read [https://docs.fastino.ai/openapi.json](https://docs.fastino.ai/openapi.json) for customer-facing routes, authentication, request schemas, and response schemas.
- Use only customer-facing HTTP operations present in that OpenAPI specification.
- Use [https://docs.fastino.ai/concepts/models](https://docs.fastino.ai/concepts/models) for documented model IDs and `GET /v1/base-models` for current availability.
- For reinforcement learning and custom post-training, follow the
  [Training API](https://docs.fastino.ai/training) custom-loop guidance.
  Native JSON clients use the public interactive operations in OpenAPI.
  Existing Tinker clients use the documented compatibility base URL.
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
- Follow the [Training API](https://docs.fastino.ai/training) for training-job
  operations. Every workload appears under `GET /v1/training/jobs`: `config`
  jobs let Fastino run the loop, while `interactive` jobs let the application
  direct each step. Check OpenAPI before using an operation because some routes
  support only one job type.
- Create a managed `config` job with `POST /v1/training/jobs` for the GLiNER
  encoders `fastino/gliner2-base-v1` and `fastino/GLiNER-2.5-Decide` only; a
  decoder base model returns `400`. Decoders train only in `interactive` jobs.
  Native JSON clients create or restore an `interactive` job with
  `POST /v1/training/jobs/create-model`, using either `base_model` or
  `from_checkpoint` with optional `load_optimizer`.
- Poll interactive work with
  `GET /v1/training/jobs/{job_id}/futures/{future_id}`. `wait=true` waits for
  at most 25 seconds and can return a nonterminal future. Only queued or
  running sample futures can be cancelled.
- Native JSON clients save, load, and sample through the direct
  `save-weights`, `save-weights-for-sampler`, `load-weights`, and `sample`
  job routes. Load and sample accept a checkpoint name or the same job's
  `tinker://` path. Sampling remains available after stop.
- Interactive checkpoint reads and mutations use
  `/v1/training/jobs/{job_id}/checkpoints/{kind}/{name}`, where `kind` is
  `weights` or `sampler_weights`. Deploy an interactive sampler checkpoint with
  `POST /v1/training/jobs/{job_id}/checkpoints/sampler_weights/{name}/deploy`.
  Config checkpoint download and deployment routes use `checkpoint_id`
  instead. Do not interchange these identifiers.
- Interactive stop acknowledges immediately and rejects new work, but queued
  operations may finish before resources are released. Usage can continue
  until release, and saved checkpoints remain available.
- Use `/v1/training/usage` only for team-scoped interactive checkpoint storage
  and its monthly price; it does not report compute usage or total billing.
  Auto Train is private access and has no public route. Direct interested
  users to `sales@fastino.ai`.
- For custom loops, start with the
  [Training loop](https://docs.fastino.ai/training/training-loop).
  Use the focused guides for verifiable rewards and recovery.
- Tinker clients query `ServiceClient.get_server_capabilities()` before
  selecting a model for a custom loop.
- Configure the supported Tinker SDK with
  `TINKER_BASE_URL=https://api.fastino.ai/v1/training/compat/tinker` and
  `TINKER_CREDENTIAL_CMD` that prints the normal `fast_sk_...` credential.
- Never wrap a Fastino key with `tml-`; `tml-` credentials are rejected.
- The SDK training run ID is the interactive Fastino job ID. Use that same ID
  with documented canonical job reads.
- Do not invent routes, model IDs, request fields, or response fields.
- Use [Migrate from Tinker](https://docs.fastino.ai/training/migrate-from-tinker#tinker-sdk-compatibility)
  for client support boundaries. The pinned `tinker==0.32.0` package defines
  method signatures, but Fastino's documented unsupported list takes
  precedence. Use [Training loop](https://docs.fastino.ai/training/training-loop)
  and [Checkpoints and recovery](https://docs.fastino.ai/training/checkpoints)
  for future and recovery behavior.
