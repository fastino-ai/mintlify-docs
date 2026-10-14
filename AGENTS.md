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
- For reinforcement learning and custom post-training of decoder models,
  follow the [Training API](https://docs.fastino.ai/training) custom-loop
  guidance and call the interactive-job routes in the OpenAPI specification.

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
- Use `POST /v1/training/jobs` (config jobs) to fine-tune GLiNER encoders:
  `fastino/gliner2-multi-v1` or `fastino/GLiNER-2.5-Decide`.
- Use `POST /v1/training/jobs/create-model` (interactive jobs) to train decoder
  models: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`,
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`, or `Qwen/Qwen3.6-35B-A3B`.
  Don't send a decoder base model to `POST /v1/training/jobs`.
- Drive an interactive job with `/v1/training/jobs/{job_id}/forward-backward`,
  `/optim-step`, `/save-weights`, `/save-weights-for-sampler`, `/load-weights`,
  and `/sample`. Each returns a `future_id`; read it with
  `GET /v1/training/jobs/{job_id}/futures/{future_id}?wait=true`.
- Tokenize client-side with the base model's Hugging Face tokenizer. Datums
  are JSON: `model_input.chunks[].tokens` and `loss_fn_inputs` tensors.
- The server orders each job's calls. Don't send sequence numbers or
  heartbeats. A job with no calls for 30 minutes is released; checkpoints are
  kept. Call `POST /v1/training/jobs/{job_id}/stop` when done.
- Auto Train is private access and has no public route. Direct interested
  users to `sales@fastino.ai`.
- For custom loops, start with the
  [Training quickstart](https://docs.fastino.ai/training/quickstart#custom-training-loop).
  Use the focused guides for verifiable rewards and recovery.
- Treat every workload as a job under `GET /v1/training/jobs`. Config jobs let
  Fastino run the loop; interactive jobs let the application direct each step.
- For an existing Tinker SDK workload, follow
  [Migrate from Tinker](https://docs.fastino.ai/training/migrate-from-tinker).
- Do not invent routes, model IDs, request fields, or response fields.
- Use [Training loop](https://docs.fastino.ai/training/training-loop)
  and [Checkpoints and recovery](https://docs.fastino.ai/training/checkpoints)
  for operation and recovery behavior.
