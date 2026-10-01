---
name: fastino-glide
description: Use Fastino's hosted GLiDE decision model to classify, route, and score structured decisions via noul/choice/score questions, and integrate inference into a project.
---

# GLiDE with Fastino

Help the user build structured decision features — classification, routing, and scoring — using Fastino's hosted GLiDE inference API. Use the hosted service.

## API configuration

| Setting | Value |
| --- | --- |
| API base URL | `https://api.fastino.ai` |
| Documentation / OpenAPI URL | `https://docs.fastino.ai/openapi.json` |
| API-key environment variable name | `FASTINO_API_KEY` |
| Authentication header and value format | `X-API-Key: $FASTINO_API_KEY` |
| Model catalog operation | `GET /v1/base-models` |
| Inference operation | `POST /v1/systemone` |

Treat the OpenAPI document as authoritative. Read credentials from `FASTINO_API_KEY`; never embed them in source, logs, or reports. The API also accepts a Fastino API key as `Authorization: Bearer $FASTINO_API_KEY`, but use one authentication style consistently.

## Request shape

```json
{
  "model": "fastino/glide",
  "state": "<text, object, or array — required>",
  "questions": {
    "<question_key>": {
      "type": "noul | choice | score",
      "instructions": "<natural language question>",
      "criteria": { "...": "..." }
    }
  }
}
```

- `state`, `questions`, and `model` are all required — omitting `model` returns `422 'model' must be provided`.
- Any other top-level request key returns `422 Extra inputs are not permitted`.
- Limits: up to 255 options per `choice` question; each question's prompt (state + that question) must fit in ~40k tokens. `usage.input_tokens` isn't a flat cap — it sums multiple internal passes per question.

## Question types

- **noul** — binary yes/no decision. `criteria`: `{"true": "description", "false": "description"}`.
- **choice** — pick one label from up to 255 options. `criteria`: `{"option_key": "description", ...}`.
- **score** — pick a discrete level on an ordered scale. `criteria`: `["level 0 description", "level 1 description", ...]` (index 0 = lowest level).

## Response shape

Top level:
```json
{
  "model": "glide",
  "answers": { "<question_key>": { ... } },
  "usage": { "input_tokens": 89, "output_tokens": 1 },
  "token_usage": 90
}
```
- `model` echoes back with the `fastino/` prefix stripped.
- `token_usage` = `input_tokens + output_tokens`.

`answers.<key>` by type:

Noul:
```json
{ "type": "noul", "noul": 0.99767683794902, "confidence": 0.9953536758980399 }
```
`noul` is the probability of true; `confidence` is `|2*noul - 1|`. There is no label field — threshold `noul` yourself.

Choice:
```json
{
  "type": "choice",
  "choice": "returns",
  "confidence": 0.9992611288391886,
  "probabilities": { "billing": 0.00031500429476013744, "returns": 0.9995761331339488, "shipping": 0.00010886257129113499 }
}
```
`probabilities` is keyed by the criteria keys and sums to ~1; `confidence` is `top1 - top2`.

Score — request question:
```json
{
  "model": "fastino/glide",
  "state": "<your context>",
  "questions": {
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": ["low urgency, can wait", "medium urgency, handle soon", "high urgency, handle immediately"]
    }
  }
}
```
Score — matching `answers.urgency`:
```json
{
  "type": "score",
  "score": 2,
  "expected_level": 1.99,
  "confidence": 0.9925,
  "probabilities": { "0": 0.0035, "1": 0.0004, "2": 0.996 },
  "legend": { "0": "low urgency, can wait", "1": "medium urgency, handle soon", "2": "high urgency, handle immediately" }
}
```
`score` is the argmax level index (int); `expected_level` is `Σ(level × probability)`; `confidence` is `top1 - top2`, same as noul/choice. `legend` isn't returned by the model — the API layer builds it from your own `criteria`, indexed.

## End-to-end example

```bash
curl -s https://api.fastino.ai/v1/systemone \
  -H "X-API-Key: $FASTINO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fastino/glide",
    "state": "Refund request: the receipt is attached, the purchase was 10 days ago, and refunds are allowed within 30 days.",
    "questions": {
      "refund_allowed": {
        "type": "noul",
        "instructions": "Does this request qualify for a refund?",
        "criteria": { "true": "Qualifies", "false": "Does not qualify" }
      }
    }
  }'
```

Response:
```json
{
  "model": "glide",
  "answers": {
    "refund_allowed": { "type": "noul", "noul": 0.99767683794902, "confidence": 0.9953536758980399 }
  },
  "usage": { "input_tokens": 89, "output_tokens": 1 },
  "token_usage": 90
}
```

## Integrate inference

- Send `state` and `questions` to `POST /v1/systemone`; response is `{model, answers, usage}`.
- Verify the response's `model` field matches what you expect — it echoes back with the `fastino/` prefix stripped.
- No multi-label primitive — one answer per question; use multiple independent questions instead. `probabilities` are already normalized per question — do not reinterpret or renormalize them.
- Tune thresholds (e.g. `noul`) on real data, not a hardcoded 0.5. Validate the chosen `choice`/`score` before acting on it.
- No documented chunking/batching contract for `/v1/systemone` — see Limits above for the token ceiling.
- Add timeouts and bounded backoff for transient errors and rate limits. Surface authentication and validation errors (e.g. `422 Extra inputs are not permitted`) clearly; avoid retrying them indefinitely.

## Note

- `score` criteria also accepts a dict with numeric-string keys (e.g. `{"1":..., "3":..., "5":...}`); keys are sorted then discarded — response is always re-indexed `0..N-1`.
- Migrating from TypeSafe (Jev): map its `score` (float, probability-weighted average) to GLiDE's `expected_level`, not GLiDE's `score` (int argmax index); cast GLiDE's string-keyed `probabilities`/`legend` to int if your code expects TypeSafe's `dict[int, float]` shape.
- For the complete TypeSafe/Jev migration contract, refer to https://docs.fastino.ai/inference/systemone#migrating-from-typesafe.

## Handoff

Provide working request/response code and the required environment-variable name. Never claim a field/value exists without having seen it in a live response.
