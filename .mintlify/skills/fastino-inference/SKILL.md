---
name: fastino-inference
description: Run NER, classification, or generation inference on a Fastino model using GLiNER-2 or the OpenAI/Anthropic-compatible endpoints.
---

# Fastino Inference

Run inference against a base GLiNER model or a trained GLiNER/decoder model on Fastino.

## Authentication

All requests require your Fastino API key (starts with `fast_sk_`; `pio_sk_` keys were retired on the move to Fastino).
Both header formats are accepted:

```
X-API-Key: YOUR_API_KEY
```
```
Authorization: Bearer YOUR_API_KEY
```

## GLiNER-2 Endpoint

Best for direct NER, classification, and structured extraction with the
hosted base GLiNER model. This endpoint does not select fine-tuned models;
use the OpenAI-compatible endpoint below with the training-job UUID.

```http
POST https://api.fastino.ai/v1/gliner-2
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "text": "Apple announced the MacBook Pro.",
  "schema": {
    "entities": [{"name": "organization"}, {"name": "product"}]
  }
}
```

**Recommended schema shape (unified):**

- Entity extraction: `{"entities": [{"name": "organization"}, {"name": "product"}]}`
- Classification: `{"classifications": [{"task": "category", "labels": ["spam", "ham"], "multi_label": false}]}`. Every classification task requires at least two labels.
- Structured extraction: `{"structures": {"Person": {"fields": [{"name": "name", "dtype": "str"}]}}}`

**Breaking change:** classification schemas with zero or one label now return
HTTP 400, including entries with `"multi_label": true`. Represent a binary
detector with two explicit labels rather than a one-label task.

**Legacy request shape (deprecated):**

The flat `task` + `list[str]`/`{"categories": [...]}` form is still
accepted on `/v1/gliner-2`, but every legacy submission returns the unified
payload along with `Deprecation: true` and a `Sunset: <RFC 7231 date>` response
header. Migrate clients to the unified schema dict above before the
sunset date to avoid breaking changes.

## OpenAI-Compatible Endpoint

Drop-in replacement for the OpenAI SDK. Set `base_url` to `https://api.fastino.ai/v1`.
Pass `schema` via `extra_body` for NER/classification tasks.

```http
POST https://api.fastino.ai/v1/chat/completions
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "model": "YOUR_TRAINING_JOB_ID",
  "messages": [{"role": "user", "content": "Extract entities from: Apple launched the iPhone."}],
  "schema": {"entities": [{"name": "organization"}, {"name": "product"}]}
}
```

`task_type` and list-form `schema` are still accepted on the
OpenAI/Anthropic/Responses compat routes too, but trigger the same
`Deprecation` + `Sunset` response headers.

### GLiNER-2.5-Decide defaults

`fastino/GLiNER-2.5-Decide` can run without a supplied schema. Fastino then
uses the catalog default classification task:

```json
{"classifications": [{"task": "intent", "labels": ["book", "cancel", "change", "status"]}]}
```

Use this default only for a quick trial. For custom intent names or
reproducible production behavior, send an explicit classification schema;
each task must contain at least two labels. When Fastino supplies the default,
the response includes:

```json
{"x_fastino": {"default_schema_applied": true}}
```

In a stream, `x_fastino` arrives in a separate metadata frame after the frame
carrying `finish_reason` and before the optional usage frame and `[DONE]`.

## OpenAI Responses API Endpoint

For agents using the OpenAI Responses API format (`/v1/responses`):

```http
POST https://api.fastino.ai/v1/responses
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "model": "YOUR_TRAINING_JOB_ID",
  "input": "Extract entities from: Apple launched the iPhone."
}
```

## Anthropic-Compatible Endpoint

Drop-in replacement for the Anthropic SDK. Set `base_url` to `https://api.fastino.ai/v1`.

```http
POST https://api.fastino.ai/v1/messages
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "model": "YOUR_TRAINING_JOB_ID",
  "max_tokens": 1024,
  "messages": [{"role": "user", "content": "Extract entities from: Apple launched the iPhone."}],
  "schema": ["organization", "product"]
}
```

## List Available Models

```http
GET https://api.fastino.ai/v1/base-models
X-API-Key: YOUR_API_KEY
```

Filterable by `training`, `inference`, and `task_type`. For compatible routes,
use a catalog model ID as `model`. To call a fine-tuned model, use its completed
training-job UUID as `model` on the OpenAI-compatible endpoint.

`supports_inference` is model-level availability. For encoders, hosted features
are `encoder_features` on each row (null on decoders). A feature missing from
that list is not hosted, even when the upstream model card documents it.
`fastino/gliner2.5-multi-v1` hosts `records` (structure `mode`, `anchor`,
`occurrence_policy`, and field `cardinality`), `span_attributes`
(`entity_attributes` on extracted spans), `constrained_classification`
(`version`, `tasks`, and `constraints`), and `joint_ie` (relation
`head` / `tail` options, `constraints` without `tasks` or `version`, and
entity configs limited to `description`, `threshold`, `candidate_threshold`,
`max_candidates`, and `allow_nested`).

## Error Codes

| Code | Meaning |
|------|---------|
| 401 | Invalid or missing API key |
| 402 | Insufficient credits |
| 404 | Resource not found |
| 400 | Invalid GLiNER schema or model-aware inference input |
| 422 | Request body does not match the endpoint schema |
| 500 | Server error — safe to retry |
