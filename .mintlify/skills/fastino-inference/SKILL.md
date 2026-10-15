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
use the compatible endpoints below with a config training-job UUID or a
deployed interactive checkpoint's `model_id`.

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
- Classification: `{"classifications": [{"task": "category", "labels": ["spam", "ham"], "multi_label": false}]}`
- Structured extraction: `{"structures": {"Person": {"fields": [{"name": "name", "dtype": "str"}]}}}`

**Legacy request shape (deprecated):**

The flat `task` + `list[str]`/`{"categories": [...]}` form is still
accepted on `/v1/gliner-2`, but every legacy submission returns the unified
payload along with `Deprecation: true` and a `Sunset: <RFC 7231 date>` response
header. Migrate clients to the unified schema dict above before the
sunset date to avoid breaking changes.

## OpenAI-Compatible Endpoint

Drop-in replacement for the OpenAI SDK. Set `base_url` to `https://api.fastino.ai/v1`.
For a completed config encoder job, use its training-job UUID as `model` and
pass the GLiNER `schema` via `extra_body`:

```http
POST https://api.fastino.ai/v1/chat/completions
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "model": "YOUR_CONFIG_TRAINING_JOB_ID",
  "messages": [{"role": "user", "content": "Extract entities from: Apple launched the iPhone."}],
  "schema": {"entities": [{"name": "organization"}, {"name": "product"}]}
}
```

Interactive checkpoints are decoder models and do not accept the GLiNER
`schema` field. When the user wants to serve one, first deploy a
`sampler_weights` checkpoint:

```http
POST https://api.fastino.ai/v1/training/jobs/YOUR_INTERACTIVE_JOB_ID/checkpoints/sampler_weights/YOUR_CHECKPOINT_NAME/deploy
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{}
```

Deployment support is model-specific. A `400` means the base model is not
servable yet or the checkpoint is not an accepted adapter; do not poll that
response, and verify model availability and the checkpoint before trying again.
A `409` is terminal for that model name or checkpoint; save the weights under a
new name before deploying again. Use bounded backoff when repeating a successful
deployment request whose status is still `deploying`; inference is ready when
the status is `deployed`.

Use the response's `model_id`—not the interactive job ID—for decoder inference:

```http
POST https://api.fastino.ai/v1/chat/completions
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "model": "YOUR_DEPLOYED_MODEL_ID",
  "messages": [{"role": "user", "content": "Summarize the attached support case."}]
}
```

`task_type` and list-form `schema` are still accepted on the
OpenAI/Anthropic/Responses compat routes too, but trigger the same
`Deprecation` + `Sunset` response headers.

## OpenAI Responses API Endpoint

For agents using the OpenAI Responses API format (`/v1/responses`):

```http
POST https://api.fastino.ai/v1/responses
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "model": "YOUR_MODEL_ID",
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
  "model": "YOUR_MODEL_ID",
  "max_tokens": 1024,
  "messages": [{"role": "user", "content": "Summarize the attached support case."}]
}
```

## List Available Models

```http
GET https://api.fastino.ai/v1/base-models
X-API-Key: YOUR_API_KEY
```

Use the supported query parameters to filter the catalog:

```http
GET https://api.fastino.ai/v1/base-models?supports_training=true
GET https://api.fastino.ai/v1/base-models?supports_inference=true
GET https://api.fastino.ai/v1/base-models?task_type=decoder
```

For compatible routes, use a catalog model ID as `model`. To call a fine-tuned
model, use its completed config training-job UUID or the `model_id` returned
when deploying an interactive job's sampler checkpoint.

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
| 422 | Validation error — check request body fields |
| 500 | Server error — safe to retry |
