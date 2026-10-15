---
name: fastino-fine-tune
description: Fine-tune a GLiNER model on Fastino from a labeled dataset, then monitor training until the model is ready for inference.
---

# Fastino Fine-Tune

Run managed config training from an uploaded encoder or decoder dataset.
Decoder config jobs support LoRA with SFT, DPO, or GRPO. Query
`GET /v1/base-models?supports_training=true` before selecting a base model.

When the application must control decoder training step by step, create an
interactive job with `POST /v1/training/jobs/create-model` and follow
https://docs.fastino.ai/training/quickstart#custom-training-loop.

## Authentication

Fastino API keys start with `fast_sk_` (`pio_sk_` keys were retired on the move to Fastino). Both header formats are accepted:

```
X-API-Key: YOUR_API_KEY
```
```
Authorization: Bearer YOUR_API_KEY
```

## Recommended Order

1. Create or confirm your dataset exists (`GET /v1/datasets/:name`)
2. Start training with that exact dataset name
3. Poll `is_terminal_status` and inspect `normalized_status`
4. Deploy an eligible checkpoint before using the job ID for inference

## Start a Training Job

Use `base_model` with a trainable model ID from
`GET /v1/base-models?supports_training=true` or a supported checkpoint UUID.
`datasets` is an array — supports multi-dataset training.

```http
POST https://api.fastino.ai/v1/training/jobs
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "model_name": "my-ner-model",
  "base_model": "fastino/gliner2-multi-v1",
  "datasets": [{"name": "YOUR_DATASET_NAME"}],
  "training_type": "lora",
  "nr_epochs": 5,
  "learning_rate": 5e-5
}
```

## Key Response Fields

`POST /v1/training/jobs` returns:
- `id` — the canonical training job ID
- `status` — current job status (see values below)
- `normalized_status` — stable polling status
- `is_terminal_status` — whether polling can stop

## Poll Training Status

```http
GET https://api.fastino.ai/v1/training/jobs/YOUR_TRAINING_JOB_ID
X-API-Key: YOUR_API_KEY
```

| Status | Meaning |
|--------|---------|
| `requested` | Queued, waiting for GPU |
| `running` | GPU training in progress |
| `complete` | GPU training finished, post-processing starting |
| `normalizing` | Artifacts being normalized and deployed to inference provider |
| `deployed` | Ready for inference — use the job `id` as `model_id` |
| `errored` | Persisted training failure — check logs |
| `stopped` | Manually stopped |
| `terminated` | Force-terminated |

Poll until `is_terminal_status` is true. A persisted `errored` status is exposed
as normalized `failed`.

## Stream Training Logs

```http
GET https://api.fastino.ai/v1/training/jobs/YOUR_TRAINING_JOB_ID/logs
X-API-Key: YOUR_API_KEY
```

## Other Useful Endpoints

```http
# List all jobs
GET https://api.fastino.ai/v1/training/jobs

# List checkpoints
GET https://api.fastino.ai/v1/training/jobs/YOUR_TRAINING_JOB_ID/checkpoints

# Download trained model
GET https://api.fastino.ai/v1/training/jobs/YOUR_TRAINING_JOB_ID/download
```

## Stopping vs Terminating a Job

These are two distinct operations with different consequences:

| Action | Endpoint | Effect | Reversible? |
|--------|----------|--------|-------------|
| Stop | `POST /v1/training/jobs/{id}/stop` | Graceful stop; all checkpoints preserved; status → `stopped` | Yes — checkpoints remain usable |
| Terminate | `POST /v1/training/jobs/{id}/terminate` | Force-stops the job; checkpoints are preserved unless `delete_checkpoints: true` | Destructive only when deletion is requested |

```http
# Stop gracefully (checkpoints preserved)
POST https://api.fastino.ai/v1/training/jobs/YOUR_TRAINING_JOB_ID/stop
X-API-Key: YOUR_API_KEY

# Terminate and preserve checkpoints
POST https://api.fastino.ai/v1/training/jobs/YOUR_TRAINING_JOB_ID/terminate
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{"delete_checkpoints": false}
```

Set `delete_checkpoints` to `true` only when permanent artifact deletion is
intended.

Either call may return **202** while the provider confirms the cancel. Poll
`GET /v1/training/jobs/{id}` instead of calling again. Termination deletes
checkpoints only when `delete_checkpoints: true`, and only after cancellation
is confirmed. A **409** with `error.code` `training_job_already_finished` means
the job completed or failed first.

## After Training

Once `deployed`, use the `id` from the response as `model` in `/v1/chat/completions`
or `POST /v1/gliner-2` for encoder tasks. See the `fastino-inference` skill.

## Error Codes

| Code | Meaning |
|------|---------|
| 401 | Invalid or missing API key |
| 402 | Insufficient credits |
| 404 | Resource not found |
| 409 | `duplicate_training_submission` — an identical job was created moments ago and is still active; poll the `job_id` in the error instead of resubmitting. `idempotency_key_job_deleted` — the job this `Idempotency-Key` created was deleted; send a new key to launch it again |
| 422 | Validation error — check request body fields; `idempotency_key_reused` means the `Idempotency-Key` was already used for a different body |
| 500 | Server error — retry `POST /v1/training/jobs` with the same `Idempotency-Key` |

Send an `Idempotency-Key` header (a fresh UUID per job you mean to create) on
`POST /v1/training/jobs` and reuse it on every retry of that create: a repeat
returns the original job instead of launching, and billing, a second run.
