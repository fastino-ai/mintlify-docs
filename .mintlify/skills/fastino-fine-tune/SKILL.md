---
name: fastino-fine-tune
description: Fine-tune a GLiNER or decoder model on Fastino from a labeled dataset, then monitor training until the model is ready for inference.
---

# Fastino Fine-Tune

Fine-tune a custom model on Fastino's GPU infrastructure.

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
3. Poll until status is `deployed`
4. Use the `id` field from the training job response as `model_id` for inference

## Start a Training Job

`base_model` is required. Use a model ID from `GET /v1/base-models` or a checkpoint UUID.
`datasets` is an array — supports multi-dataset training.

```http
POST https://api.fastino.ai/v1/training/jobs
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "model_name": "my-ner-model",
  "base_model": "fastino/gliner2-base-v1",
  "datasets": [{"name": "YOUR_DATASET_NAME"}],
  "training_type": "lora",
  "nr_epochs": 5,
  "learning_rate": 5e-5
}
```

## Key Response Fields

`POST /v1/training/jobs` returns:
- `id` — the training job ID; this is your `model_id` for inference
- `status` — current job status (see values below)

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
| `errored` | Training failed — check logs |
| `stopped` | Manually stopped |
| `terminated` | Force-terminated |

Poll until `deployed` before running inference.

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
| Terminate | `POST /v1/training/jobs/{id}/terminate` | Force-stops the job **and deletes all checkpoints and artifacts**; status → `terminated` | **No — irreversible** |

```http
# Stop gracefully (checkpoints preserved)
POST https://api.fastino.ai/v1/training/jobs/YOUR_TRAINING_JOB_ID/stop
X-API-Key: YOUR_API_KEY

# Terminate and delete all artifacts (irreversible)
POST https://api.fastino.ai/v1/training/jobs/YOUR_TRAINING_JOB_ID/terminate
X-API-Key: YOUR_API_KEY
```

Prefer `stop` unless you explicitly want to destroy all artifacts.

Either call may return **202** while the provider confirms the cancel: `stop` returns `status: "stopping"` and the job stays `running` until it flips to `stopped`; `terminate` flips to `terminated` and deletes checkpoints on confirmation. Poll `GET /v1/training/jobs/{id}` instead of calling again. A **409** with `error.code` `training_job_already_finished` means the job completed or failed first.

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
