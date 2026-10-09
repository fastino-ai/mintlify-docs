---
name: fastino-datasets
description: List, inspect, upload, and delete labeled datasets on Fastino, then pass a dataset name to a fine-tuning job.
---

# Fastino Datasets

Manage labeled datasets on Fastino. Datasets are referenced by name when starting training jobs.

## Authentication

Fastino API keys start with `fast_sk_` (`pio_sk_` keys were retired on the move to Fastino). Both header formats are accepted:

```
X-API-Key: YOUR_API_KEY
```
```
Authorization: Bearer YOUR_API_KEY
```

## List Datasets

```http
GET https://api.fastino.ai/v1/datasets
X-API-Key: YOUR_API_KEY
```

Optional query parameters: `task_type` (`ner`, `classification`, `decoder`), `project_id`.

## Get a Dataset (Check It Exists)

Returns all versions of the dataset:

```http
GET https://api.fastino.ai/v1/datasets/YOUR_DATASET_NAME
X-API-Key: YOUR_API_KEY
```

Returns 404 if the dataset does not exist. Use this to confirm a dataset is ready before starting training.

## Dataset Row Format

Each line of your JSONL must be a single JSON object. The required columns depend on the dataset type passed as `dataset_type` on `POST /v1/datasets/upload/url`.

**NER (`dataset_type: "ner"`)**

- `text` (string) — input text
- `entities` (list of `[span, label]` pairs) — each `span` must be a substring of `text`. Use `[]` for negative examples (rows that contain no entities).

```jsonl
{"text": "Ada Lovelace worked in London.", "entities": [["Ada Lovelace", "PERSON"], ["London", "LOCATION"]]}
{"text": "No tagged entities here.", "entities": []}
```

**Classification (`dataset_type: "classification"`)**

- `text` (string)
- `label` (string) for single-label, OR `labels` (list of strings) for multi-label. Pick one and use it consistently across the entire dataset.

```jsonl
{"text": "Great product", "label": "positive"}
{"text": "Mixed feelings about this", "label": "neutral"}
```

**Decoder (`dataset_type: "decoder"`)**

- `messages` (list of `{role, content}` objects) — chat-style supervision. `role` must be one of `system`, `user`, `assistant`.

```jsonl
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

> **Fastino does not accept the GLiNER2 canonical wire shape on upload.** If your data looks like `{"input": ..., "output": {"entities": {"<label>": ["<span>", ...]}}}` (the shape that GLiNER2 trainers consume internally), flatten it to top-level `text` + `entities` pair list before uploading. Per-label `entity_descriptions` are not stored on dataset rows; configure them at training-job time instead.

## Upload a Dataset

Upload in two steps: get a presigned URL, then process the uploaded file.

**Step 1 — get presigned upload URL:**

```http
POST https://api.fastino.ai/v1/datasets/upload/url
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "dataset_name": "my-dataset",
  "file_name": "data.jsonl"
}
```

Returns `upload_url` (PUT to this with the file body) and `dataset_id`.

**Step 2 — process after upload:**

```http
POST https://api.fastino.ai/v1/datasets/upload/process
X-API-Key: YOUR_API_KEY
Content-Type: application/json

{
  "dataset_id": "DATASET_ID_FROM_STEP_1"
}
```

Returns HTTP 202 immediately. Poll `GET /v1/datasets/{name}` until the version appears.

## Delete an Entire Dataset (All Versions)

```http
DELETE https://api.fastino.ai/v1/datasets/YOUR_DATASET_NAME
X-API-Key: YOUR_API_KEY
```

## Use a Dataset for Training

Pass the dataset name directly to `POST /v1/training/jobs`:

```json
{
  "datasets": [{"name": "YOUR_DATASET_NAME"}]
}
```

See the `fastino-fine-tune` skill for the full training workflow.

## Error Codes

| Code | Meaning |
|------|---------|
| 401 | Invalid or missing API key |
| 402 | Insufficient credits |
| 404 | Dataset not found |
| 422 | Validation error — check request body fields |
| 500 | Server error — safe to retry |
