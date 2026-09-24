# Fastino API instructions for agents

Use the published Fastino documentation and curated OpenAPI specification when building an integration.

## Sources of truth

- Start with the documentation index at <https://docs.fastino.ai/llms.txt>.
- Read <https://docs.fastino.ai/openapi.json> for customer-facing routes, authentication, request schemas, and response schemas.
- Use only operations present in that OpenAPI specification.
- Use <https://docs.fastino.ai/concepts/models> for documented model IDs and `GET /v1/base-models` for current availability.

## API conventions

- API base URL: `https://api.fastino.ai`
- API route prefix: `/v1`
- API-key environment variable: `FASTINO_API_KEY`
- Supported authentication: `X-API-Key: $FASTINO_API_KEY` or `Authorization: Bearer $FASTINO_API_KEY`
- Never embed API keys in source code, logs, examples, or reports.

## Integration guidance

- Follow the [Inference API](https://docs.fastino.ai/inference) for GLiNER inference requests.
- Follow the [Training API](https://docs.fastino.ai/training) for training-job operations.
- Do not invent routes, model IDs, request fields, or response fields.
- If an operation is absent from the curated OpenAPI specification, treat it as unsupported for customer integrations.
