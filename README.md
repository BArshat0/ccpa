# CCPA Compliance Checker (Open Hack 2026)

This project implements a Retrieval-Augmented Generation (RAG) compliance system for the California Consumer Privacy Act (CCPA), packaged for Docker-based evaluation with a FastAPI API.

It is designed to satisfy the hackathon requirements in `participant_instructions.pdf`, including strict JSON output, startup health checks, and containerized deployment on port `8000`.

## 1. Solution Overview

### Objective

Given a natural-language description of a company practice, the system:

1. Determines whether the practice is a CCPA violation (`harmful=true/false`).
2. Returns the relevant CCPA statute sections in canonical format (`"Section 1798.xxx"`).

### End-to-End Architecture

Input prompt -> Retrieval over CCPA statute PDF -> LLM reasoning constrained by context -> JSON validation and citation grounding -> API response

### Components

- `api.py`
  - FastAPI app and startup lifecycle.
  - `GET /health` and `POST /analyze` endpoints.
  - Strict response formatting logic and readiness handling.
  - Citation post-processing that grounds cited sections in retrieved context.
- `app.py`
  - Vector database creation/loading.
  - PDF loading and chunking.
  - Embedding model and LLM initialization.
  - RAG chain setup.
- `ccpa_statute.pdf`
  - Primary legal source document used for retrieval.
- `validate_format.py`
  - Organizer-style local validation for response format behavior.
- `download_models.py`
  - Optional pre-download utility for model caches.

### Models and Retrieval Stack

- LLM: `Qwen/Qwen2.5-1.5B-Instruct` (within the instructor limit of max 8B parameters).
- Embeddings: `BAAI/bge-large-en-v1.5`.
- Vector Store: `Chroma` persisted in `./chroma_db`.
- Retrieval depth: `k=8`.
- Chunking: `chunk_size=900`, `chunk_overlap=120` (improves section-header + obligation context continuity).

### Citation Accuracy Strategy

To improve legal section precision:

- The model is prompted to cite only sections present in retrieved context.
- Cited sections are normalized to canonical format (`Section 1798.xxx`).
- Returned citations are filtered against sections detected in retrieved chunks.
- If model citations are weak/noisy, a grounded fallback from retrieved sections is applied.

This reduces hallucinated section IDs and aligns with hidden evaluation emphasis on citation correctness.

## 2. API Contract (Strict Format)

### Endpoint: `POST /analyze`

Request body:

```json
{
  "prompt": "We sell customer browsing history to ad networks without notifying them."
}
```

Response body:

```json
{
  "harmful": true,
  "articles": ["Section 1798.100", "Section 1798.120"]
}
```

Rules enforced:

- `harmful` is a JSON boolean (`true`/`false`), not a string.
- `articles` is always a JSON array.
- If `harmful=true`, `articles` must be non-empty.
- If `harmful=false`, `articles` must be exactly `[]`.
- Response body contains only valid JSON (no explanation text, no markdown).

### Endpoint: `GET /health`

- Returns HTTP `200` with `{"status":"ok"}` when startup loading is complete.
- Returns HTTP `503` with `{"status":"loading"}` while models are still initializing.

## 3. Docker Run Command (MANDATORY)

### Required GPU command (as requested by instructors)

```bash
docker run --gpus all -p 8000:8000 -e HF_TOKEN=<token> dev1234de/ccpa-compliance:latest
```

If your model is not gated, you can omit `HF_TOKEN`:

```bash
docker run --gpus all -p 8000:8000 dev1234de/ccpa-compliance:latest
```

### CPU-only fallback command

```bash
docker run -p 8000:8000 dev1234de/ccpa-compliance:latest
```

Recommended persistent volumes:

```bash
docker run --gpus all -p 8000:8000 \
  -v ccpa_chroma_db:/app/chroma_db \
  -v ccpa_hf_cache:/app/.cache/huggingface \
  -e HF_TOKEN=<token> \
  dev1234de/ccpa-compliance:latest
```

## 4. Environment Variables

### Runtime

- `HF_TOKEN`
  - Optional if all models are public.
  - Required if model access is gated on Hugging Face.
  - Must be passed as environment variable; never hardcode.
- `FORCE_REBUILD_DB`
  - Default: `0`.
  - Set `1` to force rebuilding `chroma_db` from `ccpa_statute.pdf` on startup.

### Build-time argument

- `PRELOAD_ARTIFACTS` (Docker build arg)
  - Default: `0`.
  - Set `1` to pre-download models and pre-build vector DB during image build (improves runtime startup reliability).

Example:

```bash
docker build --build-arg PRELOAD_ARTIFACTS=1 -t dev1234de/ccpa-compliance:latest .
```

## 5. GPU Requirements

- Recommended: NVIDIA GPU with approximately 8GB+ VRAM.
- Docker runtime requirements for GPU:
  - NVIDIA drivers installed on host.
  - NVIDIA Container Toolkit configured.
  - `docker run --gpus all ...` supported on host.
- CPU-only fallback is supported, but inference will be slower.

Quick GPU runtime check:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## 6. Local Setup Instructions (Fallback, Linux VM)

These steps are intended for manual fallback if container startup fails.

### Prerequisites

- Linux VM with Python 3.10+ recommended.
- Optional CUDA-capable GPU for faster inference.
- `ccpa_statute.pdf` present in project root.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional model pre-download:

```bash
python download_models.py
```

### Run API server

```bash
python api.py
```

Server listens on:

`http://localhost:8000`

### Manual API check

```bash
curl -s http://localhost:8000/health
```

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt":"We sell customer data without opt-out."}'
```

## 7. API Usage Examples (MANDATORY)

### Example A: Violation

Request:

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt":"We sell customer browsing history to ad networks without notifying users or giving opt-out."}'
```

Expected style of response:

```json
{
  "harmful": true,
  "articles": ["Section 1798.120", "Section 1798.100"]
}
```

### Example B: Compliant

Request:

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt":"We provide clear notice and honor verified deletion requests within the legal timeline."}'
```

Expected style of response:

```json
{
  "harmful": false,
  "articles": []
}
```

### Health check

```bash
curl -s http://localhost:8000/health
```

When ready:

```json
{"status":"ok"}
```

## 8. Docker Build, Test, and Push Workflow

### Option A: Fast local build (downloads at runtime)

```bash
docker build -t dev1234de/ccpa-compliance:latest .
```

### Option B: Production-preloaded build (recommended)

```bash
docker build --build-arg PRELOAD_ARTIFACTS=1 -t dev1234de/ccpa-compliance:latest .
```

### Compose (CPU-safe default)

```bash
docker compose up -d --build
```

### Compose with GPU override

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### Validate locally

```bash
python validate_format.py
```

### Push to Docker Hub

```bash
docker login
docker push dev1234de/ccpa-compliance:latest
```

### Build and push in one step (linux/amd64)

```bash
docker buildx build --platform linux/amd64 --build-arg PRELOAD_ARTIFACTS=1 \
  -t dev1234de/ccpa-compliance:latest --push .
```

## 9. How `validate_format.py` Evaluates (Instructor Summary)

The provided validator checks:

1. Server readiness through `GET /health` for up to 5 minutes.
2. Response schema for `POST /analyze`.
3. Formatting rules:
   - harmful true -> non-empty `articles`
   - harmful false -> empty `articles`

Important:

- `validate_format.py` checks format/contract behavior.
- Final organizer grading also checks correctness and citation accuracy with hidden evaluation logic.

## 10. Submission Checklist (MANDATORY)

Submit all of the following:

1. Public Docker Hub image link:
   - Example: `dev1234de/ccpa-compliance:latest`
2. ZIP archive of complete source code:
   - Include all project files needed to build and run.
   - Do not submit only a GitHub URL.
3. This `README.md` with all required sections.
4. HF token via organizer secure form only if your model is gated.

Pre-submission verification:

1. Build image locally.
2. Run with required docker command.
3. Confirm `GET /health` becomes `200`.
4. Run `validate_format.py`.
5. Confirm no interactive/manual in-container step is required.

## 11. Common Failure Points and Mitigations

- Missing or broken `GET /health`:
  - Mitigation: keep endpoint active and return HTTP 200 only when ready.
- Wrong server port:
  - Mitigation: always bind to `0.0.0.0:8000`.
- Non-boolean `harmful` values:
  - Mitigation: enforce Pydantic schema + post-validation.
- Extra response text beyond JSON:
  - Mitigation: strict prompt constraints + parser fallback + final response shaping.
- Heavy per-request initialization:
  - Mitigation: all model/vector initialization in startup lifespan.
- Runtime model downloads causing slow startup/failure:
  - Mitigation: use `PRELOAD_ARTIFACTS=1` for production builds.
- Hardcoded secrets:
  - Mitigation: use `HF_TOKEN` env var only.
- GPU runtime errors on unsupported hosts:
  - Mitigation: use CPU mode (`docker run -p 8000:8000 ...`) or configure NVIDIA runtime.

## 12. Repository Files

- `api.py` - FastAPI server, startup lifecycle, response validation, grounded citation processing.
- `app.py` - model loading, vector DB build/load, retrieval + prompt setup.
- `download_models.py` - optional cache pre-download utility.
- `validate_format.py` - local format validator.
- `Dockerfile` - production container definition.
- `docker-compose.yml` - CPU-safe default compose profile.
- `docker-compose.gpu.yml` - GPU overlay profile.
- `.env.example` - sample environment configuration.
- `.dockerignore` - optimized Docker build context.

## 13. Notes on Accuracy and Grading

- Primary grading emphasis includes harmful classification correctness and legal section citation accuracy.
- Incorrect section citation on harmful prompts can heavily penalize score.
- This implementation includes citation grounding safeguards to reduce hallucinations, but final score depends on hidden test coverage and legal matching strictness.
