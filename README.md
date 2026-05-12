# RAG Qdrant API

## Index documents

Put PDF files in `docs/`, then index them into Qdrant:

```bash
uv run python main.py
```

## Run the API locally

```powershell
$env:PYTHONUTF8="1"
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open:

- API docs: http://localhost:8000/docs
- LangServe playground: http://localhost:8000/rag/playground
- Health check: http://localhost:8000/health

The `/rag` endpoint is backed by a LangGraph workflow:

```text
question -> rewrite_question -> retrieve_documents -> grade_context
                                      -> generate_answer
                                      -> answer_not_found
```

Invoke with PowerShell:

```powershell
$body = @{ input = "bảng quy đổi điểm đánh giá sinh viên" } | ConvertTo-Json
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/rag/invoke" `
  -ContentType "application/json; charset=utf-8" `
  -Body $bytes
```

Required environment variables:

```env
OPENAI_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=rag-qdrant-src
```

Optional:

```env
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.2
RAG_TOP_K=10
ALLOWED_ORIGINS=http://localhost:3000
```

## Docker

```bash
docker build -t rag-qdrant-api .
docker run --env-file .env -p 8000:8000 rag-qdrant-api
```
