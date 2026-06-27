# rag-agent

[![CI](https://github.com/rovidev95/rovidev-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/rovidev95/rovidev-rag-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11+-3776ab)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/rovidev95/rovidev-rag-agent)

A small retrieval-augmented generation service. Answers are built from your own
documents and come back with inline `[n]` citations pointing at the source
passage. If nothing relevant is found it says so instead of making something up.

It runs without any API key using a hashing-based embedder and an extractive
answerer, which is handy for trying things out and for tests. Swap in OpenAI
embeddings/chat through the optional extra when you want real semantic search.

## Run it

```bash
pip install -e ".[dev]"
python -m examples.ingest_and_ask     # whole flow in the terminal
uvicorn app.main:app --reload         # docs at /docs
```

Ingest some documents and ask a question:

```bash
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{
  "documents": [{
    "id": "faq1",
    "source": "Support FAQ",
    "text": "Refunds are available within 30 days to the original payment method."
  }]
}'

curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{
  "question": "How long do I have to request a refund?", "top_k": 3
}'
```

Response shape:

```json
{
  "answer": "Refunds are available within 30 days... [1]",
  "citations": [
    { "marker": 1, "document_id": "faq1", "source": "Support FAQ",
      "snippet": "Refunds are available within 30 days...", "score": 0.42 }
  ],
  "token_usage": { "prompt_tokens": 9, "context_tokens": 18, "completion_tokens": 12 },
  "grounded": true
}
```

## OpenAI backend

```bash
pip install -e ".[openai]"
export RAG_BACKEND=openai
export RAG_OPENAI_API_KEY=sk-...
uvicorn app.main:app
```

The OpenAI model is prompted to answer only from the retrieved context and to
cite with `[n]` markers — the same contract the offline answerer follows.

## Layout

```
app/
  main.py         FastAPI app (/ingest, /ask, /health)
  config.py       backend wiring from env (RAG_* vars)
  models.py       request/response schemas
  chunking.py     sentence-aware splitting with overlap
  tokens.py       token estimation + budgeting
  vectorstore.py  in-memory cosine search
  rag.py          retrieve -> ground -> answer
  embeddings/     hashing (offline) and openai
  llm/            extractive (offline) and openai
```

For production scale, the vector store and embedder are behind small interfaces,
so pgvector/Qdrant and a real embedding model drop in without touching `rag.py`.

## Tests

```bash
pytest -q
ruff check .
```

## Custom work

Want a RAG/LLM feature built on your own data and infrastructure?
Get in touch at [rovidev.com](https://rovidev.com).

## License

MIT
