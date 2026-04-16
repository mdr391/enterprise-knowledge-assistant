# data/

This directory is created at runtime.

- `chroma_db/` — ChromaDB vector store (created on first run, excluded from git)

To reset the knowledge base, delete `chroma_db/` and re-run the seed script:

```bash
python scripts/seed_knowledge_base.py
```