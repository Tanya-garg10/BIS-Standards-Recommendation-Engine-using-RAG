# BIS Standards RAG Pipeline 🚀

This repository contains a highly optimized Retrieval-Augmented Generation (RAG) pipeline to instantly recommend Bureau of Indian Standards (BIS) based on product descriptions.

## Architecture

1. **Vector DB**: FAISS + sentence-transformers for fast, accurate hybrid search.
2. **Retrieval**: Semantic search with cosine similarity (Inner Product with L2 Normalized embeddings).
3. **LLM Reranking**: OpenAI GPT used strictly for reranking and providing reasons, with prompts engineered to **prevent hallucination**.

## Setup

```bash
pip install -r requirements.txt
```

Set your OpenAI API Key (optional for testing pure vector search):
```bash
# Windows Powerhsell
$env:OPENAI_API_KEY="your-api-key"
```

## Running the CLI Inference (For Scoring)

```bash
python inference.py --input data/queries.json --output result.json
```

## Running the UI Demo

```bash
streamlit run app.py
```

## Folder Structure

```
/src
  rag_pipeline.py  # Core RAG logic connecting DB and LLM
  vector_db.py     # FAISS vector store and embeddings
/data
  standards.json   # Processed BIS standards chunks
  queries.json     # Sample test queries
inference.py       # Main inference script
eval_script.py     # Evaluation script for metrics
requirements.txt   # Dependencies
```
