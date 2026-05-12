# Advanced LLM Context Engineering

This project is a modular RAG system for legal documents in the energy domain. It explores how different context-engineering techniques affect answer quality, cost, and robustness when working with long, structured PDFs.

## What the project includes

- document ingestion for PDF files and text cleaning,
- two chunking strategies: structural and semantic,
- query understanding through query rewriting,
- retrieval with Qdrant-based vector search,
- reranking with original order or U-shape reordering,
- context compression with extractive and hierarchical methods,
- semantic cache for repeated or similar questions,
- a Streamlit chat interface for interactive use,
- evaluation of answer quality and the lost-in-the-middle effect.

## How it works

The pipeline starts by loading legal PDFs from `data/pdf`, cleaning the text, and splitting it into chunks. Those chunks are embedded and stored in Qdrant. When a user asks a question, the system can rewrite the query, retrieve relevant chunks, rerank them, compress the context, and send the final prompt to an LLM.

The application supports multiple LLM providers and lets you switch strategies directly from the sidebar. This makes it easy to compare configurations and see how each design choice changes the final answer.

## Evaluation focus

The project compares several context-engineering approaches in terms of:

- faithfulness and answer relevance,
- completeness of the response,
- latency and token usage,
- context precision and context relevance,
- mitigation of the lost-in-the-middle problem.

The experiments highlight the trade-offs between quality and cost, especially for long legal documents where preserving structure is often more useful than relying on a purely semantic split.

## Application Preview

![Application screenshot](docs/app3.png)

## Project structure

- `ingestion/` - loading, cleaning, chunking, and vector storage,
- `retrieval/`, `reranking/`, `compression/`, `query/` - the main RAG pipeline stages,
- `generation/` - embeddings and LLM adapters,
- `evaluation/` - experiment scripts and metrics,
- `cache/` - semantic cache implementation,
- `docs/` - project documentation and screenshots.