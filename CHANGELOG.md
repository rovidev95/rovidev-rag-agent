# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-06-27

### Added
- Retrieval-augmented generation with grounded answers and inline `[n]` citations.
- Offline hashing embedder and extractive answerer (runs without an API key).
- Optional OpenAI embeddings/chat backend behind the same answer contract.
- Sentence-aware chunking, token budgeting and an in-memory vector store.
