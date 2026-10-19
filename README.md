# Python Polars Vector RAG Engine

[![Python CI](https://github.com/FelipeMadson/python-polars-vector-rag-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/FelipeMadson/python-polars-vector-rag-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v1.0.0-green.svg)](https://github.com/FelipeMadson/python-polars-vector-rag-engine/releases)

High-performance vector retrieval and columnar data analysis engine combining Polars zero-copy execution with cosine similarity ranking and Pydantic v2 schemas.

## Features

- **Pydantic v2 Schema Validation**: Strict structural type checking for document and embedding ingestion.
- **Cosine Similarity Ranking**: Fast, numerical vector comparison without external API dependencies.
- **Zero-Cloud Dependency**: Designed for local-first, privacy-preserving LLM retrieval pipelines.

## Installation & Tests

```bash
pip install -e ".[test]"
pytest -v
```

## Author

Developed by **[Felipe Madison](https://github.com/FelipeMadson)**.
