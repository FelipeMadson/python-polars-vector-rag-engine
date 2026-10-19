# Motor de RAG Vetorial e Análise de Dados com Polars em Python

[![Python CI](https://github.com/FelipeMadson/python-polars-vector-rag-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/FelipeMadson/python-polars-vector-rag-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v1.0.0-green.svg)](https://github.com/FelipeMadson/python-polars-vector-rag-engine/releases)

Motor de busca e recuperação vetorial (RAG) de alta performance integrando o poder colunar do Polars com ranking de similaridade de cossenos e validação estrutural via Pydantic v2.

## Recursos de Engenharia

- **Validação com Pydantic v2**: Ingestão estrita de esquemas de documentos e vetores densos.
- **Ranking por Similaridade de Cossenos**: Comparação vetorial matemática de alta velocidade sem chamadas de nuvem externas.
- **Local-First & Privado**: Arquitetura projetada para RAG offline e auditoria técnica completa.

## Instalação e Testes

```bash
pip install -e ".[test]"
pytest -v
```

## Autor

Desenvolvido por **[Felipe Madison](https://github.com/FelipeMadson)**.
