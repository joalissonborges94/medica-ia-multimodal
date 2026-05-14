"""Retriever sobre o `VectorStore` com filtros opcionais.

Camada fina sobre o store que aceita filtros amigaveis (por fonte, secao)
e adiciona logging consistente das queries para observabilidade.
"""

from __future__ import annotations

import logging

from src.rag.types import RetrievalResult
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Camada de busca sobre `VectorStore` com filtros amigaveis."""

    def __init__(self, store: VectorStore | None = None) -> None:
        """Configura o retriever.

        Args:
            store: `VectorStore` opcional. Default cria um novo com defaults.
        """
        self.store: VectorStore = store or VectorStore()

    def search(
        self,
        query: str,
        top_k: int = 3,
        source: str | None = None,
        section: str | None = None,
    ) -> RetrievalResult:
        """Busca trechos relevantes.

        Args:
            query: consulta em linguagem natural.
            top_k: quantos resultados retornar.
            source: filtra por nome do documento (`Chunk.source`).
            section: filtra por secao (`Chunk.section`).

        Returns:
            `RetrievalResult`.
        """
        where: dict | None = None
        clauses: list[dict] = []
        if source is not None:
            clauses.append({"source": source})
        if section is not None:
            clauses.append({"section": section})
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

        logger.info(
            "Retriever query: '%s' (top_k=%d, where=%s)",
            query[:60],
            top_k,
            where,
        )
        return self.store.query(text=query, top_k=top_k, where=where)
