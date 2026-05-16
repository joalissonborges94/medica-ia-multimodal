"""Retriever sobre o `VectorStore` com filtros opcionais.

Camada fina sobre o store que aceita filtros amigaveis (por fonte, secao),
aplica threshold de similaridade para descartar resultados irrelevantes e
adiciona logging consistente das queries para observabilidade.
"""

from __future__ import annotations

import logging

from src.rag.types import Chunk, RetrievalResult
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Score minimo (1 - distance cosseno) para considerar um chunk relevante.
# Com bge-m3 multilingue e queries em PT-BR clinico:
#   > 0.5: relevancia alta (tema bem coberto pelos PDFs)
#   0.3-0.5: relevancia media (cobertura parcial)
#   < 0.3: provavelmente ruido (tema fora das diretrizes indexadas)
DEFAULT_MIN_SCORE: float = 0.3


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
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> RetrievalResult:
        """Busca trechos relevantes.

        Args:
            query: consulta em linguagem natural.
            top_k: quantos resultados retornar (antes do filtro de score).
            source: filtra por nome do documento (`Chunk.source`).
            section: filtra por secao (`Chunk.section`).
            min_score: descarta chunks com score abaixo desse limiar. Usar
                `0.0` para desativar o filtro. Default `0.3` cobre o ponto
                em que `bge-m3` ainda retorna conteudo relacionado para
                queries em PT-BR clinico.

        Returns:
            `RetrievalResult` ja filtrado por score (pode estar vazio se
            nenhum chunk passar do limiar, sinalizando ao chamador que o
            tema esta fora das diretrizes indexadas).
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
            "Retriever query: '%s' (top_k=%d, where=%s, min_score=%.2f)",
            query[:60],
            top_k,
            where,
            min_score,
        )
        raw = self.store.query(text=query, top_k=top_k, where=where)

        if min_score <= 0.0 or not raw.chunks:
            return raw

        filtered_chunks: list[Chunk] = []
        filtered_scores: list[float] = []
        for chunk, score in zip(raw.chunks, raw.scores, strict=True):
            if score >= min_score:
                filtered_chunks.append(chunk)
                filtered_scores.append(score)

        descartados = len(raw.chunks) - len(filtered_chunks)
        if descartados:
            logger.info(
                "Retriever filtrou %d/%d chunks abaixo do limiar %.2f (top score=%.3f)",
                descartados,
                len(raw.chunks),
                min_score,
                raw.scores[0] if raw.scores else 0.0,
            )
        return RetrievalResult(chunks=filtered_chunks, scores=filtered_scores)
