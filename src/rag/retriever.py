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

    def multi_search(
        self,
        queries: list[str],
        top_k_per_query: int = 3,
        total_k: int = 6,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> RetrievalResult:
        """Roda multiplas queries focadas e funde resultados, deduplicando por chunk_id.

        Util quando o caso tem multiplos eixos tematicos (clinico, saude
        mental, violencia, reprodutivo, rastreio): uma query unica mistura
        os sinais e o ranking acaba dominado pelo PDF mais volumoso. Queries
        separadas garantem que cada eixo puxe o conteudo mais relevante do
        PDF correto.

        Estrategia:
        1. Para cada query, busca `top_k_per_query` chunks (com filtro de score).
        2. Funde via interleaving (round-robin) entre queries, dedup por
           `chunk_id`. Round-robin equilibra a representacao de cada eixo
           no contexto final em vez de concatenar (que daria todo o peso
           pra primeira query).
        3. Retorna ate `total_k` chunks unicos no melhor score que apareceu
           pra cada chunk em qualquer das queries.

        Args:
            queries: lista de queries em linguagem natural. Vazias sao ignoradas.
            top_k_per_query: chunks a buscar por query (antes do dedupe).
            total_k: chunks unicos no resultado final.
            min_score: limiar de score (default `DEFAULT_MIN_SCORE`).

        Returns:
            `RetrievalResult` com ate `total_k` chunks distintos, ordenados
            por melhor score observado entre as queries.
        """
        non_empty = [q.strip() for q in queries if q and q.strip()]
        if not non_empty:
            return RetrievalResult(chunks=[], scores=[])

        # Roda cada query independente e mantem melhor score por chunk_id
        per_query_chunks: list[list[Chunk]] = []
        per_query_scores: list[list[float]] = []
        best_score_by_id: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}

        for query in non_empty:
            result = self.search(
                query, top_k=top_k_per_query, min_score=min_score,
            )
            per_query_chunks.append(list(result.chunks))
            per_query_scores.append(list(result.scores))
            for chunk, score in zip(result.chunks, result.scores, strict=True):
                if score > best_score_by_id.get(chunk.chunk_id, -1.0):
                    best_score_by_id[chunk.chunk_id] = score
                    chunks_by_id[chunk.chunk_id] = chunk

        # Interleaving round-robin entre queries: rodada 1 pega 1o chunk de
        # cada query, rodada 2 pega 2o de cada, etc. Dedupe por chunk_id.
        selected_ids: list[str] = []
        max_depth = max((len(cs) for cs in per_query_chunks), default=0)
        for depth in range(max_depth):
            for chunks_list in per_query_chunks:
                if depth >= len(chunks_list):
                    continue
                cid = chunks_list[depth].chunk_id
                if cid not in selected_ids:
                    selected_ids.append(cid)
                    if len(selected_ids) >= total_k:
                        break
            if len(selected_ids) >= total_k:
                break

        final_chunks = [chunks_by_id[cid] for cid in selected_ids]
        final_scores = [best_score_by_id[cid] for cid in selected_ids]
        logger.info(
            "Retriever multi_search: %d querie(s), %d chunks unicos retornados",
            len(non_empty),
            len(final_chunks),
        )
        return RetrievalResult(chunks=final_chunks, scores=final_scores)
