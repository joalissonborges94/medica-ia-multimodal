"""Constroi o indice Chroma a partir dos PDFs em `data/raw/`.

Procura PDFs com prefixo `manual_ms_*` ou `febrasgo_*` (reais ou sinteticos)
e faz upsert no vector store. Idempotente: rodar varias vezes nao duplica
chunks (chunk_id e estavel).

Uso:
    python scripts/build_rag_index.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.ingestion import ingest_document  # noqa: E402
from src.rag.vector_store import VectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_rag_index")

INPUT_DIR = Path("data/raw")
PATTERNS = ("manual_ms_*.pdf", "febrasgo_*.pdf", "*.pdf")


def _coletar_documentos() -> list[Path]:
    """Lista PDFs em `data/raw/` priorizando prefixos esperados."""
    encontrados: list[Path] = []
    for pattern in PATTERNS:
        for p in INPUT_DIR.glob(pattern):
            if p not in encontrados:
                encontrados.append(p)
    return encontrados


def main() -> None:
    docs = _coletar_documentos()
    if not docs:
        logger.warning(
            "Nenhum PDF em %s. Rode scripts/gen_synthetic_pdfs.py para gerar exemplos.",
            INPUT_DIR,
        )
        sys.exit(1)

    store = VectorStore()
    store.load()
    if not store._available:
        logger.error("VectorStore nao disponivel. Verifique chromadb e sentence-transformers.")
        sys.exit(2)

    total_chunks = 0
    for doc in docs:
        logger.info("Processando %s", doc.name)
        chunks = ingest_document(doc)
        total_chunks += store.add(chunks)
    logger.info(
        "Indice atualizado: %d chunks (total na colecao: %d)",
        total_chunks,
        store.count(),
    )


if __name__ == "__main__":
    main()
