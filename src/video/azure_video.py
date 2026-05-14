"""Cliente Azure Video Indexer (esqueleto).

A implementacao completa depende do provisionamento do servico (Tarefa 1.7),
que exige uma conta no portal Azure e em videoindexer.ai. Enquanto a chave
nao estiver configurada, este cliente retorna `None` em todas as chamadas,
permitindo que o pipeline rode end-to-end sem dependencia cloud.

Quando o servico for provisionado:
- Preencher `AZURE_VIDEO_INDEXER_KEY` e `AZURE_VIDEO_INDEXER_ACCOUNT_ID` em `.env`
- Implementar o fluxo: obter access token -> upload de video -> polling do estado
  -> coleta de insights (cenas, transcricao, faces, marcadores).
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config.settings import settings

logger = logging.getLogger(__name__)


class AzureVideoIndexerClient:
    """Cliente do Azure Video Indexer.

    Atualmente um esqueleto: detecta se as credenciais estao presentes e,
    em caso negativo, retorna None sem fazer chamadas remotas.
    """

    def __init__(self) -> None:
        self.account_id: str = settings.azure_video_indexer_account_id
        self.api_key: str = settings.azure_video_indexer_key.get_secret_value()

    @property
    def is_configured(self) -> bool:
        """`True` se ambos `account_id` e `api_key` estao definidos."""
        return bool(self.account_id and self.api_key)

    def analyze(self, video_path: Path) -> dict | None:
        """Envia um video para o Azure Video Indexer e retorna metadata.

        Args:
            video_path: caminho do arquivo de video.

        Returns:
            Dicionario com insights agregados, ou `None` quando o servico
            nao esta configurado (Tarefa 1.7 ainda pendente).
        """
        if not self.is_configured:
            logger.info(
                "Azure Video Indexer nao configurado (Tarefa 1.7); pulando analise cloud para %s",
                video_path,
            )
            return None
        # TODO(Tarefa 1.7+): implementar upload + polling de Insights.
        logger.warning(
            "Azure Video Indexer configurado, mas implementacao completa "
            "do upload/polling ainda pendente."
        )
        return None
