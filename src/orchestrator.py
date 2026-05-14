"""Orquestrador multimodal: ponto unico de entrada (`process_case`).

Recebe `CaseInput` com paths de video e/ou audio (e contexto opcional),
roda os pipelines disponiveis, agrega no `AnomalyClassifier`, monta o
contexto RAG, gera o relatorio, dispara o alerta e registra no audit log.

Design:
- Cada modalidade e opcional. Se `video_path` ou `audio_path` for `None`,
  o pipeline correspondente nao roda
- O RAG e consultado a partir das mensagens dos triggers + transcricao,
  e e tolerante a falhas (`try/except` no retrieval)
- Tudo que e externo (LLM, RAG, audit) e injetavel para facilitar testes
  e para a UI poder configurar sem mexer no codigo
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from src.alert import Alert, build_alert, dispatch_alert
from src.anomaly.classifier import AnomalyClassifier
from src.anomaly.types import AnomalyResult
from src.audio.pipeline import AudioPipeline
from src.audio.types import AudioAnalysis
from src.audit import AuditLogger
from src.llm.azure_openai import AzureOpenAIClient
from src.rag.retriever import Retriever
from src.rag.types import Chunk
from src.report import generate_report
from src.video.pipeline import VideoPipeline
from src.video.types import VideoEvent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.alert import Dispatcher

logger = logging.getLogger(__name__)

DEFAULT_RAG_TOP_K: int = 4
MAX_QUERY_CHARS: int = 400


class CaseInput(BaseModel):
    """Entrada do orquestrador para um caso clinico completo."""

    video_path: Path | None = None
    audio_path: Path | None = None
    text_context: str | None = None
    patient_metadata: dict = Field(default_factory=dict)


class CaseOutput(BaseModel):
    """Saida consolidada do orquestrador.

    Contem todas as analises feitas para o caso, prontas para a UI,
    para o relatorio e para a auditoria.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    case_id: str
    video_events: list[VideoEvent] | None = None
    audio_analysis: AudioAnalysis | None = None
    rag_context: list[Chunk] = Field(default_factory=list)
    anomaly: AnomalyResult
    report_markdown: str
    alert: Alert | None = None
    audit_id: int | None = None


class Orchestrator:
    """Cola entre pipelines, classificador, RAG, LLM, audit e alerta."""

    def __init__(
        self,
        *,
        video_pipeline: VideoPipeline | None = None,
        audio_pipeline: AudioPipeline | None = None,
        classifier: AnomalyClassifier | None = None,
        retriever: Retriever | None = None,
        llm_client: AzureOpenAIClient | None = None,
        auditor: AuditLogger | None = None,
        alert_dispatchers: Sequence[Dispatcher] | None = None,
        rag_top_k: int = DEFAULT_RAG_TOP_K,
    ) -> None:
        """Configura o orquestrador.

        Args:
            video_pipeline: pipeline de video; default `VideoPipeline()`.
            audio_pipeline: pipeline de audio; default `AudioPipeline()`.
            classifier: classificador final; default `AnomalyClassifier()`.
            retriever: retriever RAG; default `Retriever()`. Se a indexacao
                falhar, o retriever sera ignorado em runtime.
            llm_client: cliente LLM; default `AzureOpenAIClient()`.
            auditor: logger SQLite; default `AuditLogger()`.
            alert_dispatchers: lista de dispatchers; default `[_log_dispatcher]`.
            rag_top_k: quantos chunks recuperar do RAG.
        """
        self.video_pipeline: VideoPipeline = video_pipeline or VideoPipeline()
        self.audio_pipeline: AudioPipeline = audio_pipeline or AudioPipeline()
        self.classifier: AnomalyClassifier = classifier or AnomalyClassifier()
        self.retriever: Retriever | None = retriever if retriever is not None else None
        self.llm_client: AzureOpenAIClient = llm_client or AzureOpenAIClient()
        self.auditor: AuditLogger | None = auditor
        self.alert_dispatchers: list[Dispatcher] | None = (
            list(alert_dispatchers) if alert_dispatchers else None
        )
        self.rag_top_k: int = rag_top_k

    def process_case(self, case: CaseInput) -> CaseOutput:
        """Pipeline completo para um caso multimodal.

        Args:
            case: entrada com paths e contexto opcional.

        Returns:
            `CaseOutput` com todas as analises, relatorio, alerta e id de audit.
        """
        case_id = self._new_case_id()
        logger.info("Iniciando caso %s", case_id)

        video_events = self._run_video(case.video_path)
        audio_analysis = self._run_audio(case.audio_path)

        anomaly = self.classifier.classify(video_events=video_events, audio_analysis=audio_analysis)

        rag_context = self._retrieve_context(case, audio_analysis, anomaly)

        report_markdown = generate_report(
            anomaly=anomaly,
            audio_analysis=audio_analysis,
            video_events=video_events,
            rag_chunks=rag_context,
            patient_metadata=case.patient_metadata,
            llm_client=self.llm_client,
        )

        alert = build_alert(case_id=case_id, anomaly=anomaly)
        dispatch_alert(alert, self.alert_dispatchers)

        modalities: list[str] = []
        if video_events is not None:
            modalities.append("video")
        if audio_analysis is not None:
            modalities.append("audio")
        if case.text_context:
            modalities.append("text")

        audit_id: int | None = None
        if self.auditor is not None:
            audit_id = self.auditor.log_case(
                case_id=case_id,
                anomaly=anomaly,
                modalities=modalities,
                report_md=report_markdown,
                metadata={
                    "patient": case.patient_metadata,
                    "text_context": case.text_context,
                },
            )

        return CaseOutput(
            case_id=case_id,
            video_events=video_events,
            audio_analysis=audio_analysis,
            rag_context=rag_context,
            anomaly=anomaly,
            report_markdown=report_markdown,
            alert=alert,
            audit_id=audit_id,
        )

    def _new_case_id(self) -> str:
        """Gera um identificador unico curto para o caso."""
        return f"case-{uuid.uuid4().hex[:12]}"

    def _run_video(self, video_path: Path | None) -> list[VideoEvent] | None:
        """Roda o pipeline de video se houver path; senao retorna `None`."""
        if video_path is None:
            return None
        try:
            return self.video_pipeline.process(video_path)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Falha no pipeline de video: %s", exc)
            return None

    def _run_audio(self, audio_path: Path | None) -> AudioAnalysis | None:
        """Roda o pipeline de audio se houver path; senao retorna `None`."""
        if audio_path is None:
            return None
        try:
            return self.audio_pipeline.process(audio_path)
        except FileNotFoundError as exc:
            logger.warning("Falha no pipeline de audio: %s", exc)
            return None

    def _retrieve_context(
        self,
        case: CaseInput,
        audio_analysis: AudioAnalysis | None,
        anomaly: AnomalyResult,
    ) -> list[Chunk]:
        """Monta query a partir de triggers + transcricao + contexto e busca."""
        if self.retriever is None:
            return []

        query_parts: list[str] = []
        if case.text_context:
            query_parts.append(case.text_context)
        if audio_analysis is not None and audio_analysis.transcription:
            query_parts.append(audio_analysis.transcription)
        for trigger in anomaly.triggers:
            query_parts.append(trigger.message)

        query = " ".join(query_parts).strip()
        if not query:
            return []
        query = query[:MAX_QUERY_CHARS]

        try:
            result = self.retriever.search(query, top_k=self.rag_top_k)
        except Exception as exc:  # noqa: BLE001 - RAG pode falhar por falta de indice
            logger.warning("RAG retrieval falhou: %s", exc)
            return []
        return list(result.chunks)


def process_case(case: CaseInput) -> CaseOutput:
    """Atalho de conveniencia: instancia um `Orchestrator` padrao e processa."""
    return Orchestrator().process_case(case)
