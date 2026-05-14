"""Gerador de relatorio clinico em Markdown.

O relatorio combina:
- nivel de risco e triggers do `AnomalyResult`
- transcricao e sentimento do `AudioAnalysis`
- numero de eventos do pipeline de video
- diretrizes recuperadas (RAG) como contexto clinico

Quando o `AzureOpenAIClient` esta configurado, o LLM produz um texto
naturalizado em PT-BR a partir de um prompt estruturado. Sem credenciais,
um fallback deterministico monta um markdown com as mesmas secoes,
preservando a auditabilidade e mantendo o pipeline operacional.
"""

from __future__ import annotations

import logging

from src.anomaly.types import AnomalyResult
from src.audio.types import AudioAnalysis
from src.llm.azure_openai import AzureOpenAIClient
from src.rag.types import Chunk
from src.video.types import VideoEvent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_PT_BR: str = (
    "Voce e um assistente medico que produz relatorios clinicos em PT-BR. "
    "Use linguagem objetiva, sem inventar dados. Cite explicitamente os "
    "triggers detectados. Quando houver diretrizes clinicas no contexto, "
    "incorpore as recomendacoes relevantes."
)


def generate_report(
    *,
    anomaly: AnomalyResult,
    audio_analysis: AudioAnalysis | None = None,
    video_events: list[VideoEvent] | None = None,
    rag_chunks: list[Chunk] | None = None,
    patient_metadata: dict | None = None,
    llm_client: AzureOpenAIClient | None = None,
) -> str:
    """Gera um relatorio clinico em markdown.

    Args:
        anomaly: resultado final da analise de anomalia.
        audio_analysis: saida do pipeline de audio (opcional).
        video_events: lista de eventos do pipeline de video (opcional).
        rag_chunks: trechos das diretrizes recuperados pelo RAG (opcional).
        patient_metadata: dict com metadata do paciente (opcional).
        llm_client: cliente Azure OpenAI; quando ausente, sera instanciado.

    Returns:
        String em markdown pronta para exibicao na UI.
    """
    client = llm_client or AzureOpenAIClient()
    llm_text: str | None = None

    if client.is_configured:
        prompt = _build_user_prompt(
            anomaly=anomaly,
            audio_analysis=audio_analysis,
            video_events=video_events,
            rag_chunks=rag_chunks,
            patient_metadata=patient_metadata,
        )
        llm_text = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT_PT_BR},
                {"role": "user", "content": prompt},
            ]
        )
        if llm_text:
            logger.info("Relatorio gerado via LLM (%d chars).", len(llm_text))
            return llm_text
        logger.warning("LLM retornou vazio; usando fallback deterministico.")

    return _fallback_report(
        anomaly=anomaly,
        audio_analysis=audio_analysis,
        video_events=video_events,
        rag_chunks=rag_chunks,
        patient_metadata=patient_metadata,
    )


def _build_user_prompt(
    *,
    anomaly: AnomalyResult,
    audio_analysis: AudioAnalysis | None,
    video_events: list[VideoEvent] | None,
    rag_chunks: list[Chunk] | None,
    patient_metadata: dict | None,
) -> str:
    """Monta o prompt do usuario com todos os fatos estruturados."""
    lines: list[str] = []
    if patient_metadata:
        lines.append(f"Paciente: {patient_metadata}")
    lines.append(f"Nivel de risco final: {anomaly.level}")
    if anomaly.triggers:
        lines.append("Triggers detectados:")
        for trigger in anomaly.triggers:
            lines.append(
                f"- [{trigger.level}] ({trigger.source}/{trigger.rule_id}) {trigger.message}"
            )
    else:
        lines.append("Nenhum trigger clinico disparado.")

    if video_events is not None:
        lines.append(f"Frames analisados: {len(video_events)}.")

    if audio_analysis is not None:
        if audio_analysis.transcription:
            lines.append(f'Transcricao: "{audio_analysis.transcription.strip()}"')
        if audio_analysis.emotion is not None:
            lines.append(
                f"Emocao vocal predominante: {audio_analysis.emotion.label}"
                f" (confianca {audio_analysis.emotion.confidence:.2f})."
            )
        if audio_analysis.sentiment is not None:
            lines.append(
                f"Sentimento textual: {audio_analysis.sentiment.label}"
                f" (confianca {audio_analysis.sentiment.confidence:.2f})."
            )
        if audio_analysis.key_phrases:
            lines.append("Frases-chave: " + ", ".join(audio_analysis.key_phrases))

    if rag_chunks:
        lines.append("Diretrizes clinicas relevantes:")
        for chunk in rag_chunks[:5]:
            snippet = chunk.text.strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:277] + "..."
            lines.append(f"- ({chunk.source}) {snippet}")

    lines.append(
        "Gere um relatorio em markdown com as secoes: Resumo, Achados, "
        "Diretrizes Aplicaveis, Recomendacoes."
    )
    return "\n".join(lines)


def _fallback_report(
    *,
    anomaly: AnomalyResult,
    audio_analysis: AudioAnalysis | None,
    video_events: list[VideoEvent] | None,
    rag_chunks: list[Chunk] | None,
    patient_metadata: dict | None,
) -> str:
    """Monta um relatorio determinístico em markdown sem LLM."""
    parts: list[str] = ["# Relatorio Clinico Automatico", ""]
    parts.append(f"**Nivel de risco:** `{anomaly.level}`")
    if patient_metadata:
        parts.append(f"**Paciente:** {patient_metadata}")
    parts.append("")

    parts.append("## Resumo")
    parts.append(anomaly.explanation or "Sem observacoes adicionais.")
    parts.append("")

    parts.append("## Achados")
    if anomaly.triggers:
        for trigger in anomaly.triggers:
            parts.append(
                f"- **[{trigger.level}]** ({trigger.source}/{trigger.rule_id}) {trigger.message}"
            )
    else:
        parts.append("- Nenhum sinal clinico relevante detectado.")
    parts.append("")

    parts.append("## Modalidades Analisadas")
    if video_events is not None:
        parts.append(f"- Video: {len(video_events)} frames analisados.")
    if audio_analysis is not None:
        if audio_analysis.transcription:
            parts.append(f'- Audio: transcricao "{audio_analysis.transcription.strip()}"')
        if audio_analysis.emotion is not None:
            parts.append(
                f"- Emocao vocal: {audio_analysis.emotion.label}"
                f" (confianca {audio_analysis.emotion.confidence:.2f})"
            )
        if audio_analysis.sentiment is not None:
            parts.append(
                f"- Sentimento textual: {audio_analysis.sentiment.label}"
                f" (confianca {audio_analysis.sentiment.confidence:.2f})"
            )
    if video_events is None and audio_analysis is None:
        parts.append("- Nenhuma modalidade fornecida.")
    parts.append("")

    if rag_chunks:
        parts.append("## Diretrizes Aplicaveis")
        for chunk in rag_chunks[:5]:
            snippet = chunk.text.strip().replace("\n", " ")
            if len(snippet) > 280:
                snippet = snippet[:277] + "..."
            parts.append(f"- _{chunk.source}_: {snippet}")
        parts.append("")

    parts.append("## Recomendacoes")
    if anomaly.recommended_actions:
        for action in anomaly.recommended_actions:
            parts.append(f"- {action}")
    else:
        parts.append("- Manter monitoramento padrao.")

    return "\n".join(parts).rstrip() + "\n"
