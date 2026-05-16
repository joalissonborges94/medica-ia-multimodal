"""Pipeline de audio completo.

Recebe um arquivo de audio e roda: transcricao (Whisper local ou Azure),
features acusticas (librosa), emocao vocal (wav2vec2) e analise textual
(Azure Language). Saida agregada em `AudioAnalysis`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.audio.azure_language import AzureLanguageClient
from src.audio.emotion import EmotionClassifierProtocol, get_emotion_classifier
from src.audio.features import extract_features
from src.audio.transcriber import TranscriberProtocol, get_transcriber
from src.audio.types import AudioAnalysis

logger = logging.getLogger(__name__)


class AudioPipeline:
    """Orquestra todos os modulos de analise de audio."""

    def __init__(
        self,
        transcriber: TranscriberProtocol | None = None,
        emotion_classifier: EmotionClassifierProtocol | None = None,
        language_client: AzureLanguageClient | None = None,
    ) -> None:
        """Configura o pipeline.

        Args:
            transcriber: instancia opcional. Default usa `get_transcriber()`
                que respeita o toggle `USE_CLOUD_TRANSCRIPTION`.
            emotion_classifier: instancia opcional. Default usa
                `get_emotion_classifier()` que escolhe entre wav2vec2 local
                e GPT-4o multimodal conforme `AZURE_OPENAI_AUDIO_DEPLOYMENT`.
            language_client: instancia opcional de `AzureLanguageClient`.
        """
        self.transcriber: TranscriberProtocol = transcriber or get_transcriber()
        self.emotion_classifier: EmotionClassifierProtocol = (
            emotion_classifier or get_emotion_classifier()
        )
        self.language_client: AzureLanguageClient = language_client or AzureLanguageClient()

    def process(self, audio_path: Path) -> AudioAnalysis:
        """Processa um arquivo de audio e retorna `AudioAnalysis`.

        Args:
            audio_path: caminho do arquivo.

        Returns:
            `AudioAnalysis` com todas as analises agregadas.

        Raises:
            FileNotFoundError: se o arquivo nao existir.
        """
        audio_path = audio_path.resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio nao encontrado: {audio_path}")
        logger.info("Iniciando pipeline de audio para %s", audio_path)

        text, segments = self.transcriber.transcribe(audio_path)
        features = extract_features(audio_path)
        emotion = self.emotion_classifier.classify(audio_path)
        sentiment = self.language_client.analyze_sentiment(text) if text else None
        key_phrases = self.language_client.extract_key_phrases(text) if text else []

        analysis = AudioAnalysis(
            transcription=text,
            segments=segments,
            acoustic_features=features,
            emotion=emotion,
            sentiment=sentiment,
            key_phrases=key_phrases,
            azure_metadata=None,
        )
        logger.info(
            "Pipeline concluido: %d caracteres, %d segmentos, emocao=%s, sentimento=%s",
            len(text),
            len(segments),
            emotion.label if emotion else "N/A",
            sentiment.label if sentiment else "N/A",
        )
        return analysis
