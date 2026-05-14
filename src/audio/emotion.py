"""Classificador de emocao em fala via wav2vec2 (HuggingFace).

Usa `superb/wav2vec2-base-superb-er` (treinado em IEMOCAP, 4 emocoes).
Lazy-load + fallback gracioso: se o modelo nao puder ser carregado
(rede, espaco em disco, incompatibilidade), retorna `None` sem quebrar.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.audio.types import EmotionScore

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "superb/wav2vec2-base-superb-er"


class VocalEmotionClassifier:
    """Classifica a emocao predominante em um arquivo de audio."""

    def __init__(self, model_name: str = DEFAULT_MODEL, sample_rate: int = 16_000) -> None:
        """Configura o classificador.

        Args:
            model_name: nome do modelo HuggingFace.
            sample_rate: taxa de amostragem esperada pelo modelo (16 kHz).
        """
        self.model_name: str = model_name
        self.sample_rate: int = sample_rate
        self._pipeline = None
        self._available: bool = False
        self._load_attempted: bool = False

    def load(self) -> None:
        """Carrega o pipeline `audio-classification` do transformers."""
        self._load_attempted = True
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="audio-classification",
                model=self.model_name,
            )
            self._available = True
            logger.info("Wav2vec2 emotion carregado (%s)", self.model_name)
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning(
                "Modelo de emocao vocal indisponivel (%s). Classificador retornara None.",
                exc,
            )
            self._available = False

    def classify(self, audio_path: Path) -> EmotionScore | None:
        """Classifica a emocao predominante em um arquivo de audio.

        Args:
            audio_path: caminho do `.wav`/`.mp3`/etc.

        Returns:
            `EmotionScore` com label, confianca e distribuicao de scores;
            ou `None` se o classificador estiver indisponivel.
        """
        if not self._load_attempted:
            self.load()
        if not self._available or self._pipeline is None:
            return None
        try:
            results = self._pipeline(str(audio_path))
        except (RuntimeError, ValueError) as exc:
            logger.warning("Falha ao classificar emocao em %s: %s", audio_path, exc)
            return None
        # `results` e uma lista de dicts {label, score} ja ordenada por score desc.
        scores = {entry["label"].lower(): float(entry["score"]) for entry in results}
        if not scores:
            return None
        label, confidence = max(scores.items(), key=lambda kv: kv[1])
        return EmotionScore(label=label, confidence=confidence, scores=scores)


def _ensure_float32(samples: np.ndarray) -> np.ndarray:
    """Garante que os samples estao em float32 conforme esperado pelo modelo."""
    if samples.dtype != np.float32:
        return samples.astype(np.float32)
    return samples
