"""Detector de emocao facial baseado em FER (justinshenk/fer).

FER exige `Pillow` legado (nao builda em Python 3.14). O Dockerfile fixa
`python:3.12-slim` onde a instalacao funciona normalmente. Em ambiente
de desenvolvimento com Python 3.14 o detector faz fallback gracioso e
retorna lista vazia, sem quebrar o pipeline. Trocar por modelo HuggingFace
como evolucao futura caso desejado.

Como alternativa cloud sem vies de FER-2013 (que tende a rotular faces
femininas em repouso como `angry`/`sad`), existe `AzureOpenAIVisionEmotion`
em `src/video/azure_openai_vision.py`. Ele usa um deployment GPT-4o vision
(ex: `gpt-4o-mini`) no Foundry e segue a mesma estrategia adotada para
audio com `AzureOpenAIAudioEmotion`. A integracao plug-and-play (analoga
a `get_emotion_classifier()` em `src/audio/emotion.py`) sera feita quando
o deployment de visao for provisionado.
"""

from __future__ import annotations

import logging

import numpy as np

from src.video.types import BoundingBox, EmotionScore

logger = logging.getLogger(__name__)


class FacialEmotionDetector:
    """Classificador de emocao facial usando FER.

    O modelo e carregado lazy. Se a importacao do FER falhar (ambiente
    sem suporte), o detector marca-se como indisponivel e retorna [].
    """

    def __init__(self, use_mtcnn: bool = False) -> None:
        """Configura o detector.

        Args:
            use_mtcnn: usa MTCNN para deteccao de face (mais preciso, mais lento).
                Default usa o detector OpenCV Haar cascade do FER (mais rapido).
        """
        self.use_mtcnn: bool = use_mtcnn
        self._detector = None
        self._available: bool = False
        self._load_attempted: bool = False

    def load(self) -> None:
        """Carrega o FER. Em caso de falha, marca-se indisponivel."""
        self._load_attempted = True
        try:
            from fer import FER

            self._detector = FER(mtcnn=self.use_mtcnn)
            self._available = True
            logger.info("FER carregado (mtcnn=%s)", self.use_mtcnn)
        except ImportError as exc:
            logger.warning(
                "FER indisponivel no ambiente atual (%s). "
                "Detector facial retornara lista vazia ate ambiente compativel.",
                exc,
            )
            self._available = False

    def detect(self, frame: np.ndarray) -> list[EmotionScore]:
        """Detecta faces e classifica emocoes em um frame BGR.

        Args:
            frame: imagem BGR (HxWx3).

        Returns:
            Lista de `EmotionScore`, uma por face encontrada. Lista vazia
            se nenhuma face for detectada ou se o FER estiver indisponivel.
        """
        if not self._load_attempted:
            self.load()
        if not self._available or self._detector is None:
            return []
        results = self._detector.detect_emotions(frame)
        scores: list[EmotionScore] = []
        for entry in results:
            scores.append(_parse_fer_entry(entry))
        return scores


def _parse_fer_entry(entry: dict) -> EmotionScore:
    """Converte dicionario do FER em `EmotionScore`."""
    emotions = entry.get("emotions", {})
    if not emotions:
        return EmotionScore(label="unknown", confidence=0.0, scores={})
    label, confidence = max(emotions.items(), key=lambda kv: kv[1])
    bbox = None
    box = entry.get("box")
    if box is not None:
        x, y, w, h = box
        bbox = BoundingBox(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h))
    return EmotionScore(
        bbox=bbox,
        label=str(label),
        confidence=float(confidence),
        scores={k: float(v) for k, v in emotions.items()},
    )
