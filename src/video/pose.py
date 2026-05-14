"""Estimador de pose corporal via MediaPipe Pose.

Usa `mediapipe.solutions.pose` quando disponivel. Em Python 3.14 o pacote
`mediapipe` no PyPI vem reduzido (sem `solutions`, apenas a Tasks API);
nesse caso o estimador faz fallback gracioso e retorna lista vazia. O
`Dockerfile` (python:3.12-slim) traz `mediapipe.solutions.pose` completo,
garantindo o pipeline funcional em producao.
"""

from __future__ import annotations

import logging

import numpy as np

from src.video.types import PoseLandmark

logger = logging.getLogger(__name__)


class PoseEstimator:
    """Estima landmarks corporais (33 pontos) a partir de frames de video."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        """Configura o estimador.

        Args:
            min_detection_confidence: confianca minima do detector de pose.
        """
        self.min_detection_confidence: float = min_detection_confidence
        self._pose = None
        self._pose_landmark_enum = None
        self._available: bool = False
        self._load_attempted: bool = False

    def load(self) -> None:
        """Inicializa o solver MediaPipe Pose.

        Em ambientes onde `mediapipe.solutions.pose` nao existe (ex: pacote
        reduzido no Python 3.14), marca-se como indisponivel e segue.
        """
        self._load_attempted = True
        try:
            import mediapipe as mp

            mp_pose = mp.solutions.pose
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=self.min_detection_confidence,
            )
            self._pose_landmark_enum = mp_pose.PoseLandmark
            self._available = True
            logger.info(
                "MediaPipe Pose carregado (min_detection_confidence=%.2f)",
                self.min_detection_confidence,
            )
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "MediaPipe Pose indisponivel no ambiente atual (%s). "
                "Estimador retornara lista vazia ate ambiente compativel.",
                exc,
            )
            self._available = False

    def estimate(self, frame: np.ndarray) -> list[PoseLandmark]:
        """Estima os landmarks corporais em um frame BGR.

        Args:
            frame: imagem BGR (HxWx3) compativel com OpenCV.

        Returns:
            Lista de `PoseLandmark`. Lista vazia se nenhuma pose for detectada
            ou se MediaPipe estiver indisponivel.
        """
        if not self._load_attempted:
            self.load()
        if not self._available or self._pose is None:
            return []
        # MediaPipe espera RGB.
        rgb = frame[..., ::-1]
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return []
        landmarks: list[PoseLandmark] = []
        for idx, landmark in enumerate(result.pose_landmarks.landmark):
            landmarks.append(
                PoseLandmark(
                    name=self._landmark_name(idx),
                    x=float(landmark.x),
                    y=float(landmark.y),
                    z=float(landmark.z),
                    visibility=float(landmark.visibility),
                )
            )
        return landmarks

    def close(self) -> None:
        """Libera os recursos do solver."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def _landmark_name(self, index: int) -> str:
        """Mapeia indice (0-32) para nome legivel do enum MediaPipe."""
        if self._pose_landmark_enum is None:
            return f"landmark_{index}"
        try:
            return self._pose_landmark_enum(index).name.lower()
        except ValueError:
            return f"landmark_{index}"
