"""Estimador de pose corporal via MediaPipe Pose.

Usa `mediapipe.solutions.pose` quando disponivel. Em Python 3.14 o pacote
`mediapipe` no PyPI vem reduzido (sem `solutions`, apenas a Tasks API);
nesse caso o estimador faz fallback gracioso e retorna lista vazia. O
`Dockerfile` (python:3.12-slim) traz `mediapipe.solutions.pose` completo,
garantindo o pipeline funcional em producao.

Tambem expoe `classify_posture(landmarks)` que infere categoria
postural (ereta / inclinada / retraida / indefinido) a partir dos 33
landmarks. Util pra cenas de consulta clinica onde rosto pode estar
mascarado mas postura corporal e sinal de bem-estar/distress.
"""

from __future__ import annotations

import logging
from enum import StrEnum

import numpy as np

from src.video.types import PoseLandmark

logger = logging.getLogger(__name__)


class PostureCategory(StrEnum):
    """Categoria postural inferida via angulos dos landmarks."""

    ERETA       = "ereta"
    INCLINADA   = "inclinada"
    RETRAIDA    = "retraida"
    INDEFINIDO  = "indefinido"


# Landmarks MediaPipe Pose relevantes pra postura (nomes lower-case).
# Lista completa em https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md
_LM_NOSE              = "nose"
_LM_LEFT_SHOULDER     = "left_shoulder"
_LM_RIGHT_SHOULDER    = "right_shoulder"
_LM_LEFT_HIP          = "left_hip"
_LM_RIGHT_HIP         = "right_hip"

# Visibilidade minima pra considerar landmark confiavel (MediaPipe range 0-1).
_MIN_VISIBILITY: float = 0.5

# Threshold de angulo (graus) entre eixo coluna (ombros->quadril) e vertical.
# Postura ereta: tronco proximo da vertical (angulo pequeno).
# Postura inclinada: tronco inclinado pra frente (angulo medio).
# Postura retraida: ombros caidos + tronco curto (heuristica composta).
_ERETA_MAX_ANGLE_DEG: float = 15.0
_INCLINADA_MAX_ANGLE_DEG: float = 35.0


def classify_posture(landmarks: list[PoseLandmark]) -> PostureCategory:
    """Infere categoria postural a partir dos landmarks MediaPipe Pose.

    Estrategia:
    1. Valida que landmarks essenciais (ombros, quadril) estao visiveis
       (visibility >= `_MIN_VISIBILITY`).
    2. Calcula ponto medio dos ombros e quadril.
    3. Mede angulo entre eixo coluna (ombros -> quadril) e a vertical.
    4. Classifica em ereta/inclinada/retraida via thresholds calibrados.

    Args:
        landmarks: lista de `PoseLandmark` retornada por `PoseEstimator.estimate`.
            Espera o conjunto completo de 33 landmarks MediaPipe Pose.

    Returns:
        `PostureCategory`. Retorna `INDEFINIDO` se landmarks essenciais
        nao estiverem visiveis ou se a lista estiver vazia.
    """
    if not landmarks:
        return PostureCategory.INDEFINIDO

    by_name = {lm.name: lm for lm in landmarks}

    required = (
        _LM_LEFT_SHOULDER, _LM_RIGHT_SHOULDER,
        _LM_LEFT_HIP, _LM_RIGHT_HIP,
    )
    for name in required:
        lm = by_name.get(name)
        if lm is None or lm.visibility < _MIN_VISIBILITY:
            return PostureCategory.INDEFINIDO

    # Ponto medio dos ombros e quadril (eixo coluna)
    ls = by_name[_LM_LEFT_SHOULDER]
    rs = by_name[_LM_RIGHT_SHOULDER]
    lh = by_name[_LM_LEFT_HIP]
    rh = by_name[_LM_RIGHT_HIP]

    shoulder_mid = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])
    hip_mid      = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2])

    spine = hip_mid - shoulder_mid  # vetor ombros->quadril
    # angulo entre coluna e vertical (eixo y descendente em coords da imagem)
    vertical = np.array([0.0, 1.0])
    cos_theta = float(np.dot(spine, vertical) / (np.linalg.norm(spine) + 1e-8))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_deg = float(np.degrees(np.arccos(cos_theta)))

    # Detecta retraida: ombros muito proximos do nariz (cabeca baixa+ombros caidos)
    nose = by_name.get(_LM_NOSE)
    if nose is not None and nose.visibility >= _MIN_VISIBILITY:
        # Distancia vertical nose -> shoulder_mid; pequena = cabeca baixa
        head_drop = abs(nose.y - shoulder_mid[1])
        torso_height = float(np.linalg.norm(spine))
        # Heuristica: se cabeca quase encostando nos ombros, retraida
        if torso_height > 0 and head_drop / torso_height < 0.15:
            return PostureCategory.RETRAIDA

    if angle_deg <= _ERETA_MAX_ANGLE_DEG:
        return PostureCategory.ERETA
    if angle_deg <= _INCLINADA_MAX_ANGLE_DEG:
        return PostureCategory.INCLINADA
    return PostureCategory.RETRAIDA


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
