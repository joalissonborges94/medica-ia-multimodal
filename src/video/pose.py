"""Estimador de pose corporal via YOLOv8 Pose (multi-person).

Substituiu o MediaPipe Pose por 2 motivos:
1. MediaPipe Pose detecta apenas UMA pessoa por frame. Em consultas
   medico+paciente, frequentemente pega a pessoa errada (medico
   geralmente mais frontal/proximo da camera).
2. Em Python 3.14 o pacote PyPI `mediapipe` vem reduzido (sem
   `solutions.pose`). YOLOv8 Pose via Ultralytics nao tem essa
   limitacao e funciona uniformemente em Py 3.12+ e 3.14.

YOLOv8 Pose retorna 17 keypoints no formato COCO (vs 33 do MediaPipe),
mas todos os landmarks que `classify_posture` usa (nariz, ombros,
quadris) estao presentes. Sem perda funcional.

Detecta multiplas pessoas por frame. Pra manter compatibilidade com o
pipeline atual, `estimate()` retorna landmarks da pessoa principal
(maior bbox = mais proxima/proeminente). Heuristica imperfeita em
casos onde medico esta mais perto que paciente, mas robusto pro
caso de uso (paciente costuma estar enquadrada de forma mais ampla).

`classify_posture(landmarks)` continua infering categoria postural
(ereta / inclinada / retraida / indefinido) a partir dos landmarks
(funciona transparentemente com 17 ou 33 keypoints).
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


# Mapeamento de indices YOLOv8 Pose (COCO format) -> nomes usados pelo
# classify_posture. Mantemos compativel com nomes do MediaPipe Pose pra
# nao quebrar codigo existente.
_COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

# Landmarks essenciais pra heuristica de postura (nomes COCO).
_LM_NOSE              = "nose"
_LM_LEFT_SHOULDER     = "left_shoulder"
_LM_RIGHT_SHOULDER    = "right_shoulder"
_LM_LEFT_HIP          = "left_hip"
_LM_RIGHT_HIP         = "right_hip"

# Visibilidade minima pra considerar landmark confiavel (range 0-1).
# YOLO Pose usa confidence como proxy de visibilidade. 0.3 e permissivo
# o suficiente pra paciente parcialmente oculta mantendo qualidade.
_MIN_VISIBILITY: float = 0.3

# Threshold de angulo (graus) entre eixo coluna (ombros->quadril) e vertical.
_ERETA_MAX_ANGLE_DEG: float = 15.0
_INCLINADA_MAX_ANGLE_DEG: float = 35.0


def classify_posture(landmarks: list[PoseLandmark]) -> PostureCategory:
    """Infere categoria postural a partir dos landmarks de uma pessoa.

    Estrategia:
    1. Valida que landmarks essenciais (ombros, quadril) estao visiveis
       (visibility >= `_MIN_VISIBILITY`).
    2. Calcula ponto medio dos ombros e quadril.
    3. Mede angulo entre eixo coluna (ombros -> quadril) e a vertical.
    4. Classifica em ereta/inclinada/retraida via thresholds calibrados.

    Args:
        landmarks: lista de `PoseLandmark` da pessoa principal retornada
            por `PoseEstimator.estimate`. Compativel com 17 keypoints
            COCO (YOLO Pose) ou 33 do MediaPipe Pose.

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

    ls = by_name[_LM_LEFT_SHOULDER]
    rs = by_name[_LM_RIGHT_SHOULDER]
    lh = by_name[_LM_LEFT_HIP]
    rh = by_name[_LM_RIGHT_HIP]

    shoulder_mid = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])
    hip_mid      = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2])

    spine = hip_mid - shoulder_mid
    vertical = np.array([0.0, 1.0])
    cos_theta = float(np.dot(spine, vertical) / (np.linalg.norm(spine) + 1e-8))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_deg = float(np.degrees(np.arccos(cos_theta)))

    # Detecta retraida: cabeca muito proxima dos ombros
    nose = by_name.get(_LM_NOSE)
    if nose is not None and nose.visibility >= _MIN_VISIBILITY:
        head_drop = abs(nose.y - shoulder_mid[1])
        torso_height = float(np.linalg.norm(spine))
        if torso_height > 0 and head_drop / torso_height < 0.15:
            return PostureCategory.RETRAIDA

    if angle_deg <= _ERETA_MAX_ANGLE_DEG:
        return PostureCategory.ERETA
    if angle_deg <= _INCLINADA_MAX_ANGLE_DEG:
        return PostureCategory.INCLINADA
    return PostureCategory.RETRAIDA


class PoseEstimator:
    """Estima landmarks corporais via YOLOv8 Pose (17 keypoints COCO).

    Detecta multiplas pessoas por frame. `estimate()` retorna a pose da
    pessoa principal (maior bbox = mais proxima/proeminente).

    O modelo (`yolov8n-pose.pt`, ~6 MB) e baixado automaticamente pela
    Ultralytics na primeira chamada.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.3,
        weights: str = "yolov8n-pose.pt",
    ) -> None:
        """Configura o estimador.

        Args:
            min_detection_confidence: threshold de confianca pra deteccao
                de pessoa (0-1). Default 0.3 e permissivo pra paciente
                parcialmente oculta no enquadramento.
            weights: nome ou caminho dos pesos YOLO Pose. Default
                `yolov8n-pose.pt` (nano, ~6 MB, baixado automaticamente
                pela Ultralytics na primeira chamada).
        """
        self.min_detection_confidence: float = min_detection_confidence
        self.weights: str = weights
        self._model = None
        self._available: bool = False
        self._load_attempted: bool = False

    def load(self) -> None:
        """Carrega o modelo YOLOv8 Pose via Ultralytics.

        Em caso de falha (ex: ultralytics nao instalado, sem conexao
        pra baixar pesos), marca como indisponivel e segue. Pipeline
        retorna lista vazia em vez de quebrar.
        """
        self._load_attempted = True
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
            self._available = True
            logger.info(
                "YOLOv8 Pose carregado (weights=%s, conf=%.2f)",
                self.weights, self.min_detection_confidence,
            )
        except (ImportError, FileNotFoundError, OSError) as exc:
            logger.warning(
                "YOLOv8 Pose indisponivel (%s). Estimador retornara lista vazia.",
                exc,
            )
            self._available = False

    def estimate(self, frame: np.ndarray) -> list[PoseLandmark]:
        """Estima landmarks corporais da pessoa principal num frame BGR.

        Em cenas com multiplas pessoas, retorna a pose da pessoa com
        maior bbox (mais proxima/proeminente). Para acesso a todas as
        poses simultaneamente, usar `estimate_all`.

        Args:
            frame: imagem BGR (HxWx3) compativel com OpenCV.

        Returns:
            Lista de `PoseLandmark` (17 itens COCO format) ou vazia se
            nenhuma pessoa for detectada ou se o modelo nao carregou.
        """
        all_poses = self.estimate_all(frame)
        return all_poses[0] if all_poses else []

    def estimate_all(self, frame: np.ndarray) -> list[list[PoseLandmark]]:
        """Estima landmarks de TODAS as pessoas detectadas, ordenadas por
        tamanho (maior bbox primeiro).

        Args:
            frame: imagem BGR (HxWx3) compativel com OpenCV.

        Returns:
            Lista de listas de `PoseLandmark`, uma por pessoa, ordenadas
            por area de bbox decrescente. Lista externa vazia se ninguem
            detectado ou modelo indisponivel.
        """
        if not self._load_attempted:
            self.load()
        if not self._available or self._model is None:
            return []

        result = self._model.predict(
            frame, conf=self.min_detection_confidence, verbose=False,
        )[0]

        if (
            result.keypoints is None
            or result.boxes is None
            or len(result.boxes.data) == 0
        ):
            return []

        # Ordena pessoas por area de bbox (maior primeiro)
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (n, 4)
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        )
        order = np.argsort(-areas)  # decrescente

        h, w = frame.shape[:2]
        all_keypoints = result.keypoints.data.cpu().numpy()  # (n_persons, 17, 3)

        all_poses: list[list[PoseLandmark]] = []
        for person_idx in order:
            person_kps = all_keypoints[person_idx]  # (17, 3) -- x, y, conf
            landmarks: list[PoseLandmark] = []
            for kp_idx, (x, y, conf) in enumerate(person_kps):
                name = _COCO_KEYPOINT_NAMES[kp_idx] if kp_idx < len(
                    _COCO_KEYPOINT_NAMES
                ) else f"landmark_{kp_idx}"
                landmarks.append(
                    PoseLandmark(
                        name=name,
                        x=float(x) / w if w > 0 else 0.0,
                        y=float(y) / h if h > 0 else 0.0,
                        z=0.0,  # YOLO Pose e 2D
                        visibility=float(conf),
                    )
                )
            all_poses.append(landmarks)
        return all_poses

    def close(self) -> None:
        """Libera os recursos do modelo (no-op pra YOLO via Ultralytics)."""
        # Ultralytics gerencia memoria internamente; nao precisa close
        # explicito. Mantemos o metodo pra compat com codigo existente.
        self._model = None
