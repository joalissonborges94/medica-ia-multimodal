"""Classificador leve de tipo de cena para videos medicos.

Amostra alguns frames do video e decide, a partir de heuristica baseada em
deteccao de face (MediaPipe Face Detection) e assinatura de matiz HSV, se o
video e de:

- `cirurgia`  : campo cirurgico laparoscopico (poucos rostos, matiz vermelho-rosado escuro)
- `consulta`  : consulta clinica/ambulatorial (rostos presentes, paleta de pele/ambiente)
- `misto`     : mistura dos dois perfis
- `desconhecido`: falha ao ler frames ou dependencias indisponiveis

O classificador usa lazy import de `mediapipe`. Quando o pacote nao estiver
disponivel, retorna `SceneType.UNKNOWN` sem quebrar o pipeline.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Faixas de matiz HSV (0-180 no OpenCV, escala [0, 179])
# -----------------------------------------------------------------------
# Cirurgia laparoscopica: tecido biologico iluminado por luz fria endoscopica.
# Predominam tons vermelhos e rosados escuros.  Hue em [0..15] e [165..179].
_SURGERY_HUE_LOW_MIN: int = 0
_SURGERY_HUE_LOW_MAX: int = 15
_SURGERY_HUE_HIGH_MIN: int = 165
_SURGERY_HUE_HIGH_MAX: int = 179

# Consulta: pele humana, paredes claras, roupas variadas. Hue mais difuso.
# Valores tipicos de pele: [0..25], mas o ambiente todo e variado.
# Usamos saturacao para separar: campo cirurgico tem saturacao alta;
# consulta tem mix com areas dessaturadas (paredes, roupas neutras).
_CONSULTATION_SAT_MAX: float = 120.0   # saturacao media abaixo disso indica consulta
_SURGERY_SAT_MIN: float = 80.0         # campo cirurgico costuma ter sat alta

# -----------------------------------------------------------------------
# Limiares de decisao
# -----------------------------------------------------------------------
_FACE_MAJORITY_RATIO: float = 0.5   # >= 50 % dos frames com face -> consulta candidata
_SURGERY_HUE_RATIO: float = 0.4     # >= 40 % dos frames com hue tipico -> cirurgia candidata


class SceneType(StrEnum):
    """Tipo de cena identificada no video."""

    SURGERY = "cirurgia"
    CONSULTATION = "consulta"
    MIXED = "misto"
    UNKNOWN = "desconhecido"


def classify_scene_type(
    video_path: Path,
    num_samples: int = 5,
) -> SceneType:
    """Classifica o tipo de cena a partir de frames amostrados.

    Amostra `num_samples` frames distribuidos uniformemente pelo video. Para
    cada frame calcula:
    - `has_face`: via MediaPipe Face Detection (lazy import).
    - `hue_signature`: media de matiz e saturacao no espaco HSV.

    A decisao final combina a proporcao de frames com face e a assinatura
    de matiz para diferenciar consulta de cirurgia.

    Args:
        video_path: caminho para o arquivo de video.
        num_samples: quantidade de frames amostrados. Default 5.

    Returns:
        Um valor de `SceneType`. Retorna `UNKNOWN` quando nao e possivel
        ler frames ou quando `mediapipe` nao esta disponivel.
    """
    import cv2

    face_detector = _load_face_detector()

    video_path = Path(video_path).resolve()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("SceneClassifier: nao conseguiu abrir %s", video_path)
        return SceneType.UNKNOWN

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.warning(
                "SceneClassifier: video sem frames validos em %s", video_path
            )
            return SceneType.UNKNOWN

        indices = _sample_indices(total_frames, num_samples)
        face_hits: int = 0
        surgery_hue_hits: int = 0
        valid_frames: int = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            valid_frames += 1

            if _has_face(frame, face_detector):
                face_hits += 1

            if _is_surgery_hue(frame):
                surgery_hue_hits += 1

    finally:
        cap.release()

    if valid_frames == 0:
        logger.warning("SceneClassifier: nenhum frame legivel em %s", video_path)
        return SceneType.UNKNOWN

    face_ratio = face_hits / valid_frames
    surgery_ratio = surgery_hue_hits / valid_frames

    logger.debug(
        "SceneClassifier %s: valid=%d face_ratio=%.2f surgery_hue_ratio=%.2f",
        video_path.name,
        valid_frames,
        face_ratio,
        surgery_ratio,
    )

    return _decide(face_ratio, surgery_ratio)


# -----------------------------------------------------------------------
# Helpers internos
# -----------------------------------------------------------------------


def _sample_indices(total_frames: int, num_samples: int) -> list[int]:
    """Retorna indices uniformemente distribuidos pelo video."""
    if num_samples <= 0:
        return []
    if num_samples >= total_frames:
        return list(range(total_frames))
    step = total_frames / num_samples
    return [int(i * step) for i in range(num_samples)]


def _load_face_detector():
    """Carrega MediaPipe Face Detection. Retorna None se indisponivel."""
    try:
        import mediapipe as mp

        face_detection = mp.solutions.face_detection
        detector = face_detection.FaceDetection(
            model_selection=0,  # modelo 0 = curta distancia, mais leve
            min_detection_confidence=0.5,
        )
        logger.debug("SceneClassifier: MediaPipe Face Detection carregado.")
        return detector
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "SceneClassifier: mediapipe indisponivel (%s). "
            "Heuristica de face desabilitada; classificacao usara apenas HSV.",
            exc,
        )
        return None


def _has_face(frame: np.ndarray, detector) -> bool:
    """Retorna True se o detector encontrar pelo menos uma face no frame.

    Se `detector` for None (mediapipe indisponivel), sempre retorna False.
    """
    if detector is None:
        return False
    rgb = frame[..., ::-1]  # BGR -> RGB
    result = detector.process(rgb)
    return bool(result.detections)


def _is_surgery_hue(frame: np.ndarray) -> bool:
    """Retorna True se a assinatura HSV do frame e tipica de cirurgia laparoscopica.

    Criterios:
    - Matiz medio na faixa vermelho-rosado escuro ([0..15] ou [165..179] no OpenCV).
    - Saturacao media acima de `_SURGERY_SAT_MIN` (campo cirurgico e saturado).
    """
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)

    mean_hue = float(np.mean(hue))
    mean_sat = float(np.mean(sat))

    hue_in_low = _SURGERY_HUE_LOW_MIN <= mean_hue <= _SURGERY_HUE_LOW_MAX
    hue_in_high = _SURGERY_HUE_HIGH_MIN <= mean_hue <= _SURGERY_HUE_HIGH_MAX
    hue_ok = hue_in_low or hue_in_high
    sat_ok = mean_sat >= _SURGERY_SAT_MIN

    return hue_ok and sat_ok


def _decide(face_ratio: float, surgery_ratio: float) -> SceneType:
    """Aplica as regras de decisao final.

    Regras (em ordem de prioridade):
    1. Maioria de frames com face E saturacao moderada (nao cirurgia) -> consulta.
    2. Poucos frames com face E hue tipico de cirurgia -> cirurgia.
    3. Caso contrario -> misto.
    """
    has_faces = face_ratio >= _FACE_MAJORITY_RATIO
    is_surgery_hue = surgery_ratio >= _SURGERY_HUE_RATIO

    if has_faces and not is_surgery_hue:
        return SceneType.CONSULTATION
    if not has_faces and is_surgery_hue:
        return SceneType.SURGERY
    if has_faces and is_surgery_hue:
        # Rosto presente mas paleta cirurgica (ex: cirurgiao em campo) -> misto
        return SceneType.MIXED
    # Sem faces e sem paleta cirurgica clara -> misto (video ambiguo)
    return SceneType.MIXED
