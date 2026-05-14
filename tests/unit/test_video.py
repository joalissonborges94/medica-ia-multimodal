"""Testes smoke do pipeline de video.

Sem download de pesos ou inferencia real. Mocks isolam dependencias
externas (`YOLO`, `mediapipe`, `fer`) para manter o teste rapido e offline.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from src.video import (
    AzureVideoIndexerClient,
    BleedingDetector,
    BoundingBox,
    Detection,
    EmotionScore,
    FacialEmotionDetector,
    PoseEstimator,
    PoseLandmark,
    VideoEvent,
    VideoPipeline,
    ensure_yolo_weights,
)
from src.video.detector import DEFAULT_STUB_MODEL

# ---------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_bounding_box_calcula_dimensoes():
    bbox = BoundingBox(x1=10, y1=20, x2=110, y2=220)
    assert bbox.width == 100
    assert bbox.height == 200


@pytest.mark.smoke
def test_detection_valida_confianca_em_range():
    bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    det = Detection(class_id=0, class_name="person", confidence=0.85, bbox=bbox)
    assert det.confidence == pytest.approx(0.85)
    assert det.class_name == "person"


@pytest.mark.smoke
def test_detection_rejeita_confianca_acima_de_um():
    bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    with pytest.raises(ValidationError):
        Detection(class_id=0, class_name="x", confidence=1.5, bbox=bbox)


@pytest.mark.smoke
def test_pose_landmark_aceita_visibility_no_range():
    lm = PoseLandmark(name="nose", x=0.5, y=0.5, z=0.0, visibility=0.9)
    assert lm.name == "nose"
    assert lm.visibility == pytest.approx(0.9)


@pytest.mark.smoke
def test_emotion_score_aceita_distribuicao():
    score = EmotionScore(label="happy", confidence=0.7, scores={"happy": 0.7, "sad": 0.3})
    assert score.label == "happy"
    assert score.scores["happy"] == pytest.approx(0.7)


@pytest.mark.smoke
def test_video_event_tem_defaults_vazios():
    event = VideoEvent(frame_index=0, timestamp_ms=0)
    assert event.detections == []
    assert event.pose_landmarks == []
    assert event.facial_emotion is None
    assert event.azure_metadata is None


# ---------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_bleeding_detector_inicializa_com_path_de_settings():
    detector = BleedingDetector()
    assert detector.weights_path.is_absolute()
    assert detector.confidence_threshold == 0.25
    assert detector._model is None


@pytest.mark.smoke
def test_bleeding_detector_aceita_threshold_custom(tmp_path):
    fake_weights = tmp_path / "stub.pt"
    fake_weights.write_bytes(b"fake")
    detector = BleedingDetector(weights_path=fake_weights, confidence_threshold=0.5)
    assert detector.confidence_threshold == 0.5
    assert detector.weights_path == fake_weights.resolve()


@pytest.mark.smoke
def test_ensure_yolo_weights_retorna_path_existente_sem_download(tmp_path):
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"fake")
    with patch("src.video.detector.YOLO") as mock_yolo:
        result = ensure_yolo_weights(weights)
    assert result == weights.resolve()
    mock_yolo.assert_not_called()


@pytest.mark.smoke
def test_default_stub_model_aponta_para_yolov8n():
    assert DEFAULT_STUB_MODEL == "yolov8n.pt"


# ---------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_pose_estimator_inicializa_lazy():
    estimator = PoseEstimator(min_detection_confidence=0.7)
    assert estimator.min_detection_confidence == 0.7
    assert estimator._pose is None


@pytest.mark.smoke
def test_pose_estimator_retorna_lista_vazia_quando_nao_detecta(monkeypatch):
    estimator = PoseEstimator()
    fake_pose = MagicMock()
    fake_pose.process.return_value = MagicMock(pose_landmarks=None)
    estimator._pose = fake_pose
    estimator._available = True
    estimator._load_attempted = True
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert estimator.estimate(frame) == []


# ---------------------------------------------------------------------
# Emocao facial
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_facial_emotion_detector_fallback_quando_fer_indisponivel(monkeypatch):
    detector = FacialEmotionDetector()

    def _fake_load():
        detector._available = False

    monkeypatch.setattr(detector, "load", _fake_load)
    detector._available = False
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


@pytest.mark.smoke
def test_facial_emotion_detector_parseia_saida_do_fer(monkeypatch):
    detector = FacialEmotionDetector()
    fake_fer = MagicMock()
    fake_fer.detect_emotions.return_value = [
        {
            "box": (10, 20, 100, 200),
            "emotions": {"happy": 0.7, "sad": 0.3},
        }
    ]
    detector._detector = fake_fer
    detector._available = True
    detector._load_attempted = True
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    scores = detector.detect(frame)
    assert len(scores) == 1
    assert scores[0].label == "happy"
    assert scores[0].confidence == pytest.approx(0.7)
    assert scores[0].bbox is not None
    assert scores[0].bbox.x1 == 10


# ---------------------------------------------------------------------
# Azure Video Indexer
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_azure_client_nao_configurado_retorna_none(tmp_path):
    client = AzureVideoIndexerClient()
    # No ambiente de teste, .env esta em branco -> nao configurado.
    assert client.is_configured is False
    fake_video = tmp_path / "fake.mp4"
    fake_video.touch()
    assert client.analyze(fake_video) is None


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


@pytest.mark.smoke
def test_video_pipeline_falha_quando_arquivo_nao_existe(tmp_path):
    pipeline = VideoPipeline(
        detector=MagicMock(spec=BleedingDetector),
        pose_estimator=MagicMock(spec=PoseEstimator),
        emotion_detector=MagicMock(spec=FacialEmotionDetector),
        azure_client=MagicMock(spec=AzureVideoIndexerClient),
    )
    with pytest.raises(FileNotFoundError):
        pipeline.process(tmp_path / "inexistente.mp4")
