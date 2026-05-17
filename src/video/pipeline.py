"""Pipeline de video completo.

Recebe um caminho de video, amostra frames a uma taxa configuravel
(`target_fps`) e roda em cada frame: detector YOLO, MediaPipe Pose,
classificador de emocao facial e (quando configurado) Azure Video Indexer.
A saida e uma lista de `VideoEvent` com tudo agregado por frame.

Logica de gating por tipo de cena (ambos detector e emocao):

| Cena         | YOLO (instrumentos) | Emocao facial |
|--------------|---------------------|---------------|
| SURGERY      | rodando             | pulado        |
| CONSULTATION | pulado              | rodando       |
| MIXED        | rodando             | rodando       |
| UNKNOWN      | rodando             | rodando       |

Em CONSULTATION o YOLO custom (treinado em laparoscopia) gera falso
positivo em objetos clinicos comuns como agulhas, seringas, otoscopios,
classificando como Grasper/L-hook. Pular eleva precisao sem perder
deteccoes reais (cenas reais de cirurgia caem em SURGERY ou MIXED).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from src.video.azure_video import AzureVideoIndexerClient
from src.video.detector import BleedingDetector
from src.video.emotion import FacialEmotionClassifierProtocol, get_facial_emotion_classifier
from src.video.pose import PoseEstimator
from src.video.scene_classifier import SceneType, classify_scene_type
from src.video.types import VideoEvent

logger = logging.getLogger(__name__)


class VideoPipeline:
    """Orquestra todos os modulos de analise de video frame a frame."""

    def __init__(
        self,
        target_fps: float = 1.0,
        detector: BleedingDetector | None = None,
        pose_estimator: PoseEstimator | None = None,
        emotion_classifier: FacialEmotionClassifierProtocol | None = None,
        azure_client: AzureVideoIndexerClient | None = None,
    ) -> None:
        """Configura o pipeline.

        Args:
            target_fps: quantos frames por segundo amostrar do video.
                Default 1 fps mantem custo baixo para videos longos.
            detector: instancia opcional de `BleedingDetector`.
            pose_estimator: instancia opcional de `PoseEstimator`.
            emotion_classifier: instancia opcional de classificador de emocao
                facial. Default usa `get_facial_emotion_classifier()` que
                seleciona GPT-4o vision (quando configurado) ou FER local.
            azure_client: instancia opcional de `AzureVideoIndexerClient`.
        """
        self.target_fps: float = target_fps
        self.detector: BleedingDetector = detector or BleedingDetector()
        self.pose_estimator: PoseEstimator = pose_estimator or PoseEstimator()
        self.emotion_classifier: FacialEmotionClassifierProtocol = (
            emotion_classifier or get_facial_emotion_classifier()
        )
        self.azure_client: AzureVideoIndexerClient = azure_client or AzureVideoIndexerClient()
        # Preenchido apos cada chamada a `process()`.
        self.last_scene_type: SceneType = SceneType.UNKNOWN

    def process(self, video_path: Path) -> list[VideoEvent]:
        """Processa um video e retorna a lista de `VideoEvent`.

        Args:
            video_path: caminho do arquivo de video.

        Returns:
            Lista de eventos, um por frame amostrado.

        Raises:
            FileNotFoundError: se o video nao existir.
            RuntimeError: se o OpenCV nao conseguir abrir o arquivo.
        """
        video_path = video_path.resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video nao encontrado: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Falha ao abrir video com OpenCV: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            sampling_step = max(1, int(round(video_fps / self.target_fps)))
            logger.info(
                "Processando %s: video_fps=%.2f, target_fps=%.2f, step=%d",
                video_path,
                video_fps,
                self.target_fps,
                sampling_step,
            )

            self.last_scene_type = classify_scene_type(video_path)
            logger.info("Tipo de cena identificado: %s", self.last_scene_type.value)

            # Emocao facial so faz sentido em cenas com rosto visivel.
            # Em cirurgia laparoscopica o campo cirurgico nao expoe faces,
            # entao pular GPT-vision evita custo e ruido desnecessarios.
            run_emotion = self.last_scene_type != SceneType.SURGERY
            if not run_emotion:
                logger.info(
                    "Cena SURGERY: classificacao de emocao facial ignorada."
                )

            # Deteccao YOLO so faz sentido em cenas cirurgicas. Em consulta,
            # o modelo (treinado em laparoscopia) gera falso positivo em
            # objetos comuns de exame fisico (agulhas, seringas, otoscopios)
            # classificando como Grasper/L-hook.
            run_detection = self.last_scene_type != SceneType.CONSULTATION
            if not run_detection:
                logger.info(
                    "Cena CONSULTATION: deteccao de instrumentos ignorada "
                    "(evita falso positivo do YOLO laparoscopico)."
                )

            azure_metadata = self.azure_client.analyze(video_path)

            events: list[VideoEvent] = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sampling_step != 0:
                    frame_idx += 1
                    continue

                timestamp_ms = int(round(frame_idx * 1000 / video_fps))
                detections = self.detector.predict(frame) if run_detection else []
                pose_landmarks = self.pose_estimator.estimate(frame)
                facial_emotion = (
                    self.emotion_classifier.classify(frame) if run_emotion else None
                )

                events.append(
                    VideoEvent(
                        frame_index=frame_idx,
                        timestamp_ms=timestamp_ms,
                        detections=detections,
                        pose_landmarks=pose_landmarks,
                        facial_emotion=facial_emotion,
                        azure_metadata=azure_metadata,
                    )
                )
                frame_idx += 1

            logger.info("Processamento concluido: %d eventos", len(events))
            return events
        finally:
            cap.release()
            self.pose_estimator.close()
