"""Pipeline de video: deteccao YOLO, pose MediaPipe, emocao facial e Azure."""

from src.video.azure_video import AzureVideoIndexerClient
from src.video.detector import BleedingDetector, ensure_yolo_weights
from src.video.emotion import FacialEmotionDetector
from src.video.pipeline import VideoPipeline
from src.video.pose import PoseEstimator
from src.video.types import (
    BoundingBox,
    Detection,
    EmotionScore,
    PoseLandmark,
    VideoEvent,
)

__all__ = [
    "AzureVideoIndexerClient",
    "BleedingDetector",
    "BoundingBox",
    "Detection",
    "EmotionScore",
    "FacialEmotionDetector",
    "PoseEstimator",
    "PoseLandmark",
    "VideoEvent",
    "VideoPipeline",
    "ensure_yolo_weights",
]
