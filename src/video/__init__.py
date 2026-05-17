"""Pipeline de video: deteccao YOLO, pose MediaPipe, emocao facial e Azure."""

from src.video.azure_openai_vision import AzureOpenAIVisionEmotion
from src.video.azure_video import AzureVideoIndexerClient
from src.video.detector import BleedingDetector, ensure_yolo_weights
from src.video.emotion import (
    FacialEmotionClassifierProtocol,
    FacialEmotionDetector,
    get_facial_emotion_classifier,
)
from src.video.pipeline import VideoPipeline
from src.video.pose import PoseEstimator, PostureCategory, classify_posture
from src.video.scene_classifier import SceneType, classify_scene_type
from src.video.types import (
    BoundingBox,
    Detection,
    EmotionScore,
    PoseLandmark,
    VideoEvent,
)

__all__ = [
    "AzureOpenAIVisionEmotion",
    "AzureVideoIndexerClient",
    "BleedingDetector",
    "BoundingBox",
    "Detection",
    "EmotionScore",
    "FacialEmotionClassifierProtocol",
    "FacialEmotionDetector",
    "PoseEstimator",
    "PoseLandmark",
    "PostureCategory",
    "SceneType",
    "VideoEvent",
    "VideoPipeline",
    "classify_posture",
    "classify_scene_type",
    "ensure_yolo_weights",
    "get_facial_emotion_classifier",
]
