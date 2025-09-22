"""
MVP: Motion Vector Propagation for Zero-Shot Object Detection

A framework for efficient zero-shot object detection in videos using motion vectors
and OWL-ViT for object detection.
"""

from .mvp_detector import MVPDetector
from .owl_detector import OWLDetector
from .motion_analyzer import MotionAnalyzer

__version__ = "1.0.0"
__author__ = "Binhua Huang"
__email__ = "binhua.huang@example.com"

__all__ = [
    "MVPDetector",
    "OWLDetector", 
    "MotionAnalyzer"
]
