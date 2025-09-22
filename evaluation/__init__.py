"""
Evaluation module for MVP detector.
"""

from .evaluate import main as evaluate
from .metrics import compute_map, build_confusion_matrix
from .visualization import plot_confusion_matrix, plot_detection_results

__all__ = [
    "evaluate",
    "compute_map", 
    "build_confusion_matrix",
    "plot_confusion_matrix",
    "plot_detection_results"
]
