"""
Data utilities for MVP detector.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image


def load_imagenet_vid_classes() -> List[str]:
    """Load ImageNet VID class names."""
    return [
        "airplane", "antelope", "bear", "bicycle", "bird", "bus", "car", "cattle",
        "dog", "domestic cat", "elephant", "fox", "giant panda", "hamster", "horse",
        "lion", "lizard", "monkey", "motorcycle", "rabbit", "red panda", "sheep",
        "snake", "squirrel", "tiger", "train", "turtle", "watercraft", "whale", "zebra"
    ]


def load_synset_mapping() -> Dict[str, int]:
    """Load synset to class index mapping."""
    return {
        "n02691156": 0,  "n02419796": 1,  "n02131653": 2,  "n02834778": 3,  "n01503061": 4,
        "n02924116": 5,  "n02958343": 6,  "n02402425": 7,  "n02084071": 8,  "n02121808": 9,
        "n02503517": 10, "n02118333": 11, "n02510455": 12, "n02342885": 13, "n02374451": 14,
        "n02129165": 15, "n01674464": 16, "n02484322": 17, "n03790512": 18, "n02324045": 19,
        "n02509815": 20, "n02411705": 21, "n01726692": 22, "n02355227": 23, "n02129604": 24,
        "n04468005": 25, "n01662784": 26, "n04530566": 27, "n02062744": 28, "n02391049": 29,
    }


def load_classname_mapping() -> Dict[str, int]:
    """Load class name to class index mapping."""
    return {
        "airplane": 0, "antelope": 1, "bear": 2, "bicycle": 3, "bird": 4, "bus": 5, "car": 6,
        "cattle": 7, "dog": 8, "domestic cat": 9, "elephant": 10, "fox": 11, "giant panda": 12,
        "hamster": 13, "horse": 14, "lion": 15, "lizard": 16, "monkey": 17, "motorcycle": 18,
        "rabbit": 19, "red panda": 20, "sheep": 21, "snake": 22, "squirrel": 23, "tiger": 24,
        "train": 25, "turtle": 26, "watercraft": 27, "whale": 28, "zebra": 29
    }


def normalize_class_name(class_name: str) -> str:
    """
    Normalize class name for consistent matching.
    
    Args:
        class_name: Input class name
        
    Returns:
        Normalized class name
    """
    if not isinstance(class_name, str):
        return ""
    
    # Convert to lowercase and replace separators
    normalized = class_name.strip().lower().replace("_", " ").replace("-", " ")
    
    # Remove extra whitespace
    normalized = " ".join(normalized.split())
    
    # Handle common variations
    if normalized == "domesticcat":
        normalized = "domestic cat"
    elif normalized in ["motor bike", "motorbike"]:
        normalized = "motorcycle"
    
    return normalized


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
    """
    Convert YOLO format to xyxy format.
    
    Args:
        xc, yc, w, h: Normalized YOLO coordinates
        img_w, img_h: Image dimensions
        
    Returns:
        [x1, y1, x2, y2] in pixel coordinates
    """
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> List[float]:
    """
    Convert xyxy format to YOLO format.
    
    Args:
        x1, y1, x2, y2: Pixel coordinates
        img_w, img_h: Image dimensions
        
    Returns:
        [xc, yc, w, h] in normalized coordinates
    """
    w_box = x2 - x1
    h_box = y2 - y1
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    
    return [xc / img_w, yc / img_h, w_box / img_w, h_box / img_h]


def clamp_bbox(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """
    Clamp bounding box coordinates to valid range.
    
    Args:
        xc, yc, w, h: Normalized bounding box coordinates
        
    Returns:
        Clamped coordinates
    """
    # Clamp width and height
    w = min(w, 1.0)
    h = min(h, 1.0)
    
    # Clamp center coordinates
    half_w = w / 2
    half_h = h / 2
    xc = max(half_w, min(1.0 - half_w, xc))
    yc = max(half_h, min(1.0 - half_h, yc))
    
    return xc, yc, w, h


def load_ground_truth(gt_path: str) -> Tuple[List, int, int]:
    """
    Load ground truth annotations from JSON file.
    
    Args:
        gt_path: Path to ground truth JSON file
        
    Returns:
        Tuple of (annotations, width, height)
    """
    with open(gt_path, 'r') as f:
        data = json.load(f)
    
    W = data["size"]["width"]
    H = data["size"]["height"]
    
    annotations = []
    for obj in data["objects"]:
        if "yolo_bbox" in obj:
            bbox = obj["yolo_bbox"]
            annotations.append([
                obj.get("synset", ""),
                bbox["x_center"],
                bbox["y_center"],
                bbox["width"],
                bbox["height"],
                False  # matched flag
            ])
    
    return annotations, W, H


def load_detections(det_path: str) -> List[Dict]:
    """
    Load detection results from .npy file.
    
    Args:
        det_path: Path to detection .npy file
        
    Returns:
        List of detection dictionaries
    """
    detections = np.load(det_path, allow_pickle=True)
    return detections.tolist() if isinstance(detections, np.ndarray) else detections


def save_detections(detections: List[Dict], det_path: str) -> None:
    """
    Save detection results to .npy file.
    
    Args:
        detections: List of detection dictionaries
        det_path: Path to save detections
    """
    np.save(det_path, detections)


def get_image_resolution(image_path: str) -> Tuple[int, int]:
    """
    Get image resolution.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (width, height)
    """
    with Image.open(image_path) as img:
        return img.size


def create_video_info(video_path: str, output_dir: str) -> None:
    """
    Create video information file.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    
    info_path = os.path.join(output_dir, "video_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
