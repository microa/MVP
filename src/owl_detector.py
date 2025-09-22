"""
OWL-ViT detector integration for zero-shot object detection.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from transformers import Owlv2Processor, Owlv2ForObjectDetection


class OWLDetector:
    """
    OWL-ViT based zero-shot object detector.
    """
    
    def __init__(self, 
                 model_id: str = "google/owlv2-large-patch14-ensemble",
                 device: str = "cuda",
                 confidence_threshold: float = 0.5):
        """
        Initialize OWL-ViT detector.
        
        Args:
            model_id: OWL-ViT model identifier
            device: Device to run on
            confidence_threshold: Detection confidence threshold
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        
        # ImageNet VID classes
        self.imagenet_vid_classes = [
            "airplane", "antelope", "bear", "bicycle", "bird", "bus", "car", "cattle", 
            "dog", "domestic cat", "elephant", "fox", "giant panda", "hamster", "horse", 
            "lion", "lizard", "monkey", "motorcycle", "rabbit", "red panda", "sheep", 
            "snake", "squirrel", "tiger", "train", "turtle", "watercraft", "whale", "zebra"
        ]
        
        # Load model
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
    
    def detect(self, 
               image_path: str, 
               single_class_name: Optional[str] = None) -> List[Dict]:
        """
        Perform zero-shot object detection on an image.
        
        Args:
            image_path: Path to input image
            single_class_name: If provided, only detect this class
            
        Returns:
            List of detection dictionaries
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        
        # Build text queries
        text_queries = self._build_text_queries(single_class_name)
        
        # Process inputs
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )
        
        # Extract detections
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        labels = results[0]["labels"]
        
        # Convert to our format
        detections = []
        class_to_index = {txt: i for i, txt in enumerate(text_queries[0])}
        
        for box, score, label_idx in zip(boxes, scores, labels):
            box = [round(coord.item(), 2) for coord in box]
            score = round(score.item(), 3)
            label_text = text_queries[0][label_idx]
            class_name = label_text.replace("a photo of ", "")
            
            # Convert to normalized coordinates
            x_min, y_min, x_max, y_max = box
            w_box = x_max - x_min
            h_box = y_max - y_min
            xc = (x_min + x_max) / 2
            yc = (y_min + y_max) / 2
            
            xc_norm = xc / W
            yc_norm = yc / H
            w_norm = w_box / W
            h_norm = h_box / H
            
            # Clamp coordinates
            xc_norm, yc_norm, w_norm, h_norm = self._clamp_bbox(xc_norm, yc_norm, w_norm, h_norm)
            
            detections.append({
                'class_index': class_to_index.get(label_text, -1),
                'class_name': class_name,
                'x_center': round(xc_norm, 6),
                'y_center': round(yc_norm, 6),
                'width': round(w_norm, 6),
                'height': round(h_norm, 6),
                'score': score
            })
        
        return detections
    
    def _build_text_queries(self, single_class_name: Optional[str] = None) -> List[List[str]]:
        """
        Build text queries for OWL-ViT.
        
        Args:
            single_class_name: If provided, only query this class
            
        Returns:
            List of text query lists
        """
        if single_class_name:
            return [[f"a photo of {single_class_name}"]]
        else:
            queries = [f"a photo of {c}" for c in self.imagenet_vid_classes]
            return [queries]
    
    def _clamp_bbox(self, xc: float, yc: float, w: float, h: float) -> tuple:
        """
        Clamp bounding box coordinates to valid range.
        
        Args:
            xc, yc, w, h: Normalized bounding box coordinates
            
        Returns:
            Clamped coordinates
        """
        # Clamp width and height
        if w > 1:
            w = 1
        if h > 1:
            h = 1
        
        # Clamp center coordinates
        half_w = w / 2
        half_h = h / 2
        xc = max(half_w, min(1.0 - half_w, xc))
        yc = max(half_h, min(1.0 - half_h, yc))
        
        return xc, yc, w, h
