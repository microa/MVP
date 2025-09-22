"""
Evaluation metrics for object detection.
"""

import numpy as np
from typing import List, Dict, Tuple


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def voc_ap(rec: List[float], prec: List[float]) -> float:
    """
    Compute VOC-style Average Precision.
    
    Args:
        rec: Recall values
        prec: Precision values
        
    Returns:
        Average Precision
    """
    mrec = [0.0] + list(rec) + [1.0]
    mpre = [0.0] + list(prec) + [0.0]
    
    # Compute precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    # Compute area under curve
    ap = 0.0
    for i in range(len(mrec) - 1):
        ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    
    return ap


def compute_map(predictions: List, ground_truth: Dict, iou_thresholds: List[float]) -> Dict[float, float]:
    """
    Compute mean Average Precision (mAP) at different IoU thresholds.
    
    Args:
        predictions: List of (image_id, class_id, score, x1, y1, x2, y2)
        ground_truth: Dict mapping image_id to list of [class_id, x1, y1, x2, y2, matched]
        iou_thresholds: List of IoU thresholds
        
    Returns:
        Dict mapping IoU threshold to mAP value
    """
    # Group predictions by class
    num_classes = 30  # ImageNet VID has 30 classes
    preds_by_class = [[] for _ in range(num_classes)]
    
    for (img_id, class_id, score, x1, y1, x2, y2) in predictions:
        if 0 <= class_id < num_classes:
            preds_by_class[class_id].append((img_id, score, [x1, y1, x2, y2]))
    
    # Sort predictions by score (descending)
    for class_id in range(num_classes):
        preds_by_class[class_id].sort(key=lambda x: x[1], reverse=True)
    
    results = {}
    
    for threshold in iou_thresholds:
        sum_ap = 0.0
        valid_classes = 0
        
        for class_id in range(num_classes):
            # Count ground truth objects for this class
            gt_count = 0
            for img_id, gts in ground_truth.items():
                for gt in gts:
                    if len(gt) >= 5 and gt[0] == class_id:
                        gt_count += 1
            
            if gt_count == 0:
                continue
            
            # Reset matched flags
            for img_id, gts in ground_truth.items():
                for i, gt in enumerate(gts):
                    if len(gt) >= 6:
                        gt[5] = False
            
            # Compute AP for this class
            preds = preds_by_class[class_id]
            tp = [0] * len(preds)
            fp = [0] * len(preds)
            
            for i, (img_id, score, pred_box) in enumerate(preds):
                if img_id not in ground_truth:
                    fp[i] = 1
                    continue
                
                # Find best matching ground truth
                best_iou = 0.0
                best_gt_idx = -1
                
                for j, gt in enumerate(ground_truth[img_id]):
                    if len(gt) < 6 or gt[0] != class_id or gt[5]:  # matched
                        continue
                    
                    # Convert YOLO format to xyxy
                    gt_box = yolo_to_xyxy(gt[1], gt[2], gt[3], gt[4], 1, 1)  # normalized
                    iou = compute_iou(pred_box, gt_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= threshold and best_gt_idx != -1:
                    tp[i] = 1
                    ground_truth[img_id][best_gt_idx][5] = True  # mark as matched
                else:
                    fp[i] = 1
            
            # Compute precision and recall
            tp_cumsum = np.cumsum(tp).astype(float)
            fp_cumsum = np.cumsum(fp).astype(float)
            
            rec = tp_cumsum / float(gt_count)
            prec = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
            
            # Compute AP
            ap = voc_ap(rec, prec)
            sum_ap += ap
            valid_classes += 1
        
        results[threshold] = sum_ap / valid_classes if valid_classes > 0 else 0.0
    
    return results


def build_confusion_matrix(predictions: List, ground_truth: Dict, iou_thresh: float = 0.5) -> np.ndarray:
    """
    Build confusion matrix for object detection.
    
    Args:
        predictions: List of (image_id, class_id, score, x1, y1, x2, y2)
        ground_truth: Dict mapping image_id to list of [class_id, x1, y1, x2, y2, matched]
        iou_thresh: IoU threshold for matching
        
    Returns:
        Confusion matrix as numpy array
    """
    num_classes = 30
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # Sort predictions by score
    preds_sorted = sorted(predictions, key=lambda x: x[2], reverse=True)
    
    # Reset matched flags
    for img_id, gts in ground_truth.items():
        for i, gt in enumerate(gts):
            if len(gt) >= 6:
                gt[5] = False
    
    for (img_id, pred_class, score, x1, y1, x2, y2) in preds_sorted:
        if img_id not in ground_truth:
            continue
        
        pred_box = [x1, y1, x2, y2]
        gts = ground_truth[img_id]
        
        # Find best matching ground truth
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt in enumerate(gts):
            if len(gt) < 6 or gt[5]:  # matched
                continue
            
            # Convert YOLO format to xyxy
            gt_box = yolo_to_xyxy(gt[1], gt[2], gt[3], gt[4], 1, 1)  # normalized
            iou = compute_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_thresh and best_gt_idx != -1:
            gt_class = gts[best_gt_idx][0]
            gts[best_gt_idx][5] = True  # mark as matched
            
            if 0 <= gt_class < num_classes and 0 <= pred_class < num_classes:
                conf_matrix[gt_class, pred_class] += 1
    
    return conf_matrix


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
    """Convert YOLO format to xyxy format."""
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]
