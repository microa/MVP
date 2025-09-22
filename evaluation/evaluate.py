"""
Evaluation script for MVP detector on ImageNet VID dataset.
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.metrics import compute_map, build_confusion_matrix
from evaluation.visualization import plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate MVP detector results")
    parser.add_argument("--pred_dir", type=str, required=True,
                       help="Directory containing prediction results")
    parser.add_argument("--gt_dir", type=str, required=True,
                       help="Directory containing ground truth annotations")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for evaluation results")
    parser.add_argument("--iou_thresholds", type=float, nargs="+", 
                       default=[0.2, 0.3, 0.5],
                       help="IoU thresholds for evaluation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading predictions and ground truth...")
    
    # Load predictions and ground truth
    all_predictions, all_gts = load_dataset(args.pred_dir, args.gt_dir)
    
    if len(all_predictions) == 0 or len(all_gts) == 0:
        print("Error: No valid predictions or ground truth found")
        return
    
    print(f"Loaded {len(all_predictions)} predictions and {len(all_gts)} ground truth images")
    
    # Compute mAP
    print("Computing mAP...")
    map_results = compute_map(all_predictions, all_gts, args.iou_thresholds)
    
    # Print results
    print("\n=== Evaluation Results ===")
    for threshold in args.iou_thresholds:
        print(f"mAP@{threshold}: {map_results[threshold]:.6f}")
    
    # Compute mAP@[0.5:0.95]
    iou_range = np.arange(0.5, 1.0, 0.05)
    map_range = compute_map(all_predictions, all_gts, iou_range)
    map_5095 = np.mean([map_range[t] for t in iou_range])
    print(f"mAP@[0.5:0.95]: {map_5095:.6f}")
    
    # Build confusion matrix
    print("Building confusion matrix...")
    conf_matrix = build_confusion_matrix(all_predictions, all_gts, iou_thresh=0.5)
    
    # Save results
    results = {
        "map_results": map_results,
        "map_5095": map_5095,
        "num_predictions": len(all_predictions),
        "num_images": len(all_gts)
    }
    
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save confusion matrix
    conf_matrix_path = os.path.join(args.output_dir, "confusion_matrix.npy")
    np.save(conf_matrix_path, conf_matrix)
    
    # Plot confusion matrix
    plot_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(conf_matrix, plot_path)
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - evaluation_results.json")
    print(f"  - confusion_matrix.npy")
    print(f"  - confusion_matrix.png")


def load_dataset(pred_dir: str, gt_dir: str) -> Tuple[List, Dict]:
    """
    Load predictions and ground truth from directories.
    
    Args:
        pred_dir: Directory containing prediction results
        gt_dir: Directory containing ground truth annotations
        
    Returns:
        Tuple of (predictions, ground_truth)
    """
    all_predictions = []
    all_gts = {}
    
    # Get list of video directories
    video_dirs = [d for d in os.listdir(pred_dir) if os.path.isdir(os.path.join(pred_dir, d))]
    video_dirs.sort()
    
    for video_dir in video_dirs:
        pred_video_dir = os.path.join(pred_dir, video_dir)
        gt_video_dir = os.path.join(gt_dir, video_dir)
        
        if not os.path.exists(gt_video_dir):
            continue
        
        # Get prediction files
        pred_files = []
        for f in os.listdir(pred_video_dir):
            if f.endswith('.npy') and f.startswith('frame_'):
                pred_files.append(f)
        pred_files.sort()
        
        for pred_file in pred_files:
            # Extract frame index
            frame_idx = int(pred_file.split('_')[1].split('.')[0])
            
            # Load predictions
            pred_path = os.path.join(pred_video_dir, pred_file)
            preds = load_predictions(pred_path)
            
            # Load ground truth
            gt_path = os.path.join(gt_video_dir, f"{frame_idx:06d}.json")
            if not os.path.exists(gt_path):
                continue
            
            gts, W, H = load_ground_truth(gt_path)
            
            if len(gts) == 0 and len(preds) == 0:
                continue
            
            # Add to dataset
            image_id = f"{video_dir}_{frame_idx:06d}"
            all_gts[image_id] = gts
            
            for pred in preds:
                all_predictions.append((
                    image_id,
                    pred['class_index'],
                    pred['score'],
                    pred['x_center'],
                    pred['y_center'],
                    pred['width'],
                    pred['height']
                ))
    
    return all_predictions, all_gts


def load_predictions(pred_path: str) -> List[Dict]:
    """Load predictions from .npy file."""
    preds = np.load(pred_path, allow_pickle=True)
    return preds.tolist() if isinstance(preds, np.ndarray) else preds


def load_ground_truth(gt_path: str) -> Tuple[List, int, int]:
    """Load ground truth from JSON file."""
    with open(gt_path, 'r') as f:
        data = json.load(f)
    
    W = data["size"]["width"]
    H = data["size"]["height"]
    
    gts = []
    for obj in data["objects"]:
        if "yolo_bbox" in obj:
            bbox = obj["yolo_bbox"]
            gts.append([
                obj.get("synset", ""),
                bbox["x_center"],
                bbox["y_center"], 
                bbox["width"],
                bbox["height"],
                False  # matched flag
            ])
    
    return gts, W, H


if __name__ == "__main__":
    main()
