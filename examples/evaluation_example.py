"""
Evaluation example for MVP detector.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.evaluate import main as evaluate


def main():
    """Evaluation example."""
    
    # Example paths (replace with your actual paths)
    pred_dir = "path/to/prediction_results"
    gt_dir = "path/to/ground_truth"
    output_dir = "path/to/evaluation_output"
    
    # Check if directories exist
    if not os.path.exists(pred_dir):
        print(f"Prediction directory not found: {pred_dir}")
        print("Please update the pred_dir variable with a valid directory")
        return
    
    if not os.path.exists(gt_dir):
        print(f"Ground truth directory not found: {gt_dir}")
        print("Please update the gt_dir variable with a valid directory")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    print("Running evaluation...")
    
    # Set up command line arguments
    sys.argv = [
        "evaluate.py",
        "--pred_dir", pred_dir,
        "--gt_dir", gt_dir,
        "--output_dir", output_dir,
        "--iou_thresholds", "0.2", "0.3", "0.5"
    ]
    
    # Run evaluation
    evaluate()


if __name__ == "__main__":
    main()
