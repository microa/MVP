"""
Visualization utilities for evaluation results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def plot_confusion_matrix(conf_matrix: np.ndarray, output_path: str):
    """
    Plot confusion matrix and save to file.
    
    Args:
        conf_matrix: Confusion matrix as numpy array
        output_path: Path to save the plot
    """
    # ImageNet VID class names
    class_names = [
        "airplane", "antelope", "bear", "bicycle", "bird", "bus", "car", "cattle",
        "dog", "domestic cat", "elephant", "fox", "giant panda", "hamster", "horse",
        "lion", "lizard", "monkey", "motorcycle", "rabbit", "red panda", "sheep",
        "snake", "squirrel", "tiger", "train", "turtle", "watercraft", "whale", "zebra"
    ]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot raw confusion matrix
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt="d", 
                xticklabels=class_names, 
                yticklabels=class_names,
                cmap="YlGnBu", 
                square=True, 
                cbar_kws={"shrink": 0.8}, 
                annot_kws={"size": 6})
    
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Ground Truth", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save normalized version
    norm_path = output_path.replace('.png', '_normalized.png')
    plot_confusion_matrix_normalized(conf_matrix, norm_path, class_names)


def plot_confusion_matrix_normalized(conf_matrix: np.ndarray, output_path: str, class_names: List[str]):
    """
    Plot normalized confusion matrix and save to file.
    
    Args:
        conf_matrix: Confusion matrix as numpy array
        output_path: Path to save the plot
        class_names: List of class names
    """
    # Normalize by row (ground truth)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    conf_matrix_norm = conf_matrix / row_sums
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot normalized confusion matrix
    sns.heatmap(conf_matrix_norm, 
                annot=True, 
                fmt=".2f", 
                xticklabels=class_names, 
                yticklabels=class_names,
                cmap="YlGnBu", 
                square=True, 
                cbar_kws={"shrink": 0.8}, 
                annot_kws={"size": 6})
    
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Ground Truth", fontsize=12)
    plt.title("Confusion Matrix (Normalized)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_detection_results(image_path: str, detections: List[dict], output_path: str):
    """
    Plot detection results on an image.
    
    Args:
        image_path: Path to input image
        detections: List of detection dictionaries
        output_path: Path to save the plot
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw detections
    for det in detections:
        # Convert normalized coordinates to pixel coordinates
        W, H = image.size
        x1 = int((det['x_center'] - det['width'] / 2) * W)
        y1 = int((det['y_center'] - det['height'] / 2) * H)
        x2 = int((det['x_center'] + det['width'] / 2) * W)
        y2 = int((det['y_center'] + det['height'] / 2) * H)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Draw label
        label = f"{det['class_name']}: {det['score']:.2f}"
        draw.text((x1, y1 - 20), label, fill="red", font=font)
    
    # Save image
    image.save(output_path)


def plot_fps_comparison(results: dict, output_path: str):
    """
    Plot FPS comparison between different methods.
    
    Args:
        results: Dictionary with method names and FPS values
        output_path: Path to save the plot
    """
    methods = list(results.keys())
    fps_values = list(results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, fps_values, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, fps in zip(bars, fps_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{fps:.1f}', ha='center', va='bottom')
    
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('FPS', fontsize=12)
    plt.title('FPS Comparison', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
