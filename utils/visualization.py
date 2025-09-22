"""
Visualization utilities for MVP detector.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional


def visualize_motion_vectors(mv_path: str, output_path: str, 
                           image_path: Optional[str] = None) -> None:
    """
    Visualize motion vectors as arrows on an image.
    
    Args:
        mv_path: Path to motion vectors .npy file
        output_path: Path to save visualization
        image_path: Optional path to background image
    """
    # Load motion vectors
    mvs = np.load(mv_path, allow_pickle=True)
    
    if image_path and os.path.exists(image_path):
        # Load background image
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
    else:
        # Create blank image
        W, H = 640, 480
        image = Image.new("RGB", (W, H), "white")
    
    draw = ImageDraw.Draw(image)
    
    # Draw motion vectors
    for mv in mvs:
        if mv[0] != -1:  # Skip invalid motion vectors
            continue
        
        ref_cx = mv[3]
        ref_cy = mv[4]
        cur_cx = mv[5]
        cur_cy = mv[6]
        
        # Calculate displacement
        dx = cur_cx - ref_cx
        dy = cur_cy - ref_cy
        
        # Draw arrow
        if abs(dx) > 1 or abs(dy) > 1:  # Only draw significant motion
            start_x = int(ref_cx)
            start_y = int(ref_cy)
            end_x = int(cur_cx)
            end_y = int(cur_cy)
            
            # Draw line
            draw.line([start_x, start_y, end_x, end_y], fill="red", width=2)
            
            # Draw arrowhead
            arrow_length = 10
            angle = np.arctan2(dy, dx)
            
            arrow_x1 = end_x - arrow_length * np.cos(angle - np.pi/6)
            arrow_y1 = end_y - arrow_length * np.sin(angle - np.pi/6)
            arrow_x2 = end_x - arrow_length * np.cos(angle + np.pi/6)
            arrow_y2 = end_y - arrow_length * np.sin(angle + np.pi/6)
            
            draw.line([end_x, end_y, int(arrow_x1), int(arrow_y1)], fill="red", width=2)
            draw.line([end_x, end_y, int(arrow_x2), int(arrow_y2)], fill="red", width=2)
    
    # Save image
    image.save(output_path)
    print(f"Motion vector visualization saved to: {output_path}")


def visualize_detections(image_path: str, detections: List[Dict], 
                        output_path: str, show_scores: bool = True) -> None:
    """
    Visualize object detections on an image.
    
    Args:
        image_path: Path to input image
        detections: List of detection dictionaries
        output_path: Path to save visualization
        show_scores: Whether to show confidence scores
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    W, H = image.size
    
    # Draw detections
    for i, det in enumerate(detections):
        # Convert normalized coordinates to pixel coordinates
        x1 = int((det['x_center'] - det['width'] / 2) * W)
        y1 = int((det['y_center'] - det['height'] / 2) * H)
        x2 = int((det['x_center'] + det['width'] / 2) * W)
        y2 = int((det['y_center'] + det['height'] / 2) * H)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Draw label
        if show_scores:
            label = f"{det['class_name']}: {det['score']:.2f}"
        else:
            label = det['class_name']
        
        # Draw text background
        text_bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(text_bbox, fill="red")
        
        # Draw text
        draw.text((x1, y1 - 20), label, fill="white", font=font)
    
    # Save image
    image.save(output_path)
    print(f"Detection visualization saved to: {output_path}")


def create_video_visualization(video_path: str, detections_dir: str, 
                              output_path: str, fps: int = 30) -> None:
    """
    Create a video visualization of detections.
    
    Args:
        video_path: Path to input video
        detections_dir: Directory containing detection results
        output_path: Path to save output video
        fps: Output video FPS
    """
    import cv2
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Load detections for this frame
        det_file = os.path.join(detections_dir, f"frame_{frame_idx:05d}.npy")
        if os.path.exists(det_file):
            detections = np.load(det_file, allow_pickle=True)
            
            # Draw detections
            for det in detections:
                x1 = int((det['x_center'] - det['width'] / 2) * width)
                y1 = int((det['y_center'] - det['height'] / 2) * height)
                x2 = int((det['x_center'] + det['width'] / 2) * width)
                y2 = int((det['y_center'] + det['height'] / 2) * height)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw label
                label = f"{det['class_name']}: {det['score']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Write frame
        out.write(frame)
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video visualization saved to: {output_path}")


def plot_fps_comparison(results: Dict[str, float], output_path: str) -> None:
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
    print(f"FPS comparison plot saved to: {output_path}")


def plot_accuracy_comparison(results: Dict[str, Dict[str, float]], output_path: str) -> None:
    """
    Plot accuracy comparison between different methods.
    
    Args:
        results: Dictionary with method names and accuracy metrics
        output_path: Path to save the plot
    """
    methods = list(results.keys())
    metrics = ['mAP@0.2', 'mAP@0.3', 'mAP@0.5', 'mAP@[0.5:0.95]']
    
    x = np.arange(len(methods))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        values = [results[method].get(metric, 0) for method in methods]
        ax.bar(x + i * width, values, width, label=metric)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy comparison plot saved to: {output_path}")
