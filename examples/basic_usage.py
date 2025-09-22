"""
Basic usage example for MVP detector.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from mvp_detector import MVPDetector


def main():
    """Basic usage example."""
    
    # Initialize detector
    print("Initializing MVP detector...")
    detector = MVPDetector(
        model_id="google/owlv2-large-patch14-ensemble",
        device="cuda",  # or "cpu" if no GPU available
        confidence_threshold=0.5
    )
    
    # Example paths (replace with your actual paths)
    video_path = "path/to/your/video.mp4"
    motion_vector_dir = "path/to/motion_vectors/video_name"
    output_dir = "path/to/output"
    
    # Check if paths exist
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Please update the video_path variable with a valid video file")
        return
    
    if not os.path.exists(motion_vector_dir):
        print(f"Motion vector directory not found: {motion_vector_dir}")
        print("Please update the motion_vector_dir variable with a valid directory")
        return
    
    # Process video
    print(f"Processing video: {video_path}")
    stats = detector.process_video(
        video_path=video_path,
        motion_vector_dir=motion_vector_dir,
        output_dir=output_dir,
        extract_frames=True
    )
    
    # Print results
    print("\n=== Processing Results ===")
    print(f"Total frames: {stats['total_frames']}")
    print(f"OWL detections: {stats['owl_detections']}")
    print(f"Motion propagations: {stats['propagations']}")
    print(f"Processing time: {stats['processing_time']:.2f} seconds")
    print(f"FPS: {stats['fps']:.2f}")
    
    # Calculate efficiency
    if stats['total_frames'] > 0:
        efficiency = stats['propagations'] / stats['total_frames'] * 100
        print(f"Propagation efficiency: {efficiency:.1f}%")


if __name__ == "__main__":
    main()
