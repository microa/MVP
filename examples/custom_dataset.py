"""
Custom dataset processing example for MVP detector.
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from mvp_detector import MVPDetector


def process_custom_dataset(video_dir: str, mv_root: str, output_root: str):
    """
    Process a custom dataset with MVP detector.
    
    Args:
        video_dir: Directory containing video files
        mv_root: Root directory containing motion vectors
        output_root: Root output directory
    """
    # Initialize detector
    detector = MVPDetector(
        model_id="google/owlv2-large-patch14-ensemble",
        device="cuda",
        confidence_threshold=0.5,
        single_class_fail_limit=2,
        area_scale_threshold=2.0
    )
    
    # Get list of video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files.sort()
    
    if not video_files:
        print("No video files found in directory")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    total_stats = {
        'total_videos': len(video_files),
        'total_frames': 0,
        'total_owl_detections': 0,
        'total_propagations': 0,
        'total_processing_time': 0,
        'successful_videos': 0
    }
    
    for i, video_file in enumerate(video_files, 1):
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_dir, video_file)
        mv_dir = os.path.join(mv_root, video_name)
        output_dir = os.path.join(output_root, video_name)
        
        print(f"\nProcessing video {i}/{len(video_files)}: {video_name}")
        
        # Check if motion vectors exist
        if not os.path.exists(mv_dir):
            print(f"  Skipping {video_name}: motion vectors not found")
            continue
        
        try:
            # Process video
            stats = detector.process_video(
                video_path=video_path,
                motion_vector_dir=mv_dir,
                output_dir=output_dir,
                extract_frames=True
            )
            
            # Accumulate statistics
            total_stats['total_frames'] += stats['total_frames']
            total_stats['total_owl_detections'] += stats['owl_detections']
            total_stats['total_propagations'] += stats['propagations']
            total_stats['total_processing_time'] += stats['processing_time']
            total_stats['successful_videos'] += 1
            
            print(f"  Completed: {stats['fps']:.2f} FPS")
            
        except Exception as e:
            print(f"  Error processing {video_name}: {e}")
            continue
    
    # Print summary
    print(f"\n=== Dataset Processing Summary ===")
    print(f"Total videos: {total_stats['total_videos']}")
    print(f"Successful videos: {total_stats['successful_videos']}")
    print(f"Total frames: {total_stats['total_frames']}")
    print(f"Total OWL detections: {total_stats['total_owl_detections']}")
    print(f"Total propagations: {total_stats['total_propagations']}")
    print(f"Total processing time: {total_stats['total_processing_time']:.2f}s")
    
    if total_stats['total_frames'] > 0:
        avg_fps = total_stats['total_frames'] / total_stats['total_processing_time']
        print(f"Average FPS: {avg_fps:.2f}")
        
        if total_stats['total_owl_detections'] > 0:
            efficiency = total_stats['total_propagations'] / total_stats['total_frames'] * 100
            print(f"Propagation efficiency: {efficiency:.1f}%")


def main():
    """Main function for custom dataset processing."""
    
    # Example paths (replace with your actual paths)
    video_dir = "path/to/your/videos"
    mv_root = "path/to/motion_vectors"
    output_root = "path/to/output"
    
    # Check if directories exist
    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        print("Please update the video_dir variable with a valid directory")
        return
    
    if not os.path.exists(mv_root):
        print(f"Motion vector root not found: {mv_root}")
        print("Please update the mv_root variable with a valid directory")
        return
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Process dataset
    process_custom_dataset(video_dir, mv_root, output_root)


if __name__ == "__main__":
    main()
