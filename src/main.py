"""
Command line interface for MVP detector.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from mvp_detector import MVPDetector


def main():
    parser = argparse.ArgumentParser(description="MVP: Motion Vector Propagation for Zero-Shot Object Detection")
    
    # Input arguments
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--video_dir", type=str, help="Directory containing video files")
    parser.add_argument("--mv_dir", type=str, help="Directory containing motion vectors for single video")
    parser.add_argument("--mv_root", type=str, help="Root directory containing motion vectors for multiple videos")
    parser.add_argument("--output_dir", type=str, help="Output directory for single video")
    parser.add_argument("--output_root", type=str, help="Root output directory for multiple videos")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default="google/owlv2-large-patch14-ensemble",
                       help="OWL-ViT model identifier")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to run on")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Detection confidence threshold")
    
    # Processing arguments
    parser.add_argument("--single_class_fail_limit", type=int, default=2,
                       help="Max failures before switching from single-class mode")
    parser.add_argument("--area_scale_threshold", type=float, default=2.0,
                       help="Area scaling threshold for re-detection")
    parser.add_argument("--extract_frames", action="store_true", default=True,
                       help="Extract frames from video")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.video_path and args.video_dir:
        print("Error: Cannot specify both --video_path and --video_dir")
        sys.exit(1)
    
    if not args.video_path and not args.video_dir:
        print("Error: Must specify either --video_path or --video_dir")
        sys.exit(1)
    
    if args.video_path and not args.mv_dir:
        print("Error: Must specify --mv_dir when using --video_path")
        sys.exit(1)
    
    if args.video_dir and not args.mv_root:
        print("Error: Must specify --mv_root when using --video_dir")
        sys.exit(1)
    
    if args.video_path and not args.output_dir:
        print("Error: Must specify --output_dir when using --video_path")
        sys.exit(1)
    
    if args.video_dir and not args.output_root:
        print("Error: Must specify --output_root when using --video_dir")
        sys.exit(1)
    
    # Initialize detector
    detector = MVPDetector(
        model_id=args.model_id,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        single_class_fail_limit=args.single_class_fail_limit,
        area_scale_threshold=args.area_scale_threshold
    )
    
    if args.video_path:
        # Process single video
        print(f"Processing single video: {args.video_path}")
        stats = detector.process_video(
            video_path=args.video_path,
            motion_vector_dir=args.mv_dir,
            output_dir=args.output_dir,
            extract_frames=args.extract_frames
        )
        
        print(f"Processing completed:")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  OWL detections: {stats['owl_detections']}")
        print(f"  Propagations: {stats['propagations']}")
        print(f"  Processing time: {stats['processing_time']:.2f}s")
        print(f"  FPS: {stats['fps']:.2f}")
        
    else:
        # Process multiple videos
        print(f"Processing videos in directory: {args.video_dir}")
        
        video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
        video_files.sort()
        
        if not video_files:
            print("No video files found in directory")
            sys.exit(1)
        
        total_stats = {
            'total_videos': len(video_files),
            'total_frames': 0,
            'total_owl_detections': 0,
            'total_propagations': 0,
            'total_processing_time': 0
        }
        
        for i, video_file in enumerate(video_files, 1):
            video_name = os.path.splitext(video_file)[0]
            video_path = os.path.join(args.video_dir, video_file)
            mv_dir = os.path.join(args.mv_root, video_name)
            output_dir = os.path.join(args.output_root, video_name)
            
            print(f"\nProcessing video {i}/{len(video_files)}: {video_name}")
            
            try:
                stats = detector.process_video(
                    video_path=video_path,
                    motion_vector_dir=mv_dir,
                    output_dir=output_dir,
                    extract_frames=args.extract_frames
                )
                
                # Accumulate stats
                total_stats['total_frames'] += stats['total_frames']
                total_stats['total_owl_detections'] += stats['owl_detections']
                total_stats['total_propagations'] += stats['propagations']
                total_stats['total_processing_time'] += stats['processing_time']
                
            except Exception as e:
                print(f"Error processing {video_name}: {e}")
                continue
        
        # Print summary
        print(f"\n=== Processing Summary ===")
        print(f"Total videos: {total_stats['total_videos']}")
        print(f"Total frames: {total_stats['total_frames']}")
        print(f"Total OWL detections: {total_stats['total_owl_detections']}")
        print(f"Total propagations: {total_stats['total_propagations']}")
        print(f"Total processing time: {total_stats['total_processing_time']:.2f}s")
        
        if total_stats['total_frames'] > 0:
            avg_fps = total_stats['total_frames'] / total_stats['total_processing_time']
            print(f"Average FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
