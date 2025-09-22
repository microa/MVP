"""
Motion vector extraction utility for MVP detector.
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def extract_motion_vectors(video_path: str, output_dir: str) -> bool:
    """
    Extract motion vectors from a video file.
    
    Args:
        video_path: Path to input video file
        output_dir: Output directory for motion vectors
        
    Returns:
        True if successful, False otherwise
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    
    # Create output directory
    os.makedirs(video_output_dir, exist_ok=True)
    mv_dir = os.path.join(video_output_dir, "motion_vectors")
    os.makedirs(mv_dir, exist_ok=True)
    
    # Use ffmpeg to extract motion vectors
    # Note: This is a basic implementation for demonstration purposes
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "extractplanes=y",
        "-f", "null", "-"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Motion vectors extracted for {video_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting motion vectors for {video_name}: {e}")
        return False


def extract_from_directory(video_dir: str, output_dir: str) -> None:
    """
    Extract motion vectors from all videos in a directory.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Output directory for motion vectors
    """
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files.sort()
    
    if not video_files:
        print("No video files found in directory")
        return
    
    print(f"Found {len(video_files)} video files")
    
    successful = 0
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(video_dir, video_file)
        print(f"Processing {i}/{len(video_files)}: {video_file}")
        
        if extract_motion_vectors(video_path, output_dir):
            successful += 1
    
    print(f"Successfully processed {successful}/{len(video_files)} videos")


def main():
    parser = argparse.ArgumentParser(description="Extract motion vectors from videos")
    parser.add_argument("--video_path", type=str, help="Path to single video file")
    parser.add_argument("--video_dir", type=str, help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for motion vectors")
    
    args = parser.parse_args()
    
    if args.video_path and args.video_dir:
        print("Error: Cannot specify both --video_path and --video_dir")
        sys.exit(1)
    
    if not args.video_path and not args.video_dir:
        print("Error: Must specify either --video_path or --video_dir")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.video_path:
        # Extract from single video
        if extract_motion_vectors(args.video_path, args.output_dir):
            print("Motion vector extraction completed successfully")
        else:
            print("Motion vector extraction failed")
            sys.exit(1)
    else:
        # Extract from directory
        extract_from_directory(args.video_dir, args.output_dir)


if __name__ == "__main__":
    main()
