"""
MVP: Motion Vector Propagation for Zero-Shot Object Detection

Main detector class that combines OWL-ViT with motion vector analysis
for efficient zero-shot object detection in videos.
"""

import os
import time
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from .motion_analyzer import MotionAnalyzer
from .owl_detector import OWLDetector


class MVPDetector:
    """
    Main MVP detector class that orchestrates motion vector analysis
    and OWL-ViT detection for efficient zero-shot object detection.
    """
    
    def __init__(self, 
                 model_id: str = "google/owlv2-large-patch14-ensemble",
                 device: str = "auto",
                 confidence_threshold: float = 0.5,
                 single_class_fail_limit: int = 2,
                 area_scale_threshold: float = 2.0):
        """
        Initialize MVP detector.
        
        Args:
            model_id: OWL-ViT model identifier
            device: Device to run on ('auto', 'cuda', 'cpu')
            confidence_threshold: Detection confidence threshold
            single_class_fail_limit: Max failures before switching from single-class mode
            area_scale_threshold: Area scaling threshold for re-detection
        """
        self.device = torch.device(device if device != "auto" else 
                                 ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize OWL-ViT detector
        self.owl_detector = OWLDetector(
            model_id=model_id,
            device=self.device,
            confidence_threshold=confidence_threshold
        )
        
        # Initialize motion analyzer
        self.motion_analyzer = MotionAnalyzer()
        
        # Configuration
        self.single_class_fail_limit = single_class_fail_limit
        self.area_scale_threshold = area_scale_threshold
        
        # State tracking
        self.reset_state()
    
    def reset_state(self):
        """Reset detector state for new video."""
        self.prev_dets_path = None
        self.no_detection_count = 0
        self.last_owlvit_frame = -1
        self.last_owlvit_dets = []
        self.use_single_class_mode = False
        self.single_class_name = None
        self.single_class_fail_count = 0
        self.first_detection_done = False
    
    def process_video(self, 
                     video_path: str,
                     motion_vector_dir: str,
                     output_dir: str,
                     extract_frames: bool = True) -> Dict:
        """
        Process a single video with MVP framework.
        
        Args:
            video_path: Path to input video
            motion_vector_dir: Directory containing motion vectors
            output_dir: Output directory for results
            extract_frames: Whether to extract frames from video
            
        Returns:
            Dictionary containing processing statistics
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Setup paths
        frame_types_txt = os.path.join(motion_vector_dir, "frame_types.txt")
        mv_dir = os.path.join(motion_vector_dir, "motion_vectors")
        
        if not os.path.exists(frame_types_txt):
            raise FileNotFoundError(f"Frame types file not found: {frame_types_txt}")
        
        # Read frame types
        frame_types = self._read_frame_types(frame_types_txt)
        num_frames = len(frame_types)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, "frames")
        
        print(f"Processing video {video_name}, frames: {num_frames}")
        start_time = time.time()
        
        # Extract frames if needed
        if extract_frames:
            self._extract_frames(video_path, frames_dir)
        
        # Reset state
        self.reset_state()
        
        # Process each frame
        stats = {
            'total_frames': num_frames,
            'owl_detections': 0,
            'propagations': 0,
            'area_triggers': 0,
            'single_class_switches': 0
        }
        
        for frame_idx in range(num_frames):
            frame_type = frame_types[frame_idx]
            frame_name = f"frame_{frame_idx:05d}"
            image_path = os.path.join(frames_dir, frame_name + ".png")
            out_dets_path = os.path.join(output_dir, frame_name + ".npy")
            
            if frame_type == "I" or not self.first_detection_done:
                # Keyframe or first frame - use OWL-ViT
                self._process_keyframe(image_path, out_dets_path, frame_idx)
                stats['owl_detections'] += 1
                self.first_detection_done = True
                
            else:
                # Non-keyframe - try propagation
                success = self._process_non_keyframe(
                    image_path, out_dets_path, mv_dir, frame_idx, frame_type
                )
                
                if success:
                    stats['propagations'] += 1
                else:
                    stats['owl_detections'] += 1
        
        # Calculate processing time
        processing_time = time.time() - start_time
        stats['processing_time'] = processing_time
        stats['fps'] = num_frames / processing_time if processing_time > 0 else 0
        
        print(f"Completed {video_name}: {stats['fps']:.2f} FPS")
        return stats
    
    def _read_frame_types(self, frame_type_path: str) -> List[str]:
        """Read frame types from file."""
        with open(frame_type_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def _extract_frames(self, video_path: str, frames_dir: str):
        """Extract all frames from video."""
        os.makedirs(frames_dir, exist_ok=True)
        import subprocess
        cmd = (
            f"ffmpeg -y -i '{video_path}' "
            f"-start_number 0 "
            f"'{os.path.join(frames_dir, 'frame_%05d.png')}'"
        )
        subprocess.run(cmd, shell=True, check=True)
    
    def _process_keyframe(self, image_path: str, out_dets_path: str, frame_idx: int):
        """Process keyframe with OWL-ViT detection."""
        # Detect with OWL-ViT
        dets = self.owl_detector.detect(
            image_path, 
            single_class_name=self.single_class_name if self.use_single_class_mode else None
        )
        
        # Save detections
        np.save(out_dets_path, dets)
        
        # Update single class mode
        self._update_single_class_mode(dets)
        
        # Update state
        self.last_owlvit_frame = frame_idx
        self.last_owlvit_dets = dets
        self.no_detection_count = 0 if len(dets) > 0 else self.no_detection_count + 1
        self.prev_dets_path = out_dets_path
    
    def _process_non_keyframe(self, 
                             image_path: str, 
                             out_dets_path: str, 
                             mv_dir: str, 
                             frame_idx: int,
                             frame_type: str) -> bool:
        """Process non-keyframe with motion vector propagation."""
        
        # Check if we need to force detection
        if self.no_detection_count >= 5:
            self._process_keyframe(image_path, out_dets_path, frame_idx)
            return False
        
        # Try motion vector propagation
        mv_file = f"mvs-{frame_idx}.npy"
        mv_path = os.path.join(mv_dir, mv_file)
        
        if not os.path.exists(mv_path):
            self._process_keyframe(image_path, out_dets_path, frame_idx)
            return False
        
        # Attempt propagation
        new_dets = self.motion_analyzer.propagate_detections(
            image_path, self.prev_dets_path, mv_path
        )
        
        if new_dets is None:
            # Propagation failed - fallback to detection
            self._process_keyframe(image_path, out_dets_path, frame_idx)
            return False
        
        # Check area scaling
        if self._check_area_scaling(new_dets, frame_idx):
            # Area scaling detected - re-detect
            self._process_keyframe(image_path, out_dets_path, frame_idx)
            return False
        
        # Save propagated detections
        np.save(out_dets_path, new_dets)
        
        # Update state
        self.no_detection_count = 0 if len(new_dets) > 0 else self.no_detection_count + 1
        self.prev_dets_path = out_dets_path
        
        return True
    
    def _update_single_class_mode(self, dets: List[Dict]):
        """Update single class detection mode based on current detections."""
        if len(dets) == 1 and dets[0]['score'] >= 0.2:
            # Single high-confidence detection - switch to single class mode
            self.single_class_name = dets[0]['class_name']
            self.use_single_class_mode = True
            self.single_class_fail_count = 0
        elif self.use_single_class_mode:
            # Check if single class mode is still working
            if len(dets) == 0 or max(d['score'] for d in dets) < 0.2:
                self.single_class_fail_count += 1
                if self.single_class_fail_count >= self.single_class_fail_limit:
                    # Switch back to multi-class mode
                    self.use_single_class_mode = False
                    self.single_class_name = None
                    self.single_class_fail_count = 0
            else:
                self.single_class_fail_count = 0
    
    def _check_area_scaling(self, current_dets: List[Dict], frame_idx: int) -> bool:
        """Check if any detection has scaled significantly in area."""
        if frame_idx - self.last_owlvit_frame > 10:
            return False
        
        # Match detections by class and center
        pairs = self._match_detections(current_dets, self.last_owlvit_dets)
        
        for cur_det, last_det in pairs:
            cur_area = self._calculate_bbox_area(cur_det)
            last_area = self._calculate_bbox_area(last_det)
            
            if last_area > 0 and cur_area > self.area_scale_threshold * last_area:
                return True
        
        return False
    
    def _match_detections(self, current_dets: List[Dict], last_dets: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Match current detections with last detections by class and center."""
        pairs = []
        used_last = set()
        
        for cur_det in current_dets:
            best_match = None
            best_dist = float('inf')
            best_idx = -1
            
            for i, last_det in enumerate(last_dets):
                if i in used_last:
                    continue
                
                if last_det['class_name'] != cur_det['class_name']:
                    continue
                
                # Calculate center distance
                dx = cur_det['x_center'] - last_det['x_center']
                dy = cur_det['y_center'] - last_det['y_center']
                dist = dx * dx + dy * dy
                
                if dist < best_dist:
                    best_dist = dist
                    best_match = last_det
                    best_idx = i
            
            if best_match is not None:
                pairs.append((cur_det, best_match))
                used_last.add(best_idx)
        
        return pairs
    
    def _calculate_bbox_area(self, det: Dict) -> float:
        """Calculate bounding box area."""
        w = det['width']
        h = det['height']
        return w * h
