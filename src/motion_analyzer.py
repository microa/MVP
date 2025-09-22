"""
Motion vector analysis for object tracking and propagation.
"""

import os
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
from math import sqrt


class MotionAnalyzer:
    """
    Analyzes motion vectors to propagate object detections across frames.
    """
    
    def __init__(self):
        """Initialize motion analyzer."""
        pass
    
    def propagate_detections(self, 
                           image_path: str, 
                           ref_dets_path: str, 
                           mv_path: str) -> Optional[List[Dict]]:
        """
        Propagate detections using motion vector analysis.
        
        Args:
            image_path: Path to current frame image
            ref_dets_path: Path to reference detections (.npy file)
            mv_path: Path to motion vectors (.npy file)
            
        Returns:
            List of propagated detections or None if propagation failed
        """
        if not os.path.exists(mv_path) or not os.path.exists(ref_dets_path):
            return None
        
        # Load data
        W, H = self._get_image_resolution(image_path)
        ref_dets = np.load(ref_dets_path, allow_pickle=True)
        mvs = np.load(mv_path, allow_pickle=True)
        
        # Propagate each detection
        new_dets = []
        for det in ref_dets:
            propagated_det = self._propagate_single_detection(det, mvs, W, H)
            if propagated_det is None:
                return None  # If any detection fails, fail the entire frame
            new_dets.append(propagated_det)
        
        return new_dets
    
    def _propagate_single_detection(self, 
                                  det: Dict, 
                                  mvs: np.ndarray, 
                                  W: int, 
                                  H: int) -> Optional[Dict]:
        """
        Propagate a single detection using 9-grid motion vector analysis.
        
        Args:
            det: Detection dictionary
            mvs: Motion vectors array
            W, H: Image dimensions
            
        Returns:
            Propagated detection or None if propagation failed
        """
        # Convert to pixel coordinates
        x1, y1, x2, y2 = self._yolo_to_xyxy(
            det['x_center'], det['y_center'], det['width'], det['height'], W, H
        )
        
        # Analyze motion vectors in 9-grid
        centers, displacements = self._analyze_9grid_motion(x1, y1, x2, y2, mvs)
        
        # Try pure translation first
        translation = self._check_translation(displacements)
        if translation is not None:
            dx, dy = translation
            new_x1 = x1 + dx
            new_x2 = x2 + dx
            new_y1 = y1 + dy
            new_y2 = y2 + dy
            
            # Convert back to normalized coordinates
            xc, yc, w, h = self._xyxy_to_yolo(new_x1, new_y1, new_x2, new_y2, W, H)
            xc, yc, w, h = self._clamp_bbox(xc, yc, w, h)
            
            return {
                'class_index': det['class_index'],
                'class_name': det['class_name'],
                'x_center': xc,
                'y_center': yc,
                'width': w,
                'height': h,
                'score': det['score']
            }
        
        # Try pure scaling
        scale = self._check_scaling(centers, displacements)
        if scale is not None:
            # Scale around center
            old_xc = det['x_center']
            old_yc = det['y_center']
            old_w = det['width']
            old_h = det['height']
            
            new_w = old_w * scale
            new_h = old_h * scale
            
            xc, yc, w, h = self._clamp_bbox(old_xc, old_yc, new_w, new_h)
            
            return {
                'class_index': det['class_index'],
                'class_name': det['class_name'],
                'x_center': xc,
                'y_center': yc,
                'width': w,
                'height': h,
                'score': det['score']
            }
        
        # Both translation and scaling failed
        return None
    
    def _analyze_9grid_motion(self, 
                            x1: float, y1: float, x2: float, y2: float, 
                            mvs: np.ndarray) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Analyze motion vectors in a 9-grid pattern within the bounding box.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            mvs: Motion vectors array
            
        Returns:
            Tuple of (centers, displacements) for each grid cell
        """
        w_box = x2 - x1
        h_box = y2 - y1
        
        if w_box < 1 or h_box < 1:
            return [(0, 0)] * 9, [(0, 0)] * 9
        
        box_cx = (x1 + x2) / 2
        box_cy = (y1 + y2) / 2
        x_step = w_box / 3
        y_step = h_box / 3
        
        centers = []
        displacements = []
        
        for row in range(3):
            for col in range(3):
                # Define sub-grid boundaries
                sub_x1 = x1 + col * x_step
                sub_y1 = y1 + row * y_step
                sub_x2 = sub_x1 + x_step
                sub_y2 = sub_y1 + y_step
                
                # Collect motion vectors in this sub-grid
                dx_list = []
                dy_list = []
                
                for mv in mvs:
                    if mv[0] != -1:  # Skip invalid motion vectors
                        continue
                    
                    ref_cx = mv[3]
                    ref_cy = mv[4]
                    cur_cx = mv[5]
                    cur_cy = mv[6]
                    
                    # Check if motion vector is in this sub-grid
                    if sub_x1 <= ref_cx < sub_x2 and sub_y1 <= ref_cy < sub_y2:
                        dx_list.append(cur_cx - ref_cx)
                        dy_list.append(cur_cy - ref_cy)
                
                # Calculate average displacement
                if len(dx_list) == 0:
                    dx_avg = 0
                    dy_avg = 0
                else:
                    dx_avg = sum(dx_list) / len(dx_list)
                    dy_avg = sum(dy_list) / len(dy_list)
                
                # Calculate grid center relative to box center
                grid_cx = ((sub_x1 + sub_x2) / 2) - box_cx
                grid_cy = ((sub_y1 + sub_y2) / 2) - box_cy
                
                centers.append((grid_cx, grid_cy))
                displacements.append((dx_avg, dy_avg))
        
        return centers, displacements
    
    def _check_translation(self, displacements: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        Check if displacements represent pure translation.
        
        Args:
            displacements: List of (dx, dy) displacement tuples
            
        Returns:
            Average translation (dx, dy) or None if not pure translation
        """
        if len(displacements) != 9:
            return None
        
        dxs = [d[0] for d in displacements]
        dys = [d[1] for d in displacements]
        
        dx_avg = sum(dxs) / 9.0
        dy_avg = sum(dys) / 9.0
        
        # Check variance to determine if it's pure translation
        var_dx = sum((x - dx_avg) ** 2 for x in dxs) / 9
        var_dy = sum((y - dy_avg) ** 2 for y in dys) / 9
        
        std_dx = sqrt(var_dx)
        std_dy = sqrt(var_dy)
        
        # If standard deviation is too high, it's not pure translation
        if std_dx > 1.0 or std_dy > 1.0:
            return None
        
        return (dx_avg, dy_avg)
    
    def _check_scaling(self, 
                      centers: List[Tuple[float, float]], 
                      displacements: List[Tuple[float, float]]) -> Optional[float]:
        """
        Check if displacements represent pure scaling around center.
        
        Args:
            centers: List of (x, y) center coordinates
            displacements: List of (dx, dy) displacement tuples
            
        Returns:
            Average scale factor or None if not pure scaling
        """
        if len(centers) != len(displacements) or len(centers) < 2:
            return None
        
        ratios = []
        for (gx, gy), (dx, dy) in zip(centers, displacements):
            old_len2 = gx * gx + gy * gy
            if old_len2 < 1e-6:
                continue
            
            new_x = gx + dx
            new_y = gy + dy
            new_len2 = new_x * new_x + new_y * new_y
            
            if new_len2 < 1e-6:
                ratios.append(0)
            else:
                r = sqrt(new_len2 / old_len2)
                ratios.append(r)
        
        if len(ratios) < 2:
            return None
        
        avg_r = sum(ratios) / len(ratios)
        var_r = sum((r - avg_r) ** 2 for r in ratios) / len(ratios)
        std_r = sqrt(var_r)
        
        # If standard deviation is too high, it's not pure scaling
        if std_r > 0.1:
            return None
        
        return avg_r
    
    def _yolo_to_xyxy(self, xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert YOLO format to xyxy format."""
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        return x1, y1, x2, y2
    
    def _xyxy_to_yolo(self, x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """Convert xyxy format to YOLO format."""
        w_box = x2 - x1
        h_box = y2 - y1
        xc = x1 + w_box / 2
        yc = y1 + h_box / 2
        xc /= img_w
        yc /= img_h
        w_box /= img_w
        h_box /= img_h
        return xc, yc, w_box, h_box
    
    def _clamp_bbox(self, xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float, float]:
        """Clamp bounding box coordinates to valid range."""
        if w > 1:
            w = 1
        if h > 1:
            h = 1
        
        half_w = w / 2
        half_h = h / 2
        xc = max(half_w, min(1.0 - half_w, xc))
        yc = max(half_h, min(1.0 - half_h, yc))
        
        return xc, yc, w, h
    
    def _get_image_resolution(self, image_path: str) -> Tuple[int, int]:
        """Get image resolution."""
        with Image.open(image_path) as img:
            return img.size  # (width, height)
