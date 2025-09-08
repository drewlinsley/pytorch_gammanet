"""Evaluation metrics for edge detection.

Implements ODS (Optimal Dataset Scale) and OIS (Optimal Image Scale) F-scores
for BSDS500 evaluation.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional
from sklearn.metrics import precision_recall_curve, auc
import cv2


def compute_edge_f1(pred: np.ndarray, gt: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    """Compute F1 score for edge detection at a specific threshold.
    
    Args:
        pred: Predicted edge map [H, W]
        gt: Ground truth edge map [H, W]
        threshold: Threshold for binarization
        
    Returns:
        precision, recall, f1 score
    """
    # Binarize predictions
    pred_binary = (pred > threshold).astype(np.float32)
    
    # Compute matches within tolerance (dilate GT for tolerance)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    gt_dilated = cv2.dilate(gt.astype(np.uint8), kernel, iterations=1)
    
    # True positives: predicted edges that match dilated GT
    tp = np.sum(pred_binary * gt_dilated)
    
    # False positives: predicted edges that don't match dilated GT
    fp = np.sum(pred_binary * (1 - gt_dilated))
    
    # False negatives: GT edges not covered by predictions
    pred_dilated = cv2.dilate(pred_binary.astype(np.uint8), kernel, iterations=1)
    fn = np.sum(gt * (1 - pred_dilated))
    
    # Compute metrics
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    
    return precision, recall, f1


def compute_ods_ois(predictions: List[np.ndarray], ground_truths: List[np.ndarray], 
                    thresholds: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute ODS and OIS F-scores for edge detection.
    
    ODS: Optimal Dataset Scale - best F1 using same threshold for all images
    OIS: Optimal Image Scale - best F1 using optimal threshold per image
    
    Args:
        predictions: List of predicted edge maps
        ground_truths: List of ground truth edge maps
        thresholds: Thresholds to evaluate (default: 99 thresholds from 0.01 to 0.99)
        
    Returns:
        Dictionary with 'ods_f1', 'ois_f1', 'ods_threshold', 'ap' (average precision)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
        
    n_images = len(predictions)
    n_thresholds = len(thresholds)
    
    # Store F1 scores for each image and threshold
    f1_scores = np.zeros((n_images, n_thresholds))
    
    # Compute F1 for all combinations
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        for j, thresh in enumerate(thresholds):
            _, _, f1 = compute_edge_f1(pred, gt, thresh)
            f1_scores[i, j] = f1
            
    # ODS: Best F1 using same threshold for all images
    mean_f1_per_threshold = f1_scores.mean(axis=0)
    ods_idx = np.argmax(mean_f1_per_threshold)
    ods_f1 = mean_f1_per_threshold[ods_idx]
    ods_threshold = thresholds[ods_idx]
    
    # OIS: Best F1 using optimal threshold per image
    ois_f1 = f1_scores.max(axis=1).mean()
    
    # Average Precision (using all predictions)
    ap = compute_average_precision(predictions, ground_truths)
    
    return {
        'ods_f1': ods_f1,
        'ois_f1': ois_f1,
        'ods_threshold': ods_threshold,
        'ap': ap
    }


def compute_average_precision(predictions: List[np.ndarray], 
                            ground_truths: List[np.ndarray]) -> float:
    """Compute average precision for edge detection.
    
    Args:
        predictions: List of predicted edge probability maps
        ground_truths: List of binary ground truth edge maps
        
    Returns:
        Average precision score
    """
    # Flatten all predictions and ground truths
    all_preds = np.concatenate([p.flatten() for p in predictions])
    all_gts = np.concatenate([g.flatten() for g in ground_truths])
    
    # Ensure ground truth is binary
    all_gts = (all_gts > 0.5).astype(np.float32)
    
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(all_gts, all_preds)
    
    # Compute area under curve
    ap = auc(recall, precision)
    
    return ap


class EdgeDetectionMetrics:
    """Helper class to accumulate predictions and compute metrics."""
    
    def __init__(self):
        self.predictions = []
        self.ground_truths = []
        
    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        """Add a batch of predictions and ground truths.
        
        Args:
            pred: Predictions [B, 1, H, W] or [B, H, W]
            gt: Ground truths [B, 1, H, W] or [B, H, W]
        """
        # Handle different input shapes
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if gt.dim() == 4:
            gt = gt.squeeze(1)
            
        # Convert to numpy and store
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        
        for p, g in zip(pred_np, gt_np):
            self.predictions.append(p)
            self.ground_truths.append(g)
            
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary with ODS, OIS, AP metrics
        """
        if len(self.predictions) == 0:
            return {'ods_f1': 0.0, 'ois_f1': 0.0, 'ap': 0.0}
            
        return compute_ods_ois(self.predictions, self.ground_truths)
    
    def reset(self):
        """Reset accumulated predictions."""
        self.predictions = []
        self.ground_truths = []