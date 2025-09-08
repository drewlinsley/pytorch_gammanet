"""Loss functions for GammaNet training.

Includes specialized losses for edge detection and perceptual tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BalancedBCELoss(nn.Module):
    """Balanced Binary Cross Entropy for imbalanced edge detection.
    
    Automatically balances positive/negative samples based on the
    ground truth edge frequency.
    """
    
    def __init__(self, pos_weight: Optional[float] = None, reduction: str = 'mean'):
        """Initialize BalancedBCELoss.
        
        Args:
            pos_weight: Manual positive class weight. If None, computed automatically.
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute balanced BCE loss.
        
        Args:
            predictions: Model predictions (logits) [B, 1, H, W]
            targets: Ground truth edges [B, 1, H, W]
            
        Returns:
            Balanced BCE loss
        """
        # Compute positive weight if not provided
        if self.pos_weight is None:
            # Calculate class balance per batch
            pos_count = targets.sum()
            neg_count = (1 - targets).sum()
            pos_weight = neg_count / (pos_count + 1e-5)
            # Clamp to reasonable range
            pos_weight = torch.clamp(pos_weight, min=0.1, max=10.0)
        else:
            pos_weight = self.pos_weight
            
        # No need to apply sigmoid - binary_cross_entropy_with_logits handles it
            
        # Compute weighted BCE
        loss = F.binary_cross_entropy_with_logits(
            predictions, targets, 
            pos_weight=pos_weight,
            reduction=self.reduction
        )
        
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focuses learning on hard examples by down-weighting easy examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """Initialize FocalLoss.
        
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Exponent of the modulating factor (1 - p_t)^gamma
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            predictions: Model predictions (logits) [B, 1, H, W]
            targets: Ground truth [B, 1, H, W]
            
        Returns:
            Focal loss
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(predictions)
        
        # Compute focal term
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute final loss
        loss = alpha_t * focal_term * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PearsonCorrelationLoss(nn.Module):
    """Pearson correlation coefficient as a loss function.
    
    Maximizes correlation between predictions and targets.
    Used for perceptual similarity tasks.
    """
    
    def __init__(self, eps: float = 1e-5):
        """Initialize PearsonCorrelationLoss.
        
        Args:
            eps: Small value for numerical stability
        """
        super().__init__()
        self.eps = eps
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute negative Pearson correlation as loss.
        
        Args:
            predictions: Model predictions [B, *]
            targets: Ground truth [B, *]
            
        Returns:
            Negative correlation (to minimize)
        """
        # Flatten spatial dimensions
        predictions = predictions.flatten(1)
        targets = targets.flatten(1)
        
        # Compute means
        pred_mean = predictions.mean(dim=1, keepdim=True)
        target_mean = targets.mean(dim=1, keepdim=True)
        
        # Center the data
        pred_centered = predictions - pred_mean
        target_centered = targets - target_mean
        
        # Compute correlation
        numerator = (pred_centered * target_centered).sum(dim=1)
        
        pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1) + self.eps)
        target_std = torch.sqrt((target_centered ** 2).sum(dim=1) + self.eps)
        denominator = pred_std * target_std
        
        correlation = numerator / denominator
        
        # Return negative correlation as loss
        return -correlation.mean()


class BiBalancedBCELoss(nn.Module):
    """Binary Balanced BCE Loss from HED paper.
    
    Implements the bi_bce_hed loss from the original TensorFlow GammaNet.
    Handles soft labels by thresholding at gamma (default 0.5).
    """
    
    def __init__(self, gamma: float = 0.5, neg_weight: float = 1.1):
        """Initialize BiBalancedBCELoss.
        
        Args:
            gamma: Threshold for converting soft labels to hard labels
            neg_weight: Weight multiplier for negative pixels (default 1.1 from original)
        """
        super().__init__()
        self.gamma = gamma
        self.neg_weight = neg_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute bi_bce_hed loss.
        
        Args:
            predictions: Model predictions (logits) [B, 1, H, W]
            targets: Ground truth edges [B, 1, H, W] with soft labels
            
        Returns:
            Balanced BCE loss with sum reduction
        """
        # Threshold labels at gamma
        # Convert labels >= gamma to 1, keep 0s as 0, ignore negative values
        y = torch.where(targets >= self.gamma, torch.ones_like(targets), targets)
        
        # Create masks for positive and negative pixels
        pos_mask = (y == 1).float()
        neg_mask = (y == 0).float()
        
        # Count positive and negative pixels
        pos_count = pos_mask.sum()
        neg_count = neg_mask.sum()
        valid_count = pos_count + neg_count
        
        # Compute per-pixel weights
        # Positive pixels get weight: neg_count / valid_count
        # Negative pixels get weight: pos_count * 1.1 / valid_count
        pos_weight = pos_mask * (neg_count / (valid_count + 1e-5))
        neg_weight = neg_mask * (pos_count * self.neg_weight / (valid_count + 1e-5))
        weights = pos_weight + neg_weight
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, y, reduction='none'
        )
        
        # Apply weights and mask for valid pixels (targets >= 0)
        valid_mask = (targets >= 0).float()
        weighted_loss = bce_loss * weights * valid_mask
        
        # Sum reduction (matching original implementation)
        # Note: The original uses sum but we'll normalize by batch size for stability
        return weighted_loss.sum() / predictions.shape[0]


class CombinedLoss(nn.Module):
    """Combine multiple losses with weighting."""
    
    def __init__(self, losses: dict, weights: dict):
        """Initialize CombinedLoss.
        
        Args:
            losses: Dictionary of loss functions
            weights: Dictionary of loss weights
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss.
        
        Returns:
            total_loss: Weighted sum of losses
            loss_dict: Individual loss values
        """
        total_loss = 0
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            loss = loss_fn(predictions, targets)
            weighted_loss = self.weights.get(name, 1.0) * loss
            total_loss += weighted_loss
            loss_dict[name] = loss.item()
            
        return total_loss, loss_dict