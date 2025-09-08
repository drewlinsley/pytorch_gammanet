"""Data augmentation and preprocessing transforms."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Tuple, Optional, List


def get_train_transforms(
    crop_size: int = 320,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    p_flip: float = 0.5,
    p_rotate: float = 0.5,
    p_color: float = 0.3,
) -> A.Compose:
    """Get training augmentation pipeline.
    
    Args:
        crop_size: Size of random crops
        mean: Normalization mean
        std: Normalization std
        p_flip: Probability of horizontal/vertical flip
        p_rotate: Probability of rotation
        p_color: Probability of color augmentation
        
    Returns:
        Albumentations composition
    """
    return A.Compose([
        # Spatial augmentations
        A.RandomCrop(height=crop_size, width=crop_size, p=1.0),  # Always apply to ensure consistent size
        A.HorizontalFlip(p=p_flip),
        A.VerticalFlip(p=p_flip),
        A.RandomRotate90(p=p_rotate),
        
        # Color augmentations (only on image, not edges)
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=p_color
        ),
        
        # Normalize
        A.Normalize(mean=mean, std=std),
        
        # Convert to tensor is handled in dataset
    ])


def get_val_transforms(
    crop_size: int = 320,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> A.Compose:
    """Get validation preprocessing pipeline.
    
    Args:
        crop_size: Size of center crop
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Albumentations composition
    """
    return A.Compose([
        # Center crop for validation to ensure consistent size
        A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
        # Normalize
        A.Normalize(mean=mean, std=std),
    ])


def get_test_transforms(
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> A.Compose:
    """Get test preprocessing pipeline.
    
    Same as validation but kept separate for flexibility.
    """
    return get_val_transforms(mean, std)


def get_tta_transforms(
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> List[A.Compose]:
    """Get test-time augmentation transforms.
    
    Returns multiple transform pipelines for TTA.
    """
    base_transform = A.Normalize(mean=mean, std=std)
    
    transforms = [
        # Original
        A.Compose([base_transform]),
        
        # Horizontal flip
        A.Compose([
            A.HorizontalFlip(p=1.0),
            base_transform
        ]),
        
        # Vertical flip  
        A.Compose([
            A.VerticalFlip(p=1.0),
            base_transform
        ]),
        
        # 90 degree rotations
        A.Compose([
            A.Rotate(limit=(90, 90), p=1.0),
            base_transform
        ]),
        
        A.Compose([
            A.Rotate(limit=(180, 180), p=1.0),
            base_transform
        ]),
        
        A.Compose([
            A.Rotate(limit=(270, 270), p=1.0),
            base_transform
        ]),
    ]
    
    return transforms