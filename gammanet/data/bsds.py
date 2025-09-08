"""BSDS500 dataset for edge detection.

This module implements the BSDS500 dataset loader with support for:
- Multiple ground truth annotations per image
- Efficient preprocessing and caching
- Data augmentation via albumentations
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Union
import glob
from pathlib import Path
import json


class BSDS500Dataset(Dataset):
    """BSDS500 dataset for boundary detection.
    
    The Berkeley Segmentation Dataset contains natural images with
    human-annotated boundaries. Each image has 4-9 human annotations.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
        target_size: Tuple[int, int] = (321, 481),  # Standard BSDS size
        thin_edges: bool = True,
        cache_data: bool = True
    ):
        """Initialize BSDS500 dataset.
        
        Args:
            root_dir: Root directory containing BSDS500 data
            split: One of 'train', 'val', 'test'
            transform: Albumentations transform pipeline
            target_size: Resize images to this size (H, W)
            thin_edges: Apply morphological thinning to edges
            cache_data: Cache preprocessed data in memory
        """
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.thin_edges = thin_edges
        self.cache_data = cache_data
        
        # Setup paths
        self._setup_paths()
        
        # Load file lists
        self.image_list = self._load_image_list()
        
        # Cache for preprocessed data
        self.cache = {} if cache_data else None
        
    def _setup_paths(self):
        """Setup dataset paths based on BSDS structure."""
        self.images_dir = self.root_dir / 'images' / self.split
        self.edges_dir = self.root_dir / 'groundTruth' / self.split
        
        # Verify paths exist
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.edges_dir.exists():
            raise ValueError(f"Ground truth directory not found: {self.edges_dir}")
            
    def _load_image_list(self) -> List[str]:
        """Load list of image IDs for the split."""
        # BSDS uses .jpg for images
        image_files = sorted(glob.glob(str(self.images_dir / "*.jpg")))
        
        # Extract IDs (filename without extension)
        image_ids = [Path(f).stem for f in image_files]
        
        if len(image_ids) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
            
        print(f"Loaded {len(image_ids)} images for {self.split} split")
        return image_ids
        
    def _load_image(self, image_id: str) -> np.ndarray:
        """Load image by ID."""
        image_path = self.images_dir / f"{image_id}.jpg"
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Resize if needed
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]), 
                             interpolation=cv2.INTER_CUBIC)
            
        return image
        
    def _load_edges(self, image_id: str) -> np.ndarray:
        """Load ground truth edges for image.
        
        BSDS stores ground truth in .mat files with multiple annotations.
        For PyTorch implementation, we assume preprocessed .png edge maps.
        Also supports .npy format for pre-processed crops.
        """
        # First try .npy format (for BSDS500_crops)
        npy_path = self.edges_dir / f"{image_id}.npy"
        if npy_path.exists():
            edges = np.load(str(npy_path)).astype(np.float32)
            # Ensure edges are in [0, 1] range
            if edges.max() > 1.0:
                edges = edges / 255.0
        else:
            # Look for preprocessed edge maps
            edge_path = self.edges_dir / f"{image_id}.png"
            
            if edge_path.exists():
                # Load preprocessed edge map
                edges = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
                edges = edges.astype(np.float32) / 255.0
            else:
                # Try loading from .mat file (requires scipy)
                mat_path = self.edges_dir / f"{image_id}.mat"
                if mat_path.exists():
                    edges = self._load_edges_from_mat(mat_path)
                else:
                    # Create dummy edges for testing
                    print(f"Warning: No ground truth found for {image_id}")
                    edges = np.zeros(self.target_size, dtype=np.float32)
                
        # Resize if needed
        if edges.shape[:2] != self.target_size:
            edges = cv2.resize(edges, (self.target_size[1], self.target_size[0]), 
                             interpolation=cv2.INTER_NEAREST)
            
        # Thin edges if requested
        if self.thin_edges and edges.max() > 0:
            edges = self._thin_edges(edges)
            
        return edges
        
    def _load_edges_from_mat(self, mat_path: Path) -> np.ndarray:
        """Load edges from MATLAB .mat file.
        
        This is a placeholder - actual implementation would use scipy.io.loadmat
        """
        try:
            import scipy.io
            mat_data = scipy.io.loadmat(str(mat_path))
            
            # Extract ground truth boundaries (multiple annotations)
            groundTruth = mat_data['groundTruth'].flatten()
            
            # Average multiple annotations
            edge_maps = []
            for gt in groundTruth:
                boundaries = gt['Boundaries'][0, 0]
                edge_maps.append(boundaries.astype(np.float32))
                
            # Average all annotations
            edges = np.mean(edge_maps, axis=0)
            
            return edges
            
        except ImportError:
            print("Warning: scipy not available, cannot load .mat files")
            return np.zeros(self.target_size, dtype=np.float32)
            
    def _thin_edges(self, edges: np.ndarray) -> np.ndarray:
        """Apply morphological thinning to edge map."""
        # Convert to binary
        binary_edges = (edges > 0.1).astype(np.uint8)
        
        # Try different thinning methods
        try:
            # Method 1: OpenCV contrib (if installed)
            thinned = cv2.ximgproc.thinning(binary_edges)
        except AttributeError:
            try:
                # Method 2: scikit-image morphology
                from skimage.morphology import thin
                thinned = thin(binary_edges).astype(np.uint8)
            except ImportError:
                # Method 3: Simple morphological erosion (fallback)
                kernel = np.ones((3, 3), np.uint8)
                # Perform one iteration of erosion to thin edges
                thinned = binary_edges - cv2.morphologyEx(
                    binary_edges, cv2.MORPH_ERODE, kernel, iterations=1
                )
                # Ensure edges remain connected
                thinned = np.maximum(thinned, cv2.morphologyEx(
                    binary_edges, cv2.MORPH_ERODE, kernel, iterations=2
                ))
        
        # Convert back to float
        return thinned.astype(np.float32)
        
    def __len__(self) -> int:
        return len(self.image_list)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index.
        
        Returns:
            Dictionary containing:
                - image: RGB image tensor [3, H, W]
                - edges: Edge map tensor [1, H, W]
                - image_id: Image identifier
        """
        image_id = self.image_list[idx]
        
        # Check cache
        if self.cache_data and image_id in self.cache:
            image, edges = self.cache[image_id]
        else:
            # Load data
            image = self._load_image(image_id)
            edges = self._load_edges(image_id)
            
            # Cache if enabled
            if self.cache_data:
                self.cache[image_id] = (image.copy(), edges.copy())
                
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=edges)
            image = transformed['image']
            edges = transformed['mask']
            
        # Convert to tensors
        if isinstance(image, np.ndarray):
            # Don't normalize here - let transforms handle it
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if isinstance(edges, np.ndarray):
            edges = torch.from_numpy(edges).unsqueeze(0).float()
            
        return {
            'image': image,
            'edges': edges,
            'image_id': image_id
        }


def create_bsds_datasets(
    root_dir: str,
    train_transform=None,
    val_transform=None,
    target_size: Tuple[int, int] = (321, 481),
    cache_data: bool = True
) -> Tuple[BSDS500Dataset, BSDS500Dataset]:
    """Create train and validation BSDS datasets.
    
    Args:
        root_dir: Root directory of BSDS500
        train_transform: Training augmentations
        val_transform: Validation preprocessing
        target_size: Target image size
        cache_data: Whether to cache data in memory
        
    Returns:
        train_dataset, val_dataset
    """
    train_dataset = BSDS500Dataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform,
        target_size=target_size,
        cache_data=cache_data
    )
    
    val_dataset = BSDS500Dataset(
        root_dir=root_dir,
        split='val',
        transform=val_transform,
        target_size=target_size,
        cache_data=cache_data
    )
    
    return train_dataset, val_dataset