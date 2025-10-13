"""Training infrastructure for GammaNet with Accelerate support."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Union, Callable
import yaml
import json
from datetime import datetime

from ..models import GammaNet, GammaNetBackbone, VGG16GammaNet, VGG16GammaNetV2
from ..utils import EdgeDetectionMetrics
from .losses import BalancedBCELoss, FocalLoss, PearsonCorrelationLoss, BiBalancedBCELoss


class GammaNetTrainer:
    """Trainer class for GammaNet with Accelerate integration."""
    
    def __init__(self, config: Dict, accelerator: Optional[Accelerator] = None):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary
            accelerator: Pre-configured Accelerator instance
        """
        self.config = config
        
        # Initialize accelerator if not provided
        if accelerator is None:
            self.accelerator = Accelerator(
                mixed_precision='fp16' if config['training'].get('mixed_precision', True) else 'no',
                gradient_accumulation_steps=config['training'].get('gradient_accumulation', 1),
                log_with='wandb' if config['logging'].get('wandb', False) else None,
            )
        else:
            self.accelerator = accelerator
            
        # Set seed for reproducibility
        set_seed(config.get('seed', 42))
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_loss()
        
        # Metrics
        self.train_metrics = EdgeDetectionMetrics()
        self.val_metrics = EdgeDetectionMetrics()
        
        # Setup logging
        self._setup_logging()
        
        # Checkpointing
        self.checkpoint_dir = Path(config['logging']['log_dir']) / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        # Initialize best_metric based on metric type
        metric_name = self.config['training'].get('early_stopping_metric', 'ods_f1')
        if metric_name == 'loss':
            self.best_metric = float('inf')  # For loss, lower is better
        else:
            self.best_metric = 0.0  # For F1/ODS/OIS, higher is better
        self.patience_counter = 0
        
    def _create_model(self) -> Union[GammaNet, GammaNetBackbone, VGG16GammaNet, VGG16GammaNetV2]:
        """Create model instance."""
        # Check if we should use backbone version
        use_backbone = self.config['model'].get('use_backbone', True)
        model_version = self.config['model'].get('model_version', 'v2')  # Default to v2
        
        if use_backbone:
            # Choose between v1 and v2 based on config
            if model_version == 'v2':
                # Use the new VGG16GammaNetV2 with E/I states
                model = VGG16GammaNetV2(
                    config=self.config['model'],
                    input_channels=3,
                    output_channels=1
                )
                if hasattr(self, 'accelerator'):
                    self.accelerator.print("Using VGG16GammaNetV2 with E/I states")
            else:
                # Use the original VGG16GammaNet
                model = VGG16GammaNet(
                    config=self.config['model'],
                    input_channels=3,
                    output_channels=1
                )
                if hasattr(self, 'accelerator'):
                    self.accelerator.print("Using VGG16GammaNet (v1)")
        else:
            model = GammaNet(
                config=self.config['model'],
                input_channels=3,
                output_channels=1
            )
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with separate learning rates for VGG and fGRU."""
        opt_config = self.config['training']
        opt_type = opt_config.get('optimizer', 'adam')
        
        # Get learning rates
        fgru_lr = float(opt_config['learning_rate'])
        vgg_lr = float(opt_config.get('vgg_learning_rate', fgru_lr))  # Default to same LR if not specified
        weight_decay = float(opt_config.get('weight_decay', 0))
        
        # Separate VGG parameters from fGRU parameters
        vgg_params = []
        fgru_params = []
        
        # Check for VGG blocks in the model
        vgg_found = False
        for name, param in self.model.named_parameters():
            # VGG parameters: block*_conv layers
            if 'block' in name and '_conv' in name and not name.startswith('fgru') and not name.startswith('td_fgru'):
                vgg_params.append(param)
                vgg_found = True
            else:
                # Everything else is fGRU parameters
                fgru_params.append(param)
        
        if vgg_found and self.accelerator.is_main_process:
            self.accelerator.print(f"VGG params: {sum(p.numel() for p in vgg_params):,}, LR: {vgg_lr}")
            self.accelerator.print(f"fGRU params: {sum(p.numel() for p in fgru_params):,}, LR: {fgru_lr}")
        elif self.accelerator.is_main_process:
            self.accelerator.print(f"No VGG backbone found, using single LR: {fgru_lr}")
            # Move all params to fGRU group if no VGG found
            if vgg_params:
                fgru_params.extend(vgg_params)
                vgg_params = []
        
        # Create parameter groups
        param_groups = []
        if vgg_params:
            param_groups.append({'params': vgg_params, 'lr': vgg_lr, 'weight_decay': weight_decay})
        if fgru_params:
            param_groups.append({'params': fgru_params, 'lr': fgru_lr, 'weight_decay': weight_decay})
        
        if opt_type == 'adam':
            optimizer = Adam(param_groups)
        elif opt_type == 'adamw':
            optimizer = AdamW(param_groups)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
            
        return optimizer
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        opt_config = self.config['training']
        scheduler_type = opt_config.get('lr_scheduler', 'exponential')
        
        if scheduler_type == 'exponential':
            scheduler = ExponentialLR(
                self.optimizer,
                gamma=opt_config.get('lr_decay', 0.997)
            )
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=opt_config.get('lr_patience', 10),
                min_lr=opt_config.get('lr_min', 1e-6)
            )
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=opt_config['num_epochs'],
                eta_min=opt_config.get('lr_min', 1e-6)
            )
        else:
            scheduler = None
            
        return scheduler
    
    def _create_loss(self) -> nn.Module:
        """Create loss function."""
        loss_config = self.config['training']
        loss_type = loss_config.get('loss', 'bce')
        
        if loss_type == 'bce':
            criterion = BalancedBCELoss()
        elif loss_type == 'bi_bce_hed':
            criterion = BiBalancedBCELoss(gamma=0.5, neg_weight=1.1)
        elif loss_type == 'focal':
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        elif loss_type == 'pearson':
            criterion = PearsonCorrelationLoss()
        else:
            raise ValueError(f"Unknown loss: {loss_type}")
            
        return criterion
    
    def _setup_logging(self):
        """Setup logging (tensorboard/wandb)."""
        if self.accelerator.is_main_process:
            # Create log directory
            log_dir = Path(self.config['logging']['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize wandb if requested
            if self.config['logging'].get('wandb', False):
                wandb.init(
                    project=self.config['logging'].get('wandb_project', 'gammanet'),
                    config=self.config,
                    name=f"gammanet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
    def prepare_for_training(self, train_loader: DataLoader, val_loader: DataLoader):
        """Prepare model, optimizers, and dataloaders for distributed training.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
        """
        # Prepare with accelerator
        self.model, self.optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader
        )
        
        if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler = self.accelerator.prepare(self.scheduler)
            
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch}",
            disable=not self.accelerator.is_local_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Reset hidden states at the beginning of each sequence
            self.model.module.reset_hidden_states() if hasattr(self.model, 'module') else self.model.reset_hidden_states()
            
            # Forward pass
            images = batch['image']
            targets = batch['edges']
            
            with self.accelerator.accumulate(self.model):
                predictions = self.model(images)
                loss = self.criterion(predictions, targets)
                # import pdb;pdb.set_trace()
                # from matplotlib import pyplot as plt
                # im = images.cpu()
                # im = (im.permute(0,2,3,1).squeeze() - im.permute(0,2,3,1).squeeze().min(0)[0].min(0)[0]) / (im.permute(0,2,3,1).squeeze().max(0)[0].max(0)[0] - im.permute(0,2,3,1).squeeze().min(0)[0].min(0)[0])
                # plt.subplot(121);plt.imshow(im);plt.subplot(122);plt.imshow(torch.sigmoid(predictions.squeeze().detach().cpu()));plt.savefig("temp.png");plt.show()
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.config['training'].get('grad_clip', 0) > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )
                    
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Convert predictions to probabilities for metrics
            with torch.no_grad():
                probs = torch.sigmoid(predictions)
                self.train_metrics.update(probs, targets)
                
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log to wandb
            if self.accelerator.is_main_process and self.config['logging'].get('wandb', False):
                log_dict = {
                    'train/loss': loss.item(),
                    'global_step': self.global_step
                }
                
                # Log learning rates for all parameter groups
                if len(self.optimizer.param_groups) > 1:
                    # Multiple learning rates (VGG + fGRU)
                    log_dict['train/lr_vgg'] = self.optimizer.param_groups[0]['lr']  # VGG (first group)
                    log_dict['train/lr_fgru'] = self.optimizer.param_groups[1]['lr']  # fGRU (second group)
                else:
                    # Single learning rate
                    log_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                
                wandb.log(log_dict)
                
            self.global_step += 1
            
        # Compute epoch metrics
        metrics = self.train_metrics.compute()
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        first_batch = None
        first_predictions = None
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", 
                         disable=not self.accelerator.is_local_main_process)):
            # Reset hidden states
            self.model.module.reset_hidden_states() if hasattr(self.model, 'module') else self.model.reset_hidden_states()
            
            images = batch['image']
            targets = batch['edges']
            
            predictions = self.model(images)
            loss = self.criterion(predictions, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Save first batch for image logging
            if batch_idx == 0:
                first_batch = batch
                first_predictions = predictions.detach()
            
            # Convert to probabilities for metrics
            probs = torch.sigmoid(predictions)
            self.val_metrics.update(probs, targets)
            
        # Compute metrics
        metrics = self.val_metrics.compute()
        metrics['loss'] = total_loss / num_batches
        
        # Log validation images if it's the right epoch
        if first_batch is not None and self.current_epoch % self.config['logging'].get('log_images_freq', 5) == 0:
            self.log_validation_images(first_batch, first_predictions, self.current_epoch)
        
        return metrics
    
    def log_validation_images(self, batch: Dict[str, torch.Tensor], predictions: torch.Tensor, epoch: int):
        """Log validation images to wandb.
        
        Args:
            batch: Validation batch containing images and targets
            predictions: Model predictions (logits)
            epoch: Current epoch number
        """
        if not self.config['logging'].get('wandb', False):
            return
            
        if not self.accelerator.is_main_process:
            return
            
        # Get configuration
        num_images = min(
            self.config['logging'].get('num_images_to_log', 8),
            batch['image'].shape[0]
        )
        
        # Get data
        images = batch['image'][:num_images].cpu()
        targets = batch['edges'][:num_images].cpu()
        preds = torch.sigmoid(predictions[:num_images]).cpu()
        
        # Denormalize images for visualization
        mean = torch.tensor(self.config['data']['normalize_mean']).view(1, 3, 1, 1)
        std = torch.tensor(self.config['data']['normalize_std']).view(1, 3, 1, 1)
        images_denorm = images * std + mean
        images_denorm = torch.clamp(images_denorm, 0, 1)
        
        # Create wandb images
        wandb_images = []
        for i in range(num_images):
            # Convert to numpy and rearrange dimensions
            img = images_denorm[i].permute(1, 2, 0).numpy()
            gt = targets[i, 0].numpy()
            pred = preds[i, 0].numpy()
            
            # Create a figure with subplots
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Original image
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(gt, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            # Thresholded prediction
            pred_thresh = (pred > 0.5).astype(float)
            axes[3].imshow(pred_thresh, cmap='gray', vmin=0, vmax=1)
            axes[3].set_title('Prediction (>0.5)')
            axes[3].axis('off')
            
            plt.tight_layout()
            
            # Convert to wandb image
            wandb_images.append(wandb.Image(fig, caption=f"Sample {i+1}"))
            plt.close(fig)
        
        # Log to wandb
        wandb.log({
            'validation/predictions': wandb_images,
            'epoch': epoch
        })
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        if not self.accelerator.is_main_process:
            return
            
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
        # Keep only last N checkpoints
        self._cleanup_checkpoints()
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
    def _cleanup_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoints) > keep_last:
            for ckpt in checkpoints[:-keep_last]:
                ckpt.unlink()
                
    def fit(self, num_epochs: Optional[int] = None):
        """Main training loop.
        
        Args:
            num_epochs: Number of epochs to train (overrides config if provided)
        """
        num_epochs = num_epochs or self.config['training']['num_epochs']
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            if epoch % self.config['training'].get('validation_frequency', 1) == 0:
                val_metrics = self.validate()
                
                # Log metrics
                if self.accelerator.is_main_process:
                    self.accelerator.print(f"Epoch {epoch}:")
                    self.accelerator.print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                                         f"ODS: {train_metrics['ods_f1']:.4f}")
                    self.accelerator.print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                                         f"ODS: {val_metrics['ods_f1']:.4f}, "
                                         f"OIS: {val_metrics['ois_f1']:.4f}")
                    
                    if self.config['logging'].get('wandb', False):
                        wandb.log({
                            'epoch': epoch,
                            'train/ods_f1': train_metrics['ods_f1'],
                            'val/loss': val_metrics['loss'],
                            'val/ods_f1': val_metrics['ods_f1'],
                            'val/ois_f1': val_metrics['ois_f1'],
                            'val/ap': val_metrics['ap']
                        })
                        
                # Check for improvement
                metric_name = self.config['training'].get('early_stopping_metric', 'ods_f1')
                # Map config names to metric keys
                metric_map = {
                    'f1_score': 'ods_f1',
                    'ods': 'ods_f1',
                    'ois': 'ois_f1',
                    'loss': 'loss'
                }
                metric_key = metric_map.get(metric_name, metric_name)
                metric = val_metrics[metric_key]
                
                # For loss, lower is better
                if metric_key == 'loss':
                    is_best = metric < self.best_metric
                else:
                    is_best = metric > self.best_metric
                
                if is_best:
                    self.best_metric = metric
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                # Save checkpoint
                if epoch % self.config['training'].get('save_frequency', 5) == 0:
                    self.save_checkpoint(is_best)
                    
                # Early stopping
                if self.patience_counter >= self.config['training'].get('early_stopping_patience', 30):
                    self.accelerator.print(f"Early stopping triggered after {epoch} epochs")
                    break
                    
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['ods_f1'])
                else:
                    self.scheduler.step()
                    
        # Save final checkpoint
        self.save_checkpoint()
        
        if self.config['logging'].get('wandb', False):
            wandb.finish()
