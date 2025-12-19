"""
Training script for Age-Invariant Face Recognition
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
import json

import config
from models import create_arcface_model, create_magface_model
from utils.metrics import evaluate_verification
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """Trainer for face recognition models"""
    
    def __init__(self, model_type='arcface', num_classes=None, backbone='magface', use_temporal=False):
        """
        Args:
            model_type: 'arcface' or 'magface'
            num_classes: Number of identity classes
            backbone: 'arcface' or 'magface' - backbone model
            use_temporal: Whether to use temporal training (default: False)
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.backbone = backbone
        self.use_temporal = use_temporal
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        
        # Create model
        if model_type == 'arcface':
            self.model = create_arcface_model(num_classes=num_classes)
        elif model_type == 'magface':
            self.model = create_magface_model(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'arcface' or 'magface'")
        
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = SGD(
            self.model.parameters(),
            lr=config.TRAIN_CONFIG["learning_rate"],
            momentum=config.TRAIN_CONFIG["momentum"],
            weight_decay=config.TRAIN_CONFIG["weight_decay"]
        )
        
        # Scheduler
        self.scheduler = None
        
        # For tracking
        self.current_epoch = 0
        self.best_accuracy = 0.0
        
        # TensorBoard
        log_dir = os.path.join(config.LOG_ROOT, f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.writer = SummaryWriter(log_dir)
        
        print(f"Initialized {model_type} trainer")
        print(f"Device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_dataloader(self, train_df, val_df=None):
        """
        Setup data loaders
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
        """
        # Define transforms
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Use identity dataset for classification training
        from utils.data_processor import IdentityDataset, AgeDBDataset
        
        # Check if we have metadata DataFrame or annotation file
        if isinstance(train_df, pd.DataFrame) and 'identity' in train_df.columns:
            # Training with identity labels
            train_dataset = IdentityDataset(train_df, transform=train_transform)
            val_dataset = None  # No validation for now, will use pairs later
        else:
            # Use annotation file for verification pairs
            annotation_file = config.AGEDB_CONFIG["annotation_file"]
            train_dataset = AgeDBDataset(
                annotation_file=annotation_file,
                image_folder=config.AGEDB_CONFIG["image_folder"],
                transform=train_transform
            )
            val_dataset = AgeDBDataset(
                annotation_file=annotation_file,
                image_folder=config.AGEDB_CONFIG["image_folder"],
                transform=val_transform
            ) if val_df is not None else None
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.TRAIN_CONFIG["batch_size"],
            shuffle=True,
            num_workers=config.TRAIN_CONFIG["num_workers"],
            pin_memory=config.TRAIN_CONFIG["pin_memory"]
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.EVAL_CONFIG["batch_size"],
            shuffle=False,
            num_workers=config.TRAIN_CONFIG["num_workers"],
            pin_memory=config.TRAIN_CONFIG["pin_memory"]
        ) if val_dataset is not None else None
        
        print(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Validation samples: {len(val_dataset)}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            if self.use_temporal:
                # Temporal triplet: (anchor, positive, negative, labels)
                anchor, positive, negative, labels = batch
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, embeddings, temporal_emb, negative_emb = self.model(
                    anchor, labels, positive, negative
                )
                
                # Compute loss
                loss, loss_dict = self.model.compute_loss(
                    logits, labels, embeddings, temporal_emb, negative_emb
                )
            else:
                # Standard training with identity labels
                # Check if it's verification pairs (5 items) or identity labels (2 items)
                if len(batch) == 5:
                    # Verification pairs: (img1, img2, label, path1, path2)
                    img1, img2, label, _, _ = batch
                    images = img1.to(self.device)
                    # Use dummy labels as before (for compatibility)
                    labels = torch.zeros(images.size(0), dtype=torch.long).to(self.device)
                elif len(batch) == 2:
                    # Identity labels: (image, identity)
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} items")
                
                # Forward pass
                outputs = self.model(images, labels)
                
                # Handle different model outputs
                if self.model_type == 'magface':
                    # MagFace returns (logits, embeddings, magnitudes)
                    logits, embeddings, magnitudes = outputs
                else:
                    # ArcFace returns (logits, embeddings)
                    logits, embeddings = outputs
                
                # Compute loss
                criterion = nn.CrossEntropyLoss()
                loss = criterion(logits, labels)
                loss_dict = {'total_loss': loss.item()}
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN/Inf detected in loss at batch {batch_idx}")
                print(f"Loss value: {loss.item()}")
                # Skip this batch
                continue
            
            # Track losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = []
                loss_components[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_components = {key: np.mean(values) for key, values in loss_components.items()}
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/train', avg_loss, self.current_epoch)
        for key, value in avg_components.items():
            self.writer.add_scalar(f'Loss/{key}', value, self.current_epoch)
        
        return avg_loss, avg_components
    
    def validate(self):
        """Validate model"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        
        all_embeddings1 = []
        all_embeddings2 = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if self.use_temporal:
                    anchor, positive, negative, labels = batch
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    
                    emb1 = self.model.extract_features(anchor)
                    emb2 = self.model.extract_features(positive)
                    
                    # Handle MagFace returning (embeddings, magnitudes)
                    if isinstance(emb1, tuple):
                        emb1 = emb1[0]
                    if isinstance(emb2, tuple):
                        emb2 = emb2[0]
                    
                    all_embeddings1.append(emb1)
                    all_embeddings2.append(emb2)
                    all_labels.append(torch.ones(len(labels)))  # All positive pairs
                else:
                    img1, img2, label, _, _ = batch
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    
                    emb1 = self.model.extract_features(img1)
                    emb2 = self.model.extract_features(img2)
                    
                    # Handle MagFace returning (embeddings, magnitudes)
                    if isinstance(emb1, tuple):
                        emb1 = emb1[0]
                    if isinstance(emb2, tuple):
                        emb2 = emb2[0]
                    
                    all_embeddings1.append(emb1)
                    all_embeddings2.append(emb2)
                    all_labels.append(label)
        
        # Concatenate all batches
        all_embeddings1 = torch.cat(all_embeddings1, dim=0)
        all_embeddings2 = torch.cat(all_embeddings2, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Evaluate
        metrics = evaluate_verification(all_embeddings1, all_embeddings2, all_labels)
        
        # Log to tensorboard
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Val/{key}', value, self.current_epoch)
        
        return metrics
    
    def train(self, num_epochs):
        """
        Train model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
        """
        # Setup scheduler
        if config.TRAIN_CONFIG["lr_scheduler"] == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif config.TRAIN_CONFIG["lr_scheduler"] == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        print(f"\nStarting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Update model's epoch for warmup strategy (if temporal model)
            if hasattr(self.model, 'set_epoch'):
                self.model.set_epoch(epoch)
            
            # Train
            train_loss, loss_components = self.train_epoch()
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            for key, value in loss_components.items():
                print(f"    {key}: {value:.4f}")
            
            # Validate
            if self.val_loader is not None and epoch % 5 == 0:
                val_metrics = self.validate()
                if val_metrics:
                    print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                    print(f"  Val EER: {val_metrics['eer']:.4f}")
                    
                    # Save best model
                    if val_metrics['accuracy'] > self.best_accuracy:
                        self.best_accuracy = val_metrics['accuracy']
                        self.save_checkpoint('best_model.pth')
                        print(f"  Saved best model (accuracy: {self.best_accuracy:.4f})")
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save final model after training
        final_model_name = f'{self.model_type}_best_model.pth'
        self.save_checkpoint(final_model_name)
        print(f"\nSaved final model: {final_model_name}")
        
        print("\nTraining complete!")
        self.writer.close()
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': {
                'model_type': self.model_type,
                'use_temporal': self.use_temporal,
                'backbone': self.backbone if self.model_type == 'temporal' else None,
            }
        }
        
        save_path = os.path.join(config.MODEL_ROOT, filename)
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        load_path = os.path.join(config.MODEL_ROOT, filename)
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


if __name__ == "__main__":
    # Example: Train ArcFace model
    print("Training ArcFace baseline model...")
    
    # Load metadata (you would load your actual data here)
    # metadata_path = os.path.join(config.OUTPUT_ROOT, "processed_images", "metadata.csv")
    # if os.path.exists(metadata_path):
    #     metadata_df = pd.read_csv(metadata_path)
    #     train_df, val_df = create_train_val_split(metadata_df)
    # else:
    #     print("Metadata not found. Please run data_processor.py first")
    
    # For testing, create dummy trainer
    trainer = Trainer(model_type='arcface', num_classes=1000, use_temporal=False)
    
    # Setup data loaders (would use real data)
    # trainer.setup_dataloader(train_df, val_df)
    
    # Train
    # trainer.train(num_epochs=config.TRAIN_CONFIG["num_epochs"])
    
    print("Training script ready. Configure your data and run.")
