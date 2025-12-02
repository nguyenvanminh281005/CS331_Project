"""
Temporal-Aware Face Recognition Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.arcface_model import ArcFaceModel
from models.magface_model import MagFaceModel
from models.temporal_loss import TemporalAwareLoss, CombinedLoss
import config


class TemporalAwareModel(nn.Module):
    """
    Temporal-Aware Face Recognition Model
    
    Extends ArcFace with temporal contrastive learning
    """
    
    def __init__(self, embedding_size=512, num_classes=None, dropout=0.0,
                 use_temporal_loss=True, lambda_tcl=0.3, backbone='magface'):
        """
        Args:
            embedding_size: Size of feature embeddings
            num_classes: Number of identity classes
            dropout: Dropout rate
            use_temporal_loss: Whether to use temporal contrastive loss
            lambda_tcl: Weight for temporal contrastive loss
            backbone: 'arcface' or 'magface' - which backbone model to use
        """
        super(TemporalAwareModel, self).__init__()
        
        # Base model - choose backbone
        if backbone == 'arcface':
            self.base_model = ArcFaceModel(
                embedding_size=embedding_size,
                num_classes=num_classes,
                dropout=dropout
            )
        elif backbone == 'magface':
            self.base_model = MagFaceModel(
                embedding_size=embedding_size,
                num_classes=num_classes,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Choose 'arcface' or 'magface'.")
        
        self.backbone = backbone
        self.use_temporal_loss = use_temporal_loss
        self.num_classes = num_classes
        
        # Loss function
        if use_temporal_loss:
            self.criterion = TemporalAwareLoss(
                lambda_tcl=lambda_tcl,
                lambda_mag=0.1,
                margin=config.TEMPORAL_CONFIG["margin"],
                alpha=config.TEMPORAL_CONFIG["alpha"]
            )
        else:
            self.criterion = CombinedLoss(
                lambda_tcl=lambda_tcl,
                margin=config.TEMPORAL_CONFIG["margin"],
                alpha=config.TEMPORAL_CONFIG["alpha"]
            )
    
    def forward(self, x, labels=None, x_temporal=None, x_negative=None):
        """
        Forward pass with temporal learning
        
        Args:
            x: Input images at time t (batch_size, 3, 112, 112)
            labels: Ground truth labels (batch_size,)
            x_temporal: Images at time t+Δt (batch_size, 3, 112, 112) - optional
            x_negative: Negative samples (batch_size, 3, 112, 112) - optional
            
        Returns:
            If training: logits, embeddings, (temporal_embeddings if provided)
            If inference: embeddings
        """
        # Extract features from main input
        if self.training and labels is not None:
            # Handle different return values based on backbone
            result = self.base_model(x, labels)
            if len(result) == 3:
                # MagFace returns (logits, embeddings, magnitudes)
                logits, embeddings, magnitudes = result
            else:
                # ArcFace returns (logits, embeddings)
                logits, embeddings = result
        else:
            # Inference mode
            result = self.base_model(x)
            logits = None
            if isinstance(result, tuple):
                # Handle tuple return (embeddings, magnitudes or quality_scores)
                embeddings = result[0]
            else:
                embeddings = result
        
        # Extract temporal features if provided
        temporal_embeddings = None
        negative_embeddings = None
        
        if x_temporal is not None:
            if self.training and labels is not None:
                # Don't need logits for temporal samples
                self.base_model.eval()
                with torch.no_grad():
                    result = self.base_model(x_temporal)
                    temporal_embeddings = result[0] if isinstance(result, tuple) else result
                self.base_model.train()
            else:
                result = self.base_model(x_temporal)
                temporal_embeddings = result[0] if isinstance(result, tuple) else result
        
        if x_negative is not None:
            if self.training and labels is not None:
                self.base_model.eval()
                with torch.no_grad():
                    result = self.base_model(x_negative)
                    negative_embeddings = result[0] if isinstance(result, tuple) else result
                self.base_model.train()
            else:
                result = self.base_model(x_negative)
                negative_embeddings = result[0] if isinstance(result, tuple) else result
        
        if self.training and labels is not None:
            return logits, embeddings, temporal_embeddings, negative_embeddings
        else:
            return embeddings
    
    def compute_loss(self, logits, labels, embeddings, temporal_embeddings=None, 
                    negative_embeddings=None):
        """
        Compute loss
        
        Args:
            logits: Classification logits
            labels: Ground truth labels
            embeddings: Current embeddings
            temporal_embeddings: Temporal pair embeddings
            negative_embeddings: Negative pair embeddings
            
        Returns:
            total_loss, loss_dict
        """
        if self.use_temporal_loss:
            return self.criterion(
                logits, labels, embeddings,
                embeddings_t=embeddings,
                embeddings_t_delta=temporal_embeddings,
                embeddings_neg=negative_embeddings
            )
        else:
            return self.criterion(
                logits, labels,
                embeddings_t=embeddings,
                embeddings_t_delta=temporal_embeddings,
                embeddings_neg=negative_embeddings
            )
    
    def extract_features(self, x):
        """Extract normalized feature embeddings"""
        self.eval()
        with torch.no_grad():
            result = self.base_model.extract_features(x)
            
            # Handle different return types based on backbone
            if isinstance(result, tuple):
                # MagFace returns (embeddings, quality_scores)
                embeddings = result[0]
            else:
                # ArcFace returns just embeddings
                embeddings = result
            
            # Ensure embeddings are 2D (batch_size, embedding_size)
            if embeddings.dim() == 3:
                # If base model returns (batch_size, 1, embedding_size), squeeze
                embeddings = embeddings.squeeze(1)
        return embeddings


class TemporalDataLoader:
    """
    Data loader that provides temporal triplets
    """
    
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=4):
        """
        Args:
            dataset: Dataset with temporal pairs
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of worker processes
        """
        from torch.utils.data import DataLoader
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


class TemporalTripletDataset(torch.utils.data.Dataset):
    """
    Dataset that yields temporal triplets (anchor, positive, negative)
    """
    
    def __init__(self, pairs_df, transform=None):
        """
        Args:
            pairs_df: DataFrame with columns [enrollment_path, probe_path, label, identity, ...]
            transform: Image transformations
        """
        import pandas as pd
        import cv2
        
        self.pairs_df = pairs_df
        self.transform = transform
        
        # Separate positive and negative pairs
        self.positive_pairs = pairs_df[pairs_df['label'] == 1].reset_index(drop=True)
        self.negative_pairs = pairs_df[pairs_df['label'] == 0].reset_index(drop=True)
        
        print(f"TemporalTripletDataset initialized:")
        print(f"  Total pairs: {len(pairs_df)}")
        print(f"  Positive pairs: {len(self.positive_pairs)}")
        print(f"  Negative pairs: {len(self.negative_pairs)}")
        
        if len(self.positive_pairs) == 0:
            raise ValueError("No positive pairs found in dataset!")
        if len(self.negative_pairs) == 0:
            raise ValueError("No negative pairs found in dataset!")
    
    def __len__(self):
        return len(self.positive_pairs)
    
    def __getitem__(self, idx):
        """
        Returns:
            anchor: Image at time t
            positive: Same person at time t+Δt
            negative: Different person
            label: Identity label
        """
        import cv2
        
        # Get positive pair (same person, different time)
        pos_row = self.positive_pairs.iloc[idx]
        
        # Load anchor and positive
        anchor_img = self._load_image(pos_row['enrollment_path'])
        positive_img = self._load_image(pos_row['probe_path'])
        
        # Get random negative pair
        neg_row = self.negative_pairs.sample(1).iloc[0]
        negative_img = self._load_image(neg_row['probe_path'])
        
        # Apply transformations
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        # Get label (identity) - use the identity column directly
        if 'identity' in pos_row.index:
            label = int(pos_row['identity'])
        else:
            # Fallback: extract from path or use 0
            label = 0
        
        return anchor_img, positive_img, negative_img, label
    
    def _load_image(self, path):
        """Load and preprocess image"""
        import cv2
        import numpy as np
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def create_temporal_model(num_classes=None, pretrained=False, checkpoint_path=None,
                         use_temporal_loss=True, backbone='magface'):
    """
    Create Temporal-Aware model
    
    Args:
        num_classes: Number of identity classes
        pretrained: Load pretrained weights
        checkpoint_path: Path to checkpoint file
        use_temporal_loss: Whether to use temporal contrastive loss
        backbone: 'arcface' or 'magface' - which backbone model to use
        
    Returns:
        Temporal-Aware model
    """
    checkpoint = None
    
    # If loading from checkpoint, extract info
    if pretrained and checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Infer num_classes if not provided
        if num_classes is None:
            if 'base_model.margin.weight' in checkpoint['model_state_dict']:
                num_classes = checkpoint['model_state_dict']['base_model.margin.weight'].shape[0]
                print(f"Inferred num_classes={num_classes} from checkpoint")
        
        # Infer backbone from checkpoint structure if not in config
        if 'config' in checkpoint and 'backbone' in checkpoint['config'] and checkpoint['config']['backbone']:
            backbone = checkpoint['config']['backbone']
            print(f"Loaded backbone from checkpoint config: {backbone}")
        else:
            # Try to infer from model keys
            state_dict_keys = checkpoint['model_state_dict'].keys()
            has_bn3 = any('bn3' in k for k in state_dict_keys)
            has_fc_bias = 'base_model.fc.bias' in state_dict_keys
            
            if has_bn3 and has_fc_bias:
                inferred_backbone = 'magface'
                print(f"Inferred backbone=magface from checkpoint (has bn3 and fc.bias)")
                backbone = inferred_backbone
            elif not has_bn3 and not has_fc_bias:
                inferred_backbone = 'arcface'
                print(f"Inferred backbone=arcface from checkpoint (no bn3, no fc.bias)")
                backbone = inferred_backbone
    
    model = TemporalAwareModel(
        embedding_size=config.MODEL_CONFIG["embedding_size"],
        num_classes=num_classes,
        dropout=config.MODEL_CONFIG["dropout"],
        use_temporal_loss=use_temporal_loss,
        lambda_tcl=config.TEMPORAL_CONFIG["lambda_tcl"],
        backbone=backbone
    )
    
    if pretrained and checkpoint_path is not None:
        if checkpoint is None:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load state dict with strict=False to handle model structure changes
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print(f"Loaded pretrained model from {checkpoint_path}")
        except RuntimeError as e:
            print(f"Warning: Could not load with strict=True: {e}")
            print("Loading with strict=False (ignoring mismatched keys)...")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
            print(f"Loaded pretrained model from {checkpoint_path} (with warnings)")
    
    return model


if __name__ == "__main__":
    # Test temporal model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_temporal_model(num_classes=1000, use_temporal_loss=True)
    model = model.to(device)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 112, 112).to(device)
    x_temporal = torch.randn(batch_size, 3, 112, 112).to(device)
    x_negative = torch.randn(batch_size, 3, 112, 112).to(device)
    labels = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # Training mode
    model.train()
    logits, embeddings, temporal_emb, negative_emb = model(
        x, labels, x_temporal, x_negative
    )
    
    print(f"Training mode:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Temporal embeddings shape: {temporal_emb.shape}")
    print(f"  Negative embeddings shape: {negative_emb.shape}")
    
    # Compute loss
    loss, loss_dict = model.compute_loss(
        logits, labels, embeddings, temporal_emb, negative_emb
    )
    print(f"\nLoss computation:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        features = model.extract_features(x)
    print(f"\nInference mode:")
    print(f"  Features shape: {features.shape}")
    print(f"  Feature norm: {torch.norm(features, p=2, dim=1)}")
