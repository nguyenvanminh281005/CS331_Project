"""
MagFace Model Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config


class MagLinear(nn.Module):
    """
    MagFace: A Universal Representation for Face Recognition and Quality Assessment
    Paper: https://arxiv.org/abs/2103.06627
    """
    
    def __init__(self, in_features, out_features, s=64.0, l_a=10, u_a=110, 
                 l_margin=0.45, u_margin=0.8):
        """
        Args:
            in_features: Size of input features (embedding size)
            out_features: Size of output features (number of classes)
            s: Scale parameter
            l_a: Lower bound of feature magnitude
            u_a: Upper bound of feature magnitude
            l_margin: Lower bound of margin
            u_margin: Upper bound of margin
        """
        super(MagLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.l_a = l_a
        self.u_a = u_a
        self.l_margin = l_margin
        self.u_margin = u_margin
        
        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def _calc_adaptive_margin(self, x_norm):
        """
        Calculate adaptive margin based on feature magnitude
        
        Args:
            x_norm: Feature magnitude (batch_size,)
            
        Returns:
            Adaptive margin (batch_size,)
        """
        # Linear interpolation between l_margin and u_margin
        # based on feature magnitude
        margin = (self.u_margin - self.l_margin) / (self.u_a - self.l_a) * (x_norm - self.l_a) + self.l_margin
        
        # Clip to valid range
        margin = torch.clamp(margin, self.l_margin, self.u_margin)
        
        return margin
    
    def forward(self, input, label):
        """
        Args:
            input: Feature embeddings (batch_size, in_features)
            label: Ground truth labels (batch_size,)
            
        Returns:
            output: Logits (batch_size, out_features)
            x_norm: Feature magnitudes (for quality assessment)
        """
        # Calculate feature magnitude
        x_norm = torch.norm(input, p=2, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        
        # Normalize input
        x_normalized = input / x_norm
        
        # Normalize weight
        w_normalized = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(x_normalized, w_normalized)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Calculate adaptive margin for each sample
        margin = self._calc_adaptive_margin(x_norm.squeeze())
        
        # Expand margin to match cosine shape
        cos_m = torch.cos(margin).view(-1, 1)
        sin_m = torch.sin(margin).view(-1, 1)
        
        # cos(theta + m)
        phi = cosine * cos_m - sine * sin_m
        
        # Convert labels to one-hot
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply margin to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale by feature magnitude and s
        output = output * self.s * (x_norm / self.u_a).squeeze().view(-1, 1)
        
        return output, x_norm.squeeze()


class MagFaceModel(nn.Module):
    """
    MagFace model with ResNet backbone
    """
    
    def __init__(self, embedding_size=512, num_classes=None, dropout=0.0):
        """
        Args:
            embedding_size: Size of feature embeddings
            num_classes: Number of identity classes (for training)
            dropout: Dropout rate
        """
        super(MagFaceModel, self).__init__()
        
        self.embedding_size = embedding_size
        
        # Import ResNet blocks from ArcFace model
        from models.arcface_model import ResNetBlock
        
        # Backbone: ResNet-like architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet blocks
        self.layer1 = self._make_layer(ResNetBlock, 64, 64, 2, stride=2)
        self.layer2 = self._make_layer(ResNetBlock, 64, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResNetBlock, 128, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResNetBlock, 256, 512, 2, stride=2)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Batch normalization
        self.bn2 = nn.BatchNorm1d(512)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Embedding layer (no normalization - magnitude matters!)
        self.fc = nn.Linear(512, embedding_size, bias=False)
        
        # MagFace margin (only used during training)
        self.num_classes = num_classes
        if num_classes is not None:
            self.margin = MagLinear(
                embedding_size,
                num_classes,
                s=config.MAGFACE_CONFIG["scale"],
                l_a=config.MAGFACE_CONFIG["l_a"],
                u_a=config.MAGFACE_CONFIG["u_a"],
                l_margin=config.MAGFACE_CONFIG["l_margin"],
                u_margin=config.MAGFACE_CONFIG["u_margin"]
            )
    
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks"""
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x, labels=None):
        """
        Args:
            x: Input images (batch_size, 3, 112, 112)
            labels: Ground truth labels (batch_size,) - only for training
            
        Returns:
            If training (labels provided): logits, embeddings, magnitudes
            If inference (no labels): embeddings, magnitudes
        """
        # Backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        x = self.bn2(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Embedding (DO NOT normalize - magnitude is important!)
        embeddings = self.fc(x)
        
        # Calculate magnitude
        magnitudes = torch.norm(embeddings, p=2, dim=1)
        
        # Training mode: apply MagFace margin
        if self.training and labels is not None and self.num_classes is not None:
            logits, magnitudes = self.margin(embeddings, labels)
            return logits, embeddings, magnitudes
        
        # Inference mode: return embeddings and magnitudes
        else:
            return embeddings, magnitudes
    
    def extract_features(self, x):
        """
        Extract feature embeddings and quality scores
        
        Returns:
            embeddings: Feature vectors
            quality_scores: Quality scores based on feature magnitude
        """
        self.eval()
        with torch.no_grad():
            embeddings, magnitudes = self.forward(x)
            
            # Calculate quality score (normalize magnitude to 0-1)
            quality_scores = (magnitudes - config.MAGFACE_CONFIG["l_a"]) / \
                           (config.MAGFACE_CONFIG["u_a"] - config.MAGFACE_CONFIG["l_a"])
            quality_scores = torch.clamp(quality_scores, 0, 1)
        
        return embeddings, quality_scores


def create_magface_model(num_classes=None, pretrained=False, checkpoint_path=None):
    """
    Create MagFace model
    
    Args:
        num_classes: Number of identity classes
        pretrained: Load pretrained weights
        checkpoint_path: Path to checkpoint file
        
    Returns:
        MagFace model
    """
    # If loading from checkpoint, extract num_classes if not provided
    if pretrained and checkpoint_path is not None and num_classes is None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Try to infer num_classes from margin.weight shape
        if 'margin.weight' in checkpoint['model_state_dict']:
            num_classes = checkpoint['model_state_dict']['margin.weight'].shape[0]
            print(f"Inferred num_classes={num_classes} from checkpoint")
    
    model = MagFaceModel(
        embedding_size=config.MAGFACE_CONFIG["embedding_size"],
        num_classes=num_classes,
        dropout=config.MODEL_CONFIG["dropout"]
    )
    
    if pretrained and checkpoint_path is not None:
        if 'checkpoint' not in locals():
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained model from {checkpoint_path}")
    
    return model


if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_magface_model(num_classes=1000)
    model = model.to(device)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 112, 112).to(device)
    labels = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # Training mode
    model.train()
    logits, embeddings, magnitudes = model(x, labels)
    print(f"Training mode:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Magnitudes: {magnitudes}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        features, quality_scores = model.extract_features(x)
    print(f"\nInference mode:")
    print(f"  Features shape: {features.shape}")
    print(f"  Feature magnitudes: {torch.norm(features, p=2, dim=1)}")
    print(f"  Quality scores: {quality_scores}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
