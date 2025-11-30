"""
ArcFace Model Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config


class ArcMarginProduct(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    Paper: https://arxiv.org/abs/1801.07698
    """
    
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        """
        Args:
            in_features: Size of input features (embedding size)
            out_features: Size of output features (number of classes)
            s: Scale parameter
            m: Margin parameter
            easy_margin: Use easy margin
        """
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, input, label):
        """
        Args:
            input: Feature embeddings (batch_size, in_features)
            label: Ground truth labels (batch_size,)
            
        Returns:
            output: Logits (batch_size, out_features)
        """
        # Normalize features and weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Convert labels to one-hot
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply margin to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output


class ResNetBlock(nn.Module):
    """Residual block for ResNet"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ArcFaceModel(nn.Module):
    """
    ArcFace model with ResNet backbone
    """
    
    def __init__(self, embedding_size=512, num_classes=None, dropout=0.0):
        """
        Args:
            embedding_size: Size of feature embeddings
            num_classes: Number of identity classes (for training)
            dropout: Dropout rate
        """
        super(ArcFaceModel, self).__init__()
        
        self.embedding_size = embedding_size
        
        # Backbone: ResNet-like architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Batch normalization for embeddings
        self.bn2 = nn.BatchNorm1d(512)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Embedding layer
        self.fc = nn.Linear(512, embedding_size)
        self.bn3 = nn.BatchNorm1d(embedding_size)
        
        # ArcFace margin (only used during training)
        self.num_classes = num_classes
        if num_classes is not None:
            self.margin = ArcMarginProduct(
                embedding_size,
                num_classes,
                s=config.ARCFACE_CONFIG["scale"],
                m=config.ARCFACE_CONFIG["margin"]
            )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a layer with multiple residual blocks"""
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x, labels=None):
        """
        Args:
            x: Input images (batch_size, 3, 112, 112)
            labels: Ground truth labels (batch_size,) - only for training
            
        Returns:
            If training (labels provided): logits
            If inference (no labels): normalized embeddings
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
        
        # Embedding
        embeddings = self.fc(x)
        embeddings = self.bn3(embeddings)
        
        # Training mode: apply ArcFace margin
        if self.training and labels is not None and self.num_classes is not None:
            logits = self.margin(embeddings, labels)
            return logits, embeddings
        
        # Inference mode: return normalized embeddings
        else:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
    
    def extract_features(self, x):
        """Extract normalized feature embeddings"""
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(x)
        return embeddings


def create_arcface_model(num_classes=None, pretrained=False, checkpoint_path=None):
    """
    Create ArcFace model
    
    Args:
        num_classes: Number of identity classes
        pretrained: Load pretrained weights
        checkpoint_path: Path to checkpoint file
        
    Returns:
        ArcFace model
    """
    # If loading from checkpoint, extract num_classes if not provided
    if pretrained and checkpoint_path is not None and num_classes is None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Try to infer num_classes from margin.weight shape
        if 'margin.weight' in checkpoint['model_state_dict']:
            num_classes = checkpoint['model_state_dict']['margin.weight'].shape[0]
            print(f"Inferred num_classes={num_classes} from checkpoint")
    
    model = ArcFaceModel(
        embedding_size=config.ARCFACE_CONFIG["embedding_size"],
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
    model = create_arcface_model(num_classes=1000)
    model = model.to(device)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 112, 112).to(device)
    labels = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # Training mode
    model.train()
    logits, embeddings = model(x, labels)
    print(f"Training mode:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        features = model(x)
    print(f"\nInference mode:")
    print(f"  Features shape: {features.shape}")
    print(f"  Feature norm: {torch.norm(features, p=2, dim=1)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
