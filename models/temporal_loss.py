"""
Temporal Contrastive Loss for Age-Invariant Face Recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class TemporalContrastiveLoss(nn.Module):
    """
    Temporal Contrastive Loss (TCL)
    
    Pulls embeddings of the same person at different time points closer,
    while pushing embeddings of different people apart.
    
    L_TCL = (1 - cos(f(x_t), f(x_{t+Δt}))) + α * max(0, m - cos(f(x_t), f(x_neg)))
    """
    
    def __init__(self, margin=0.4, alpha=0.5):
        """
        Args:
            margin: Margin for negative pairs
            alpha: Weight for negative pair loss
        """
        super(TemporalContrastiveLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
    
    def forward(self, embeddings_t, embeddings_t_delta, embeddings_neg=None):
        """
        Args:
            embeddings_t: Embeddings at time t (batch_size, embedding_dim)
            embeddings_t_delta: Embeddings at time t+Δt (batch_size, embedding_dim)
            embeddings_neg: Embeddings of negative samples (batch_size, embedding_dim)
            
        Returns:
            loss: Temporal contrastive loss
        """
        # Normalize embeddings
        embeddings_t = F.normalize(embeddings_t, p=2, dim=1)
        embeddings_t_delta = F.normalize(embeddings_t_delta, p=2, dim=1)
        
        # Positive pair loss: minimize distance between same person at different times
        cos_pos = F.cosine_similarity(embeddings_t, embeddings_t_delta, dim=1)
        loss_pos = (1 - cos_pos).mean()
        
        # Negative pair loss: maximize distance between different people
        if embeddings_neg is not None:
            embeddings_neg = F.normalize(embeddings_neg, p=2, dim=1)
            cos_neg = F.cosine_similarity(embeddings_t, embeddings_neg, dim=1)
            loss_neg = torch.clamp(self.margin - cos_neg, min=0).mean()
            
            # Combined loss
            loss = loss_pos + self.alpha * loss_neg
        else:
            loss = loss_pos
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: ArcFace Loss + Temporal Contrastive Loss
    
    L_total = L_ArcFace + λ * L_TCL
    """
    
    def __init__(self, lambda_tcl=0.3, margin=0.4, alpha=0.5):
        """
        Args:
            lambda_tcl: Weight for temporal contrastive loss
            margin: Margin for TCL negative pairs
            alpha: Weight for TCL negative pair loss
        """
        super(CombinedLoss, self).__init__()
        self.lambda_tcl = lambda_tcl
        self.tcl = TemporalContrastiveLoss(margin=margin, alpha=alpha)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels, embeddings_t=None, embeddings_t_delta=None, 
                embeddings_neg=None):
        """
        Args:
            logits: Classification logits from ArcFace (batch_size, num_classes)
            labels: Ground truth labels (batch_size,)
            embeddings_t: Embeddings at time t (optional)
            embeddings_t_delta: Embeddings at time t+Δt (optional)
            embeddings_neg: Negative embeddings (optional)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # ArcFace classification loss
        loss_arcface = self.ce_loss(logits, labels)
        
        # Temporal contrastive loss (if temporal pairs provided)
        if embeddings_t is not None and embeddings_t_delta is not None:
            loss_tcl = self.tcl(embeddings_t, embeddings_t_delta, embeddings_neg)
            total_loss = loss_arcface + self.lambda_tcl * loss_tcl
        else:
            loss_tcl = torch.tensor(0.0, device=logits.device)
            total_loss = loss_arcface
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'arcface_loss': loss_arcface.item(),
            'tcl_loss': loss_tcl.item()
        }
        
        return total_loss, loss_dict


class TripletTemporalLoss(nn.Module):
    """
    Triplet-based Temporal Loss
    
    Uses triplet loss formulation for temporal awareness
    """
    
    def __init__(self, margin=0.5):
        """
        Args:
            margin: Margin for triplet loss
        """
        super(TripletTemporalLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Anchor embeddings (time t)
            positive: Positive embeddings (same person, time t+Δt)
            negative: Negative embeddings (different person)
            
        Returns:
            loss: Triplet temporal loss
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Calculate distances
        dist_pos = 1 - F.cosine_similarity(anchor, positive, dim=1)
        dist_neg = 1 - F.cosine_similarity(anchor, negative, dim=1)
        
        # Triplet loss
        loss = torch.clamp(dist_pos - dist_neg + self.margin, min=0).mean()
        
        return loss


class MagnitudeRegularizationLoss(nn.Module):
    """
    Regularization loss for feature magnitude (inspired by MagFace)
    
    Encourages feature magnitudes to be within a desirable range
    """
    
    def __init__(self, l_a=10, u_a=110, lambda_g=0.35):
        """
        Args:
            l_a: Lower bound of feature magnitude
            u_a: Upper bound of feature magnitude
            lambda_g: Weight for magnitude regularization
        """
        super(MagnitudeRegularizationLoss, self).__init__()
        self.l_a = l_a
        self.u_a = u_a
        self.lambda_g = lambda_g
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: Feature embeddings (batch_size, embedding_dim)
            
        Returns:
            loss: Magnitude regularization loss
        """
        # Calculate magnitude
        magnitudes = torch.norm(embeddings, p=2, dim=1)
        
        # Penalize magnitudes outside [l_a, u_a]
        loss_lower = torch.clamp(self.l_a - magnitudes, min=0)
        loss_upper = torch.clamp(magnitudes - self.u_a, min=0)
        
        loss = self.lambda_g * (loss_lower + loss_upper).mean()
        
        return loss


class TemporalAwareLoss(nn.Module):
    """
    Complete temporal-aware loss combining multiple objectives
    """
    
    def __init__(self, lambda_tcl=0.3, lambda_mag=0.1, margin=0.4, alpha=0.5):
        """
        Args:
            lambda_tcl: Weight for temporal contrastive loss
            lambda_mag: Weight for magnitude regularization
            margin: Margin for TCL
            alpha: Weight for TCL negative pairs
        """
        super(TemporalAwareLoss, self).__init__()
        
        self.lambda_tcl = lambda_tcl
        self.lambda_mag = lambda_mag
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.tcl = TemporalContrastiveLoss(margin=margin, alpha=alpha)
        self.mag_reg = MagnitudeRegularizationLoss(
            l_a=config.MAGFACE_CONFIG["l_a"],
            u_a=config.MAGFACE_CONFIG["u_a"],
            lambda_g=lambda_mag
        )
    
    def forward(self, logits, labels, embeddings, embeddings_t=None, 
                embeddings_t_delta=None, embeddings_neg=None):
        """
        Complete loss computation
        
        Args:
            logits: Classification logits
            labels: Ground truth labels
            embeddings: Current embeddings
            embeddings_t: Temporal anchor embeddings
            embeddings_t_delta: Temporal positive embeddings
            embeddings_neg: Negative embeddings
            
        Returns:
            total_loss, loss_dict
        """
        # Classification loss
        loss_cls = self.ce_loss(logits, labels)
        
        # Temporal contrastive loss
        if embeddings_t is not None and embeddings_t_delta is not None:
            loss_tcl = self.tcl(embeddings_t, embeddings_t_delta, embeddings_neg)
        else:
            loss_tcl = torch.tensor(0.0, device=logits.device)
        
        # Magnitude regularization
        loss_mag = self.mag_reg(embeddings)
        
        # Total loss
        total_loss = loss_cls + self.lambda_tcl * loss_tcl + loss_mag
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'classification_loss': loss_cls.item(),
            'temporal_contrastive_loss': loss_tcl.item(),
            'magnitude_regularization': loss_mag.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test losses
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 16
    embedding_dim = 512
    num_classes = 1000
    
    # Dummy data
    embeddings_t = torch.randn(batch_size, embedding_dim).to(device)
    embeddings_t_delta = torch.randn(batch_size, embedding_dim).to(device)
    embeddings_neg = torch.randn(batch_size, embedding_dim).to(device)
    logits = torch.randn(batch_size, num_classes).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # Test Temporal Contrastive Loss
    print("Testing Temporal Contrastive Loss:")
    tcl = TemporalContrastiveLoss(margin=0.4, alpha=0.5).to(device)
    loss = tcl(embeddings_t, embeddings_t_delta, embeddings_neg)
    print(f"  TCL Loss: {loss.item():.4f}")
    
    # Test Combined Loss
    print("\nTesting Combined Loss:")
    combined_loss = CombinedLoss(lambda_tcl=0.3).to(device)
    total_loss, loss_dict = combined_loss(
        logits, labels, embeddings_t, embeddings_t_delta, embeddings_neg
    )
    print(f"  Total Loss: {loss_dict['total_loss']:.4f}")
    print(f"  ArcFace Loss: {loss_dict['arcface_loss']:.4f}")
    print(f"  TCL Loss: {loss_dict['tcl_loss']:.4f}")
    
    # Test Temporal-Aware Loss
    print("\nTesting Temporal-Aware Loss:")
    ta_loss = TemporalAwareLoss(lambda_tcl=0.3, lambda_mag=0.1).to(device)
    total_loss, loss_dict = ta_loss(
        logits, labels, embeddings_t, embeddings_t, embeddings_t_delta, embeddings_neg
    )
    print(f"  Total Loss: {loss_dict['total_loss']:.4f}")
    print(f"  Classification Loss: {loss_dict['classification_loss']:.4f}")
    print(f"  Temporal Contrastive Loss: {loss_dict['temporal_contrastive_loss']:.4f}")
    print(f"  Magnitude Regularization: {loss_dict['magnitude_regularization']:.4f}")
