"""
Script to verify if two face images belong to the same person using MagFace model
"""
import os
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

import config
from models import create_magface_model


class FaceVerifier:
    """Face verification using MagFace model"""
    
    def __init__(self, checkpoint_path=None, threshold=0.3):
        """
        Initialize face verifier
        
        Args:
            checkpoint_path: Path to MagFace checkpoint (default: use config path)
            threshold: Similarity threshold for same person (default: 0.3)
        """
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        
        # Default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.MODEL_ROOT, 'magface_best_model.pth')
        
        # Load model
        print(f"Loading MagFace model from: {checkpoint_path}")
        self.model = create_magface_model(pretrained=True, checkpoint_path=checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.threshold = threshold
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"Initialized FaceVerifier")
        print(f"Device: {self.device}")
        print(f"Threshold: {self.threshold}")
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
            
        Returns:
            tensor: Preprocessed image tensor
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 112x112 if needed
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        
        # Apply transform
        img_tensor = self.transform(img)
        
        return img_tensor
    
    def extract_feature(self, image_path):
        """
        Extract feature embedding from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            feature: Feature embedding (normalized)
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image_path)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Extract feature
        with torch.no_grad():
            feature = self.model.extract_features(img_tensor)
            
            # Handle MagFace returning (embeddings, magnitudes)
            if isinstance(feature, tuple):
                feature = feature[0]
            
            # Normalize feature
            feature = torch.nn.functional.normalize(feature, p=2, dim=1)
        
        return feature
    
    def compute_similarity(self, feature1, feature2):
        """
        Compute cosine similarity between two features
        
        Args:
            feature1: First feature embedding
            feature2: Second feature embedding
            
        Returns:
            similarity: Cosine similarity score
        """
        similarity = torch.nn.functional.cosine_similarity(feature1, feature2)
        return similarity.item()
    
    def verify(self, image_path1, image_path2, return_score=False):
        """
        Verify if two images belong to the same person
        
        Args:
            image_path1: Path to first image
            image_path2: Path to second image
            return_score: Whether to return similarity score
            
        Returns:
            is_same_person: Boolean indicating if same person
            similarity_score: (optional) Similarity score if return_score=True
        """
        # Extract features
        print(f"Processing image 1: {os.path.basename(image_path1)}")
        feature1 = self.extract_feature(image_path1)
        
        print(f"Processing image 2: {os.path.basename(image_path2)}")
        feature2 = self.extract_feature(image_path2)
        
        # Compute similarity
        similarity = self.compute_similarity(feature1, feature2)
        
        # Determine if same person
        is_same_person = similarity >= self.threshold
        
        print(f"\nSimilarity score: {similarity:.4f}")
        print(f"Threshold: {self.threshold:.4f}")
        print(f"Result: {'SAME PERSON ✓' if is_same_person else 'DIFFERENT PERSON ✗'}")
        
        if return_score:
            return is_same_person, similarity
        else:
            return is_same_person


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify if two face images belong to the same person')
    parser.add_argument('image1', type=str, help='Path to first image')
    parser.add_argument('image2', type=str, help='Path to second image')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to MagFace checkpoint (default: saved_models/magface_best_model.pth)')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Similarity threshold (default: 0.3)')
    
    args = parser.parse_args()
    
    # Check if images exist
    if not os.path.exists(args.image1):
        print(f"Error: Image not found: {args.image1}")
        return
    
    if not os.path.exists(args.image2):
        print(f"Error: Image not found: {args.image2}")
        return
    
    # Create verifier
    verifier = FaceVerifier(checkpoint_path=args.checkpoint, threshold=args.threshold)
    
    # Verify
    print("\n" + "="*60)
    print("FACE VERIFICATION")
    print("="*60)
    
    is_same, score = verifier.verify(args.image1, args.image2, return_score=True)
    
    print("\n" + "="*60)
    print(f"Confidence: {score*100:.2f}%")
    if is_same:
        print("✓ These images show the SAME person")
    else:
        print("✗ These images show DIFFERENT people")
    print("="*60)


if __name__ == "__main__":
    main()
