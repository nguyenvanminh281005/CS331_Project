"""
Data Processing Pipeline for AgeDB-30 dataset
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import config
from utils.face_detector import FaceDetector


class IdentityDataset(Dataset):
    """Dataset for training with identity labels"""
    
    def __init__(self, metadata_df: pd.DataFrame, transform=None):
        """
        Args:
            metadata_df: DataFrame with columns [aligned_path, identity, age]
            transform: Optional transform to be applied
        """
        self.metadata_df = metadata_df
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        img_path = row['aligned_path']
        identity = int(row['identity'])
        
        # Load image
        img = self._load_image(img_path)
        
        if self.transform:
            img = self.transform(img)
        
        return img, identity
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load image from path"""
        if not os.path.exists(img_path):
            raise ValueError(f"Failed to load image: {img_path}")
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class AgeDBDataset(Dataset):
    """AgeDB-30 Dataset loader for verification pairs"""
    
    def __init__(self, annotation_file: str, image_folder: str, transform=None):
        """
        Args:
            annotation_file: Path to annotation file
            image_folder: Path to image folder
            transform: Optional transform to be applied
        """
        self.image_folder = image_folder
        self.transform = transform
        self.pairs = self._load_annotations(annotation_file)
        
    def _load_annotations(self, annotation_file: str) -> List[Tuple]:
        """
        Load annotation file
        
        Expected format:
        image1_path image2_path label
        or
        image1_path image2_path same_person
        """
        pairs = []
        
        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file not found: {annotation_file}")
            return pairs
        
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # Format: label image1_path image2_path
                    label, img1, img2 = int(parts[0]), parts[1], parts[2]
                    pairs.append((img1, img2, label))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        
        # Load images
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label, img1_path, img2_path
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load image from path"""
        # img_path is already a full path from the pairs CSV
        if not os.path.exists(img_path):
            # Try with image_folder prefix as fallback
            full_path = os.path.join(self.image_folder, os.path.basename(img_path))
            if not os.path.exists(full_path):
                raise ValueError(f"Failed to load image: {img_path}")
        else:
            full_path = img_path
        
        img = cv2.imread(full_path)
        if img is None:
            raise ValueError(f"Failed to load image: {full_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


class DataProcessor:
    """Process raw images with face detection and quality filtering"""
    
    def __init__(self, use_quality_filter=True, skip_detection=False):
        """
        Args:
            use_quality_filter: Whether to apply quality filtering
            skip_detection: Skip face detection (for pre-aligned images like agedb_30_112x112)
        """
        self.skip_detection = skip_detection
        if not skip_detection:
            self.detector = FaceDetector(
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.detector = None
        self.use_quality_filter = use_quality_filter
        
    def process_dataset(self, image_folder: str, output_folder: str,
                       annotation_file: str = None) -> pd.DataFrame:
        """
        Process entire dataset with face detection and alignment
        
        Args:
            image_folder: Input image folder
            output_folder: Output folder for processed images
            annotation_file: Optional annotation file for metadata
            
        Returns:
            DataFrame with processing results
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        import glob
        image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.JPG', '*.PNG', '*.JPEG', '*.BMP']:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        
        print(f"Found {len(image_files)} images to process")
        
        # Handle empty folder case
        if len(image_files) == 0:
            print(f"Warning: No images found in {image_folder}")
            print("Creating empty metadata file...")
            df = pd.DataFrame(columns=['image_path', 'output_path', 'confidence', 'quality_passed'])
            metadata_path = os.path.join(output_folder, 'metadata.csv')
            df.to_csv(metadata_path, index=False)
            return df
        
        results = []
        
        for img_path in tqdm(image_files, desc="Processing images"):
            result = self._process_single_image(img_path, output_folder)
            if result is not None:
                results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save metadata
        metadata_path = os.path.join(output_folder, 'metadata.csv')
        df.to_csv(metadata_path, index=False)
        
        print(f"\nProcessing complete:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Successfully processed: {len(results)}")
        print(f"  Success rate: {len(results)/len(image_files)*100:.2f}%")
        
        return df
    
    def _process_single_image(self, img_path: str, output_folder: str) -> Dict:
        """Process a single image"""
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Check if we should skip detection (pre-aligned images)
            if self.skip_detection:
                # Images are already aligned (e.g., agedb_30_112x112)
                aligned_face = img
                prob = 1.0
                landmarks = None
                
                # Basic quality check on pre-aligned images
                if self.use_quality_filter:
                    quality_score = self._calculate_quality_score(aligned_face)
                    if quality_score < 0.1:  # Very blurry
                        return None
            else:
                # Detect and align
                result = self.detector.detect_and_align(img)
                
                if result is None:
                    return None
                
                aligned_face, landmarks, prob = result
                
                # Quality check
                if self.use_quality_filter:
                    if not self.detector.quality_check(aligned_face, landmarks, prob):
                        return None
                
            # Calculate quality score (blur detection)
            quality_score = self._calculate_quality_score(aligned_face)
            
            # Save aligned face (copy to output folder)
            basename = os.path.basename(img_path)
            output_path = os.path.join(output_folder, basename)
            
            aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, aligned_face_bgr)
            
            # Extract image ID from filename (for agedb, it's just the number)
            image_id = self._parse_filename(basename)
            
            return {
                'image_id': image_id,
                'original_path': img_path,
                'aligned_path': output_path,
                'detection_prob': prob,
                'quality_score': quality_score,
                'landmarks': landmarks.tolist() if landmarks is not None else None
            }
            
        except Exception as e:
            # Silently skip failed images in batch processing
            return None
    
    def _calculate_quality_score(self, image: np.ndarray) -> float:
        """
        Calculate image quality score based on sharpness
        
        Args:
            image: Input image
            
        Returns:
            Quality score (0-1)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Laplacian variance for sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 (empirically determined range)
        quality_score = min(laplacian_var / 1000.0, 1.0)
        
        return quality_score
    
    def _parse_filename(self, filename: str) -> str:
        """
        Parse image ID from filename
        
        For agedb_30, files are named like 0.bmp, 1.bmp, etc.
        """
        # Remove extension
        name = os.path.splitext(filename)[0]
        return name
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [-1, 1] range
        
        Args:
            image: Input image (0-255)
            
        Returns:
            Normalized image
        """
        # Convert to float and normalize to [-1, 1]
        normalized = (image.astype(np.float32) - 127.5) / 128.0
        return normalized


def add_identity_from_annotations(metadata_df: pd.DataFrame, annotation_file: str) -> pd.DataFrame:
    """
    Add identity information to metadata using annotation file
    
    For AgeDB-30, pairs with label=1 are same person, label=0 are different people.
    We build identity clusters from positive pairs.
    
    Args:
        metadata_df: DataFrame with image metadata
        annotation_file: Path to annotation file with pairs
        
    Returns:
        metadata_df with 'identity' column added
    """
    # Create image_id to identity mapping
    image_id_to_identity = {}
    next_identity_id = 0
    
    # Read annotation file
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                label = int(parts[0])
                img1_path = parts[1]
                img2_path = parts[2]
                
                # Extract image IDs (remove folder prefix and extension)
                img1_id = os.path.splitext(os.path.basename(img1_path))[0]
                img2_id = os.path.splitext(os.path.basename(img2_path))[0]
                
                # Only process positive pairs (same person)
                if label == 1:
                    # Check if either image already has an identity
                    id1 = image_id_to_identity.get(img1_id)
                    id2 = image_id_to_identity.get(img2_id)
                    
                    if id1 is not None and id2 is not None:
                        # Both have identities - they should be the same
                        # If different, merge them (keep the smaller ID)
                        if id1 != id2:
                            old_id = max(id1, id2)
                            new_id = min(id1, id2)
                            # Update all images with old_id to new_id
                            for img_id, identity in image_id_to_identity.items():
                                if identity == old_id:
                                    image_id_to_identity[img_id] = new_id
                    elif id1 is not None:
                        # img1 has identity, assign to img2
                        image_id_to_identity[img2_id] = id1
                    elif id2 is not None:
                        # img2 has identity, assign to img1
                        image_id_to_identity[img1_id] = id2
                    else:
                        # Neither has identity, create new one
                        image_id_to_identity[img1_id] = next_identity_id
                        image_id_to_identity[img2_id] = next_identity_id
                        next_identity_id += 1
    
    # For images not in positive pairs, assign unique identities
    for idx, row in metadata_df.iterrows():
        img_id = str(row['image_id'])
        if img_id not in image_id_to_identity:
            image_id_to_identity[img_id] = next_identity_id
            next_identity_id += 1
    
    # Add identity column to metadata
    metadata_df['identity'] = metadata_df['image_id'].astype(str).map(image_id_to_identity)
    
    print(f"Identity mapping complete:")
    print(f"  Total images: {len(metadata_df)}")
    print(f"  Total identities: {metadata_df['identity'].nunique()}")
    
    return metadata_df


def create_train_val_split(metadata_df: pd.DataFrame, val_ratio=0.2, random_seed=42):
    """
    Split dataset into train and validation sets
    
    For verification datasets like AgeDB-30, we split by identities to avoid data leakage
    
    Args:
        metadata_df: DataFrame with image metadata (must have 'identity' column)
        val_ratio: Validation set ratio
        random_seed: Random seed for reproducibility
        
    Returns:
        train_df, val_df
    """
    np.random.seed(random_seed)
    
    # Split by identities to avoid data leakage
    identities = metadata_df['identity'].unique()
    np.random.shuffle(identities)
    
    split_idx = int(len(identities) * (1 - val_ratio))
    train_identities = set(identities[:split_idx])
    val_identities = set(identities[split_idx:])
    
    train_df = metadata_df[metadata_df['identity'].isin(train_identities)].reset_index(drop=True)
    val_df = metadata_df[metadata_df['identity'].isin(val_identities)].reset_index(drop=True)
    
    print(f"\nTrain/Val Split:")
    print(f"  Train set: {len(train_df)} images, {len(train_identities)} identities")
    print(f"  Val set: {len(val_df)} images, {len(val_identities)} identities")
    
    return train_df, val_df


if __name__ == "__main__":
    # Test data processor
    processor = DataProcessor(use_quality_filter=True)
    
    # Process AgeDB-30 dataset
    image_folder = config.AGEDB_CONFIG["image_folder"]
    output_folder = os.path.join(config.OUTPUT_ROOT, "processed_images")
    
    if os.path.exists(image_folder):
        df = processor.process_dataset(image_folder, output_folder)
        print("\nSample results:")
        print(df.head())
        
        # Create train/val split
        train_df, val_df = create_train_val_split(df)
    else:
        print(f"Image folder not found: {image_folder}")
