"""
MORPH-2 Dataset Processor
Handles loading and processing MORPH-2 dataset with age information
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import cv2
from tqdm import tqdm

import config


class MORPH2Processor:
    """Processor for MORPH-2 dataset"""
    
    def __init__(self, target_size=(112, 112)):
        """
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
        
    def load_metadata(self, split='train') -> pd.DataFrame:
        """
        Load MORPH-2 metadata from CSV files
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            df: DataFrame with columns [age, gender, filename, filepath, identity]
        """
        if split == 'train':
            csv_path = config.MORPH2_CONFIG["train_csv"]
        elif split == 'val':
            csv_path = config.MORPH2_CONFIG["val_csv"]
        elif split == 'test':
            csv_path = config.MORPH2_CONFIG["test_csv"]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Extract identity from filename (e.g., "00013_00M19.JPG" -> "00013")
        df['identity'] = df['filename'].apply(lambda x: x.split('_')[0])
        
        # Add image_id (use filename without extension)
        df['image_id'] = df['filename'].apply(lambda x: os.path.splitext(x)[0])
        
        print(f"Loaded {len(df)} samples from {split} split")
        print(f"  Age range: {df['age'].min()} - {df['age'].max()}")
        print(f"  Number of identities: {df['identity'].nunique()}")
        print(f"  Gender distribution: {df['gender'].value_counts().to_dict()}")
        
        return df
    
    def process_dataset(self, split='train', output_folder=None) -> pd.DataFrame:
        """
        Process MORPH-2 dataset and prepare for training
        
        Args:
            split: 'train', 'val', or 'test'
            output_folder: Folder to save processed images (optional)
            
        Returns:
            df: Processed metadata DataFrame
        """
        # Load metadata
        df = self.load_metadata(split)
        
        # If output folder is specified, copy and resize images
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            
            aligned_paths = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split} images"):
                src_path = row['filepath']
                
                if not os.path.exists(src_path):
                    print(f"Warning: Image not found: {src_path}")
                    aligned_paths.append(None)
                    continue
                
                # Read image
                img = cv2.imread(src_path)
                
                if img is None:
                    print(f"Warning: Failed to read image: {src_path}")
                    aligned_paths.append(None)
                    continue
                
                # Resize if needed
                if img.shape[:2] != self.target_size:
                    img = cv2.resize(img, (self.target_size[1], self.target_size[0]))
                
                # Save to output folder
                output_path = os.path.join(output_folder, row['filename'])
                cv2.imwrite(output_path, img)
                aligned_paths.append(output_path)
            
            # Add aligned_path column
            df['aligned_path'] = aligned_paths
            
            # Remove rows with missing images
            df = df.dropna(subset=['aligned_path'])
            
            print(f"Processed {len(df)} images to {output_folder}")
        else:
            # Use original filepath as aligned_path
            df['aligned_path'] = df['filepath']
        
        return df
    
    def generate_temporal_pairs(self, metadata_df: pd.DataFrame, 
                               output_path: str,
                               time_gaps: list = None,
                               pairs_per_gap: int = 1000) -> pd.DataFrame:
        """
        Generate verification pairs with time gap information
        
        Args:
            metadata_df: Metadata DataFrame with age and identity info
            output_path: Path to save pairs CSV
            time_gaps: List of time gaps (in years) to generate pairs for
            pairs_per_gap: Number of pairs per time gap
            
        Returns:
            pairs_df: DataFrame with verification pairs
        """
        if time_gaps is None:
            time_gaps = config.TEMPORAL_CONFIG["time_gaps"]
        
        pairs = []
        
        # Group by identity
        identity_groups = metadata_df.groupby('identity')
        
        print(f"Generating temporal pairs with time gaps: {time_gaps}")
        
        for time_gap in time_gaps:
            print(f"\nGenerating pairs for time gap: {time_gap} years")
            
            # Generate positive pairs (same identity)
            positive_pairs = 0
            
            for identity, group in tqdm(identity_groups, desc=f"Positive pairs (gap={time_gap}y)"):
                if len(group) < 2:
                    continue
                
                # Sort by age
                group = group.sort_values('age')
                
                # Find pairs with specified time gap
                for i, row1 in group.iterrows():
                    for j, row2 in group.iterrows():
                        if i >= j:
                            continue
                        
                        age_diff = abs(row2['age'] - row1['age'])
                        
                        # Check if age difference is close to target time gap
                        if abs(age_diff - time_gap) <= 0.5:  # Allow 0.5 year tolerance
                            pairs.append({
                                'enrollment_path': row1['aligned_path'],
                                'probe_path': row2['aligned_path'],
                                'label': 1,
                                'time_gap': age_diff,
                                'identity': identity,
                                'enrollment_age': row1['age'],
                                'probe_age': row2['age']
                            })
                            positive_pairs += 1
                            
                            if positive_pairs >= pairs_per_gap:
                                break
                
                if positive_pairs >= pairs_per_gap:
                    break
            
            print(f"  Generated {positive_pairs} positive pairs")
            
            # Generate negative pairs (different identities) - match the number of positive pairs
            target_negative_pairs = positive_pairs  # Match positive pairs count
            negative_pairs = 0
            identities = list(identity_groups.groups.keys())
            
            np.random.seed(config.SEED + time_gap)  # Different seed per time_gap
            
            # Try with time gap matching first
            attempts = 0
            max_attempts_with_gap = target_negative_pairs * 20  # Reasonable limit
            
            while negative_pairs < target_negative_pairs and attempts < max_attempts_with_gap:
                attempts += 1
                # Random select two different identities
                id1, id2 = np.random.choice(identities, size=2, replace=False)
                
                group1 = identity_groups.get_group(id1)
                group2 = identity_groups.get_group(id2)
                
                # Random select one image from each identity
                row1 = group1.sample(n=1).iloc[0]
                row2 = group2.sample(n=1).iloc[0]
                
                age_diff = abs(row2['age'] - row1['age'])
                
                # Try to match the time gap for negative pairs too
                if abs(age_diff - time_gap) <= 2.0:  # More tolerance for negative pairs
                    pairs.append({
                        'enrollment_path': row1['aligned_path'],
                        'probe_path': row2['aligned_path'],
                        'label': 0,
                        'time_gap': age_diff,
                        'identity': f"{id1}_{id2}",
                        'enrollment_age': row1['age'],
                        'probe_age': row2['age']
                    })
                    negative_pairs += 1
            
            # If we still need more negative pairs, generate without strict age gap constraint
            while negative_pairs < target_negative_pairs:
                # Random select two different identities
                id1, id2 = np.random.choice(identities, size=2, replace=False)
                
                group1 = identity_groups.get_group(id1)
                group2 = identity_groups.get_group(id2)
                
                # Random select one image from each identity
                row1 = group1.sample(n=1).iloc[0]
                row2 = group2.sample(n=1).iloc[0]
                
                age_diff = abs(row2['age'] - row1['age'])
                
                pairs.append({
                    'enrollment_path': row1['aligned_path'],
                    'probe_path': row2['aligned_path'],
                    'label': 0,
                    'time_gap': age_diff,  # Keep actual age difference
                    'identity': f"{id1}_{id2}",
                    'enrollment_age': row1['age'],
                    'probe_age': row2['age']
                })
                negative_pairs += 1
            
            print(f"  Generated {negative_pairs} negative pairs")
        
        # Create DataFrame
        pairs_df = pd.DataFrame(pairs)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pairs_df.to_csv(output_path, index=False)
        
        print(f"\nTotal pairs generated: {len(pairs_df)}")
        print(f"  Positive: {len(pairs_df[pairs_df['label'] == 1])}")
        print(f"  Negative: {len(pairs_df[pairs_df['label'] == 0])}")
        print(f"  Saved to: {output_path}")
        
        return pairs_df
    
    def create_train_val_split(self, metadata_df: pd.DataFrame, 
                              val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train and validation sets by identity
        
        Args:
            metadata_df: Complete metadata DataFrame
            val_ratio: Ratio of validation identities
            
        Returns:
            train_df, val_df: Train and validation DataFrames
        """
        # Get unique identities
        identities = metadata_df['identity'].unique()
        np.random.seed(config.SEED)
        np.random.shuffle(identities)
        
        # Split identities
        val_size = int(len(identities) * val_ratio)
        val_identities = set(identities[:val_size])
        
        # Split data
        val_df = metadata_df[metadata_df['identity'].isin(val_identities)]
        train_df = metadata_df[~metadata_df['identity'].isin(val_identities)]
        
        print(f"Train/Val split:")
        print(f"  Train: {len(train_df)} images, {train_df['identity'].nunique()} identities")
        print(f"  Val: {len(val_df)} images, {val_df['identity'].nunique()} identities")
        
        return train_df, val_df


if __name__ == "__main__":
    # Test MORPH-2 processor
    print("Testing MORPH-2 Processor...")
    
    processor = MORPH2Processor()
    
    # Load train metadata
    train_df = processor.load_metadata('train')
    print(f"\nTrain metadata shape: {train_df.shape}")
    print(f"Columns: {train_df.columns.tolist()}")
    print(f"\nSample data:")
    print(train_df.head())
    
    # Load validation metadata
    val_df = processor.load_metadata('val')
    print(f"\nValidation metadata shape: {val_df.shape}")
    
    print("\nMORPH-2 Processor test complete!")
