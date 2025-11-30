"""
Pair Generation for Time-gap Analysis
"""
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from collections import defaultdict
import json
from tqdm import tqdm
import config


class PairGenerator:
    """Generate enrollment-probe pairs with different time gaps"""
    
    def __init__(self, metadata_df: pd.DataFrame):
        """
        Args:
            metadata_df: DataFrame with columns ['identity', 'age', 'aligned_path', ...]
        """
        self.metadata_df = metadata_df
        self.identity_groups = self._group_by_identity()
        
    def _group_by_identity(self) -> Dict[str, pd.DataFrame]:
        """Group images by identity"""
        groups = {}
        for identity, group in self.metadata_df.groupby('identity'):
            # Sort by age
            group = group.sort_values('age')
            groups[identity] = group
        return groups
    
    def generate_pairs_by_time_gap(self, time_gaps: List[int] = None,
                                   num_pairs_per_gap: int = 1000) -> pd.DataFrame:
        """
        Generate positive pairs (same identity) with specific time gaps
        
        Args:
            time_gaps: List of time gaps in years (e.g., [1, 2, 4, 6])
            num_pairs_per_gap: Number of pairs to generate per time gap
            
        Returns:
            DataFrame with columns: [enrollment_path, probe_path, time_gap, identity, label]
        """
        if time_gaps is None:
            time_gaps = config.TEMPORAL_CONFIG["time_gaps"]
        
        all_pairs = []
        
        for time_gap in time_gaps:
            pairs = self._generate_pairs_for_gap(time_gap, num_pairs_per_gap, same_identity=True)
            all_pairs.extend(pairs)
        
        df = pd.DataFrame(all_pairs, columns=[
            'enrollment_path', 'probe_path', 'enrollment_age', 'probe_age',
            'time_gap', 'identity', 'label'
        ])
        
        return df
    
    def _generate_pairs_for_gap(self, time_gap: int, num_pairs: int,
                                same_identity: bool = True) -> List[Tuple]:
        """
        Generate pairs for a specific time gap
        
        Args:
            time_gap: Time gap in years
            num_pairs: Number of pairs to generate
            same_identity: If True, generate positive pairs; else negative pairs
            
        Returns:
            List of tuples (enrollment_path, probe_path, enroll_age, probe_age, time_gap, identity, label)
        """
        pairs = []
        
        if same_identity:
            # Positive pairs: same identity
            available_identities = [
                identity for identity, group in self.identity_groups.items()
                if len(group) >= 2  # Need at least 2 images
            ]
            
            attempts = 0
            max_attempts = num_pairs * 10
            
            while len(pairs) < num_pairs and attempts < max_attempts:
                attempts += 1
                
                # Random identity
                identity = np.random.choice(available_identities)
                group = self.identity_groups[identity]
                
                if len(group) < 2:
                    continue
                
                # Select two images with desired time gap
                for i in range(len(group) - 1):
                    for j in range(i + 1, len(group)):
                        row1 = group.iloc[i]
                        row2 = group.iloc[j]
                        
                        if row1['age'] is None or row2['age'] is None:
                            continue
                        
                        age_diff = abs(row2['age'] - row1['age'])
                        
                        # Check if age difference matches time gap (within ±0.5 years)
                        if abs(age_diff - time_gap) <= 0.5:
                            pairs.append((
                                row1['aligned_path'],
                                row2['aligned_path'],
                                row1['age'],
                                row2['age'],
                                time_gap,
                                identity,
                                1  # Same identity
                            ))
                            
                            if len(pairs) >= num_pairs:
                                break
                    
                    if len(pairs) >= num_pairs:
                        break
        
        else:
            # Negative pairs: different identities
            identities = list(self.identity_groups.keys())
            
            while len(pairs) < num_pairs:
                # Select two different identities
                id1, id2 = np.random.choice(identities, size=2, replace=False)
                
                group1 = self.identity_groups[id1]
                group2 = self.identity_groups[id2]
                
                # Random image from each identity
                row1 = group1.sample(1).iloc[0]
                row2 = group2.sample(1).iloc[0]
                
                if row1['age'] is None or row2['age'] is None:
                    continue
                
                pairs.append((
                    row1['aligned_path'],
                    row2['aligned_path'],
                    row1['age'],
                    row2['age'],
                    time_gap,  # Doesn't matter for negative pairs
                    f"{id1}_{id2}",
                    0  # Different identity
                ))
        
        return pairs
    
    def generate_balanced_pairs(self, time_gaps: List[int] = None,
                               num_pairs_per_gap: int = 1000) -> pd.DataFrame:
        """
        Generate balanced positive and negative pairs
        
        Args:
            time_gaps: List of time gaps in years
            num_pairs_per_gap: Number of pairs per time gap (for each class)
            
        Returns:
            Balanced DataFrame with positive and negative pairs
        """
        if time_gaps is None:
            time_gaps = config.TEMPORAL_CONFIG["time_gaps"]
        
        positive_pairs = self.generate_pairs_by_time_gap(time_gaps, num_pairs_per_gap)
        
        # Generate negative pairs
        negative_pairs = []
        for time_gap in time_gaps:
            pairs = self._generate_pairs_for_gap(time_gap, num_pairs_per_gap, same_identity=False)
            negative_pairs.extend(pairs)
        
        negative_df = pd.DataFrame(negative_pairs, columns=[
            'enrollment_path', 'probe_path', 'enrollment_age', 'probe_age',
            'time_gap', 'identity', 'label'
        ])
        
        # Combine positive and negative
        combined_df = pd.concat([positive_pairs, negative_df], ignore_index=True)
        
        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
        
        print(f"Generated {len(combined_df)} pairs:")
        print(f"  Positive: {len(positive_pairs)}")
        print(f"  Negative: {len(negative_df)}")
        
        return combined_df
    
    def generate_protocol_pairs(self) -> Dict[str, pd.DataFrame]:
        """
        Generate pairs following standard verification protocols
        
        Returns:
            Dictionary with different protocols
        """
        protocols = {}
        
        # Protocol 1: Standard verification (all pairs regardless of age gap)
        all_pairs = self._generate_all_possible_pairs()
        protocols['standard'] = all_pairs
        
        # Protocol 2: Time-gap specific (separate by age difference)
        for gap in config.TEMPORAL_CONFIG["time_gaps"]:
            gap_pairs = self.generate_balanced_pairs(
                time_gaps=[gap],
                num_pairs_per_gap=config.PAIR_CONFIG["same_identity_pairs"]
            )
            protocols[f'gap_{gap}y'] = gap_pairs
        
        return protocols
    
    def _generate_all_possible_pairs(self) -> pd.DataFrame:
        """Generate all possible pairs (exhaustive)"""
        all_pairs = []
        
        # Positive pairs
        for identity, group in tqdm(self.identity_groups.items(), desc="Generating all pairs"):
            if len(group) < 2:
                continue
            
            # All combinations within identity
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    row1 = group.iloc[i]
                    row2 = group.iloc[j]
                    
                    if row1['age'] is None or row2['age'] is None:
                        continue
                    
                    age_gap = abs(row2['age'] - row1['age'])
                    
                    all_pairs.append({
                        'enrollment_path': row1['aligned_path'],
                        'probe_path': row2['aligned_path'],
                        'enrollment_age': row1['age'],
                        'probe_age': row2['age'],
                        'time_gap': age_gap,
                        'identity': identity,
                        'label': 1
                    })
        
        # Sample negative pairs (same number as positive)
        num_positive = len(all_pairs)
        identities = list(self.identity_groups.keys())
        
        for _ in tqdm(range(num_positive), desc="Generating negative pairs"):
            id1, id2 = np.random.choice(identities, size=2, replace=False)
            
            row1 = self.identity_groups[id1].sample(1).iloc[0]
            row2 = self.identity_groups[id2].sample(1).iloc[0]
            
            if row1['age'] is None or row2['age'] is None:
                continue
            
            all_pairs.append({
                'enrollment_path': row1['aligned_path'],
                'probe_path': row2['aligned_path'],
                'enrollment_age': row1['age'],
                'probe_age': row2['age'],
                'time_gap': 0,  # Not applicable
                'identity': f"{id1}_{id2}",
                'label': 0
            })
        
        df = pd.DataFrame(all_pairs)
        return df
    
    def save_pairs(self, pairs_df: pd.DataFrame, output_path: str):
        """Save pairs to CSV"""
        pairs_df.to_csv(output_path, index=False)
        print(f"Saved {len(pairs_df)} pairs to {output_path}")
    
    def get_statistics(self, pairs_df: pd.DataFrame) -> Dict:
        """Get statistics about generated pairs"""
        stats = {
            'total_pairs': len(pairs_df),
            'positive_pairs': len(pairs_df[pairs_df['label'] == 1]),
            'negative_pairs': len(pairs_df[pairs_df['label'] == 0]),
            'time_gap_distribution': pairs_df[pairs_df['label'] == 1].groupby('time_gap').size().to_dict(),
            'unique_identities': pairs_df['identity'].nunique(),
        }
        
        return stats


def generate_all_protocols():
    """Generate all pair protocols for the dataset"""
    # Load metadata
    metadata_path = os.path.join(config.OUTPUT_ROOT, "processed_images", "metadata.csv")
    
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        print("Please run data_processor.py first")
        return
    
    metadata_df = pd.read_csv(metadata_path)
    
    # Create pair generator
    generator = PairGenerator(metadata_df)
    
    # Generate protocols
    print("Generating pair protocols...")
    protocols = generator.generate_protocol_pairs()
    
    # Save each protocol
    output_dir = os.path.join(config.OUTPUT_ROOT, "pairs")
    os.makedirs(output_dir, exist_ok=True)
    
    for protocol_name, pairs_df in protocols.items():
        output_path = os.path.join(output_dir, f"{protocol_name}_pairs.csv")
        generator.save_pairs(pairs_df, output_path)
        
        # Print statistics
        stats = generator.get_statistics(pairs_df)
        print(f"\n{protocol_name} statistics:")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    generate_all_protocols()
