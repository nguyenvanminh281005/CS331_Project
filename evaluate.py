"""
Evaluation script for face recognition models
"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import pickle
from typing import Dict
import cv2

import config
from models import create_arcface_model, create_magface_model, create_temporal_model
from utils.metrics import (
    evaluate_verification,
    evaluate_by_time_gap,
    calculate_degradation_rate
)


class Evaluator:
    """Evaluator for face recognition models"""
    
    def __init__(self, model_type='arcface', checkpoint_path=None, backbone='magface'):
        """
        Args:
            model_type: 'arcface', 'magface', or 'temporal'
            checkpoint_path: Path to model checkpoint
            backbone: 'arcface' or 'magface' - backbone for temporal model
        """
        self.model_type = model_type
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        
        # Load model
        if model_type == 'arcface':
            self.model = create_arcface_model(pretrained=True, checkpoint_path=checkpoint_path)
        elif model_type == 'magface':
            self.model = create_magface_model(pretrained=True, checkpoint_path=checkpoint_path)
        elif model_type == 'temporal':
            # Try to infer backbone from checkpoint
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    if 'config' in checkpoint and 'backbone' in checkpoint['config']:
                        backbone = checkpoint['config']['backbone'] or backbone
                        print(f"Loaded backbone from checkpoint: {backbone}")
                except Exception as e:
                    print(f"Could not load backbone from checkpoint: {e}")
                    print(f"Using specified backbone: {backbone}")
            
            self.model = create_temporal_model(
                pretrained=True, 
                checkpoint_path=checkpoint_path,
                backbone=backbone
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Initialized {model_type} evaluator")
        if model_type == 'temporal':
            print(f"Backbone: {backbone}")
        print(f"Device: {self.device}")
    
    def extract_features(self, image_paths, batch_size=128, save_path=None):
        """
        Extract features for all images
        
        Args:
            image_paths: List of image paths or DataFrame with 'aligned_path' column
            batch_size: Batch size for extraction
            save_path: Path to save extracted features
            
        Returns:
            features_dict: Dictionary mapping image paths to features
        """
        import torchvision.transforms as transforms
        
        # Handle DataFrame input
        if isinstance(image_paths, pd.DataFrame):
            image_paths = image_paths['aligned_path'].tolist()
        
        # Transform
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        features_dict = {}
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_paths = []
            
            # Load images
            for img_path in batch_paths:
                if not os.path.exists(img_path):
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = transform(img)
                batch_images.append(img)
                valid_paths.append(img_path)
            
            if len(batch_images) == 0:
                continue
            
            # Stack and move to device
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Extract features
            with torch.no_grad():
                if self.model_type == 'magface':
                    features, quality_scores = self.model.extract_features(batch_tensor)
                else:
                    features = self.model.extract_features(batch_tensor)
            
            # Store features
            for path, feat in zip(valid_paths, features):
                features_dict[path] = feat.cpu()
        
        print(f"Extracted features for {len(features_dict)} images")
        
        # Save features
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(features_dict, f)
            print(f"Saved features to {save_path}")
        
        return features_dict
    
    def evaluate_pairs(self, pairs_df, features_dict=None, extract_features=True):
        """
        Evaluate on verification pairs
        
        Args:
            pairs_df: DataFrame with columns [enrollment_path, probe_path, label, ...]
            features_dict: Pre-computed features (optional)
            extract_features: Whether to extract features if not provided
            
        Returns:
            metrics: Evaluation metrics
        """
        # Extract features if needed
        if features_dict is None and extract_features:
            all_paths = pd.concat([
                pairs_df['enrollment_path'],
                pairs_df['probe_path']
            ]).unique()
            
            features_dict = self.extract_features(all_paths)
        
        # Get embeddings for pairs
        embeddings1 = []
        embeddings2 = []
        labels = []
        
        for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Loading pairs"):
            if row['enrollment_path'] in features_dict and \
               row['probe_path'] in features_dict:
                embeddings1.append(features_dict[row['enrollment_path']])
                embeddings2.append(features_dict[row['probe_path']])
                labels.append(row['label'])
        
        if len(embeddings1) == 0:
            print("No valid pairs found!")
            return None
        
        # Stack embeddings
        embeddings1 = torch.stack(embeddings1)
        embeddings2 = torch.stack(embeddings2)
        labels = torch.tensor(labels)
        
        # Evaluate
        metrics = evaluate_verification(
            embeddings1, embeddings2, labels,
            distance_metric=config.EVAL_CONFIG["distance_metric"]
        )
        
        return metrics
    
    def evaluate_by_time_gap(self, pairs_df, features_dict=None, extract_features=True):
        """
        Evaluate performance across different time gaps
        
        Args:
            pairs_df: DataFrame with time gap information
            features_dict: Pre-computed features
            extract_features: Whether to extract features
            
        Returns:
            results: Dictionary with results for each time gap
        """
        # Extract features if needed
        if features_dict is None and extract_features:
            all_paths = pd.concat([
                pairs_df['enrollment_path'],
                pairs_df['probe_path']
            ]).unique()
            
            features_dict = self.extract_features(all_paths)
        
        # Evaluate by time gap
        results = evaluate_by_time_gap(
            pairs_df,
            features_dict,
            time_gaps=config.TEMPORAL_CONFIG["time_gaps"]
        )
        
        return results
    
    def evaluate_all_protocols(self, pairs_dir=None, evaluate_dataset=None):
        """
        Evaluate on all pair protocols
        
        Args:
            pairs_dir: Directory containing pair CSV files
            evaluate_dataset: Filter to only evaluate this dataset (e.g., 'morph_2', 'agedb_30')
            
        Returns:
            all_results: Results for all protocols
        """
        if pairs_dir is None:
            pairs_dir = os.path.join(config.OUTPUT_ROOT, "pairs")
        
        # Find all pair files
        import glob
        pair_files = glob.glob(os.path.join(pairs_dir, "*_pairs.csv"))
        
        # Filter by dataset if specified
        if evaluate_dataset:
            dataset_filter = evaluate_dataset.lower().replace('_', '')
            pair_files = [f for f in pair_files if dataset_filter in os.path.basename(f).lower().replace('_', '')]
        
        if len(pair_files) == 0:
            print(f"No pair files found in {pairs_dir}" + (f" for dataset {evaluate_dataset}" if evaluate_dataset else ""))
            return None
        
        all_results = {}
        
        for pair_file in pair_files:
            protocol_name = os.path.basename(pair_file).replace('_pairs.csv', '')
            print(f"\n{'='*60}")
            print(f"Evaluating protocol: {protocol_name}")
            print(f"{'='*60}")
            
            # Load pairs
            pairs_df = pd.read_csv(pair_file)
            
            # Rename columns if needed (for compatibility)
            if 'img1_path' in pairs_df.columns and 'enrollment_path' not in pairs_df.columns:
                pairs_df = pairs_df.rename(columns={
                    'img1_path': 'enrollment_path',
                    'img2_path': 'probe_path'
                })
            
            # Evaluate
            if 'time_gap' in pairs_df.columns:
                results = self.evaluate_by_time_gap(pairs_df)
            else:
                results = self.evaluate_pairs(pairs_df)
            
            all_results[protocol_name] = results
            
            # Print results
            if isinstance(results, dict):
                if 'gap_' in list(results.keys())[0]:  # Time-gap results
                    for gap_key, metrics in results.items():
                        print(f"\n{gap_key}:")
                        print(f"  Accuracy@EER: {metrics.get('accuracy_at_eer', 0):.4f}")
                        print(f"  EER: {metrics['eer']:.4f}")
                        print(f"  TAR@FAR=0.1%: {metrics.get('tar@far=0.001', 0):.4f}")
                        print(f"  AUC: {metrics['auc']:.4f}")
                        if 'degradation_rate' in metrics:
                            print(f"  Degradation rate: {metrics['degradation_rate']:.6f}/year")
                else:  # Single result
                    print(f"  Accuracy@EER: {results.get('accuracy_at_eer', 0):.4f}")
                    print(f"  EER: {results['eer']:.4f}")
                    print(f"  AUC: {results['auc']:.4f}")
        
        # Save results
        results_path = os.path.join(
            config.OUTPUT_ROOT,
            "results",
            f"{self.model_type}_evaluation_results.json"
        )
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        # Convert to JSON-serializable format
        json_results = {}
        for protocol, results in all_results.items():
            json_results[protocol] = {}
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict):
                        json_results[protocol][key] = {
                            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in value.items()
                        }
                    else:
                        json_results[protocol][key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nSaved results to {results_path}")
        
        return all_results


def compare_models(checkpoint_paths: Dict[str, str], pairs_df: pd.DataFrame):
    """
    Compare multiple models on the same dataset
    
    Args:
        checkpoint_paths: Dictionary mapping model names to checkpoint paths
        pairs_df: Pairs DataFrame for evaluation
    """
    results_comparison = {}
    
    for model_name, checkpoint_path in checkpoint_paths.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Determine model type from name
        if 'arcface' in model_name.lower():
            model_type = 'arcface'
        elif 'magface' in model_name.lower():
            model_type = 'magface'
        elif 'temporal' in model_name.lower():
            model_type = 'temporal'
        else:
            model_type = 'arcface'
        
        # Evaluate
        evaluator = Evaluator(model_type=model_type, checkpoint_path=checkpoint_path)
        
        if 'time_gap' in pairs_df.columns:
            results = evaluator.evaluate_by_time_gap(pairs_df)
        else:
            results = evaluator.evaluate_pairs(pairs_df)
        
        results_comparison[model_name] = results
    
    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for model_name, results in results_comparison.items():
        print(f"\n{model_name}:")
        if isinstance(results, dict) and 'gap_' in list(results.keys())[0]:
            for gap_key, metrics in results.items():
                print(f"  {gap_key}: TAR@FAR=0.1% = {metrics.get('tar@far=0.001', 0):.4f}")
        else:
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  EER: {results['eer']:.4f}")
    
    return results_comparison


if __name__ == "__main__":
    # Example usage
    print("Evaluation script ready")
    
    # Evaluate ArcFace baseline
    # evaluator = Evaluator(model_type='arcface', checkpoint_path='path/to/checkpoint.pth')
    # results = evaluator.evaluate_all_protocols()
    
    print("\nTo use:")
    print("1. Train models using train.py")
    print("2. Generate pairs using utils/pair_generator.py")
    print("3. Run evaluation:")
    print("   evaluator = Evaluator(model_type='arcface', checkpoint_path='model.pth')")
    print("   results = evaluator.evaluate_all_protocols()")
