"""
Test Image Quality Robustness
Evaluate model robustness against various image degradation factors:
- Resolution (downsampling & upsampling)
- Blur (Gaussian Blur)
- Noise (Salt & Pepper, Gaussian Noise)
"""
import os
import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pickle

import config
from models import create_arcface_model, create_magface_model
from utils.metrics import evaluate_verification


class ImageDegradation:
    """Apply various degradation to images for robustness testing"""
    
    @staticmethod
    def downscale_upscale(image, target_size):
        """
        Downscale image to target size then upscale back to original
        Args:
            image: Input image (H, W, C)
            target_size: Target size (height, width) for downscaling
        Returns:
            Degraded image with original size
        """
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        downscaled = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        upscaled = cv2.resize(downscaled, original_size, interpolation=cv2.INTER_LINEAR)
        return upscaled
    
    @staticmethod
    def apply_gaussian_blur(image, kernel_size):
        """
        Apply Gaussian blur
        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel (must be odd)
        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def add_gaussian_noise(image, mean=0, std=25):
        """
        Add Gaussian noise to image
        Args:
            image: Input image (0-255)
            mean: Mean of Gaussian noise
            std: Standard deviation of Gaussian noise
        Returns:
            Noisy image
        """
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
    @staticmethod
    def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
        """
        Add salt and pepper noise
        Args:
            image: Input image
            salt_prob: Probability of salt noise
            pepper_prob: Probability of pepper noise
        Returns:
            Noisy image
        """
        noisy_image = image.copy()
        
        # Salt noise (white pixels)
        salt_mask = np.random.random(image.shape[:2]) < salt_prob
        noisy_image[salt_mask] = 255
        
        # Pepper noise (black pixels)
        pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
        noisy_image[pepper_mask] = 0
        
        return noisy_image


class RobustnessEvaluator:
    """Evaluate model robustness against image degradations"""
    
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
        
        self.degradation = ImageDegradation()
        
        print(f"Initialized {model_type} robustness evaluator")
        if model_type == 'temporal':
            print(f"Backbone: {backbone}")
        print(f"Device: {self.device}")
    
    def extract_features_from_image(self, image):
        """
        Extract features from a single image
        Args:
            image: Image array (H, W, C) in BGR format
        Returns:
            Feature vector
        """
        import torchvision.transforms as transforms
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        image_tensor = transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            if self.model_type == 'magface':
                features, _ = self.model.extract_features(image_tensor)
            else:
                features = self.model.extract_features(image_tensor)
        
        return features[0].cpu()
    
    def test_resolution_robustness(self, pairs_df, resolution_levels=None):
        """
        Test robustness against different resolutions
        Args:
            pairs_df: DataFrame with pairs
            resolution_levels: List of (height, width) tuples
        Returns:
            results: Dictionary with results for each resolution
        """
        if resolution_levels is None:
            # Default resolution levels: 112x112, 56x56, 28x28, 14x14
            resolution_levels = [
                (112, 112),  # Original
                (84, 84),
                (56, 56),
                (42, 42),
                (28, 28),
                (21, 21),
                (14, 14),
            ]
        
        results = {}
        
        for res_h, res_w in tqdm(resolution_levels, desc="Testing resolutions"):
            print(f"\nTesting resolution: {res_h}x{res_w}")
            
            embeddings1 = []
            embeddings2 = []
            labels = []
            
            for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc=f"Processing {res_h}x{res_w}"):
                # Load images
                img1 = cv2.imread(row['enrollment_path'])
                img2 = cv2.imread(row['probe_path'])
                
                if img1 is None or img2 is None:
                    continue
                
                # Apply degradation (except for original resolution)
                if (res_h, res_w) != (112, 112):
                    img1 = self.degradation.downscale_upscale(img1, (res_h, res_w))
                    img2 = self.degradation.downscale_upscale(img2, (res_h, res_w))
                
                # Extract features
                feat1 = self.extract_features_from_image(img1)
                feat2 = self.extract_features_from_image(img2)
                
                embeddings1.append(feat1)
                embeddings2.append(feat2)
                labels.append(row['label'])
            
            if len(embeddings1) == 0:
                continue
            
            # Stack embeddings
            embeddings1 = torch.stack(embeddings1)
            embeddings2 = torch.stack(embeddings2)
            labels = torch.tensor(labels)
            
            # Evaluate
            metrics = evaluate_verification(
                embeddings1, embeddings2, labels,
                distance_metric=config.EVAL_CONFIG["distance_metric"]
            )
            
            resolution_key = f"{res_h}x{res_w}"
            results[resolution_key] = metrics
            
            print(f"Resolution {resolution_key}:")
            print(f"  Accuracy@EER: {metrics.get('accuracy_at_eer', 0):.4f}")
            print(f"  EER: {metrics['eer']:.4f}")
            print(f"  TAR@FAR=0.1%: {metrics.get('tar@far=0.001', 0):.4f}")
        
        return results
    
    def test_blur_robustness(self, pairs_df, blur_levels=None):
        """
        Test robustness against Gaussian blur
        Args:
            pairs_df: DataFrame with pairs
            blur_levels: List of kernel sizes
        Returns:
            results: Dictionary with results for each blur level
        """
        if blur_levels is None:
            # Default blur levels
            blur_levels = [1, 3, 5, 7, 9, 11, 15, 21]
        
        results = {}
        
        for kernel_size in tqdm(blur_levels, desc="Testing blur levels"):
            print(f"\nTesting blur kernel: {kernel_size}x{kernel_size}")
            
            embeddings1 = []
            embeddings2 = []
            labels = []
            
            for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc=f"Processing kernel={kernel_size}"):
                # Load images
                img1 = cv2.imread(row['enrollment_path'])
                img2 = cv2.imread(row['probe_path'])
                
                if img1 is None or img2 is None:
                    continue
                
                # Apply degradation (except for kernel=1, which is no blur)
                if kernel_size > 1:
                    img1 = self.degradation.apply_gaussian_blur(img1, kernel_size)
                    img2 = self.degradation.apply_gaussian_blur(img2, kernel_size)
                
                # Extract features
                feat1 = self.extract_features_from_image(img1)
                feat2 = self.extract_features_from_image(img2)
                
                embeddings1.append(feat1)
                embeddings2.append(feat2)
                labels.append(row['label'])
            
            if len(embeddings1) == 0:
                continue
            
            # Stack embeddings
            embeddings1 = torch.stack(embeddings1)
            embeddings2 = torch.stack(embeddings2)
            labels = torch.tensor(labels)
            
            # Evaluate
            metrics = evaluate_verification(
                embeddings1, embeddings2, labels,
                distance_metric=config.EVAL_CONFIG["distance_metric"]
            )
            
            blur_key = f"kernel_{kernel_size}"
            results[blur_key] = metrics
            
            print(f"Blur kernel {kernel_size}:")
            print(f"  Accuracy@EER: {metrics.get('accuracy_at_eer', 0):.4f}")
            print(f"  EER: {metrics['eer']:.4f}")
            print(f"  TAR@FAR=0.1%: {metrics.get('tar@far=0.001', 0):.4f}")
        
        return results
    
    def test_noise_robustness(self, pairs_df, noise_levels=None, noise_type='gaussian'):
        """
        Test robustness against noise
        Args:
            pairs_df: DataFrame with pairs
            noise_levels: List of noise levels
            noise_type: 'gaussian' or 'salt_pepper'
        Returns:
            results: Dictionary with results for each noise level
        """
        if noise_levels is None:
            if noise_type == 'gaussian':
                # Gaussian noise: standard deviations
                noise_levels = [0, 10, 20, 30, 40, 50, 60, 80, 100]
            else:
                # Salt & Pepper: probabilities
                noise_levels = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
        
        results = {}
        
        for noise_level in tqdm(noise_levels, desc=f"Testing {noise_type} noise"):
            print(f"\nTesting {noise_type} noise: {noise_level}")
            
            embeddings1 = []
            embeddings2 = []
            labels = []
            
            for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc=f"Processing noise={noise_level}"):
                # Load images
                img1 = cv2.imread(row['enrollment_path'])
                img2 = cv2.imread(row['probe_path'])
                
                if img1 is None or img2 is None:
                    continue
                
                # Apply degradation
                if noise_level > 0:
                    if noise_type == 'gaussian':
                        img1 = self.degradation.add_gaussian_noise(img1, std=noise_level)
                        img2 = self.degradation.add_gaussian_noise(img2, std=noise_level)
                    elif noise_type == 'salt_pepper':
                        img1 = self.degradation.add_salt_pepper_noise(img1, salt_prob=noise_level, pepper_prob=noise_level)
                        img2 = self.degradation.add_salt_pepper_noise(img2, salt_prob=noise_level, pepper_prob=noise_level)
                
                # Extract features
                feat1 = self.extract_features_from_image(img1)
                feat2 = self.extract_features_from_image(img2)
                
                embeddings1.append(feat1)
                embeddings2.append(feat2)
                labels.append(row['label'])
            
            if len(embeddings1) == 0:
                continue
            
            # Stack embeddings
            embeddings1 = torch.stack(embeddings1)
            embeddings2 = torch.stack(embeddings2)
            labels = torch.tensor(labels)
            
            # Evaluate
            metrics = evaluate_verification(
                embeddings1, embeddings2, labels,
                distance_metric=config.EVAL_CONFIG["distance_metric"]
            )
            
            noise_key = f"noise_{noise_level}"
            results[noise_key] = metrics
            
            print(f"{noise_type} noise {noise_level}:")
            print(f"  Accuracy@EER: {metrics.get('accuracy_at_eer', 0):.4f}")
            print(f"  EER: {metrics['eer']:.4f}")
            print(f"  TAR@FAR=0.1%: {metrics.get('tar@far=0.001', 0):.4f}")
        
        return results
    
    def run_all_robustness_tests(self, pairs_df, output_dir=None):
        """
        Run all robustness tests
        Args:
            pairs_df: DataFrame with pairs
            output_dir: Directory to save results
        Returns:
            all_results: Dictionary with all test results
        """
        all_results = {}
        
        print("\n" + "="*80)
        print("TESTING RESOLUTION ROBUSTNESS")
        print("="*80)
        all_results['resolution'] = self.test_resolution_robustness(pairs_df)
        
        print("\n" + "="*80)
        print("TESTING BLUR ROBUSTNESS")
        print("="*80)
        all_results['blur'] = self.test_blur_robustness(pairs_df)
        
        print("\n" + "="*80)
        print("TESTING GAUSSIAN NOISE ROBUSTNESS")
        print("="*80)
        all_results['gaussian_noise'] = self.test_noise_robustness(pairs_df, noise_type='gaussian')
        
        print("\n" + "="*80)
        print("TESTING SALT & PEPPER NOISE ROBUSTNESS")
        print("="*80)
        all_results['salt_pepper_noise'] = self.test_noise_robustness(pairs_df, noise_type='salt_pepper')
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save as pickle
            results_path = os.path.join(output_dir, f"{self.model_type}_robustness_results.pkl")
            with open(results_path, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"\nSaved results to: {results_path}")
            
            # Save as JSON for readability
            import json
            json_results = {}
            for test_type, test_results in all_results.items():
                json_results[test_type] = {}
                for key, metrics in test_results.items():
                    json_results[test_type][key] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in metrics.items()
                    }
            
            json_path = os.path.join(output_dir, f"{self.model_type}_robustness_results.json")
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"Saved JSON results to: {json_path}")
        
        return all_results


def visualize_robustness_results(results_dict, output_dir=None):
    """
    Visualize robustness test results
    Args:
        results_dict: Dictionary mapping model names to their robustness results
        output_dir: Directory to save plots
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # Metrics to plot
    metrics_to_plot = ['accuracy_at_eer', 'eer', 'tar@far=0.001']
    metric_labels = {
        'accuracy_at_eer': 'Accuracy@EER',
        'eer': 'EER',
        'tar@far=0.001': 'TAR @ FAR=0.1%'
    }
    
    # 1. Resolution Robustness
    if any('resolution' in results for results in results_dict.values()):
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 6))
            
            for model_name, all_results in results_dict.items():
                if 'resolution' not in all_results:
                    continue
                
                resolution_results = all_results['resolution']
                
                # Extract resolution levels and metric values
                resolutions = []
                metric_values = []
                
                for res_key in sorted(resolution_results.keys(), key=lambda x: int(x.split('x')[0]), reverse=True):
                    res_value = int(res_key.split('x')[0])
                    resolutions.append(res_value)
                    
                    if metric in resolution_results[res_key]:
                        metric_values.append(resolution_results[res_key][metric])
                    else:
                        metric_values.append(0)
                
                plt.plot(resolutions, metric_values, marker='o', linewidth=2, markersize=8, label=model_name)
            
            plt.xlabel('Resolution (pixels)', fontsize=14, fontweight='bold')
            plt.ylabel(metric_labels.get(metric, metric), fontsize=14, fontweight='bold')
            plt.title(f'Resolution Robustness - {metric_labels.get(metric, metric)}', fontsize=16, fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'resolution_robustness_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 2. Blur Robustness
    if any('blur' in results for results in results_dict.values()):
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 6))
            
            for model_name, all_results in results_dict.items():
                if 'blur' not in all_results:
                    continue
                
                blur_results = all_results['blur']
                
                # Extract blur levels and metric values
                blur_levels = []
                metric_values = []
                
                for blur_key in sorted(blur_results.keys(), key=lambda x: int(x.split('_')[1])):
                    blur_value = int(blur_key.split('_')[1])
                    blur_levels.append(blur_value)
                    
                    if metric in blur_results[blur_key]:
                        metric_values.append(blur_results[blur_key][metric])
                    else:
                        metric_values.append(0)
                
                plt.plot(blur_levels, metric_values, marker='s', linewidth=2, markersize=8, label=model_name)
            
            plt.xlabel('Blur Kernel Size', fontsize=14, fontweight='bold')
            plt.ylabel(metric_labels.get(metric, metric), fontsize=14, fontweight='bold')
            plt.title(f'Blur Robustness - {metric_labels.get(metric, metric)}', fontsize=16, fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'blur_robustness_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. Gaussian Noise Robustness
    if any('gaussian_noise' in results for results in results_dict.values()):
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 6))
            
            for model_name, all_results in results_dict.items():
                if 'gaussian_noise' not in all_results:
                    continue
                
                noise_results = all_results['gaussian_noise']
                
                # Extract noise levels and metric values
                noise_levels = []
                metric_values = []
                
                for noise_key in sorted(noise_results.keys(), key=lambda x: float(x.split('_')[1])):
                    noise_value = float(noise_key.split('_')[1])
                    noise_levels.append(noise_value)
                    
                    if metric in noise_results[noise_key]:
                        metric_values.append(noise_results[noise_key][metric])
                    else:
                        metric_values.append(0)
                
                plt.plot(noise_levels, metric_values, marker='^', linewidth=2, markersize=8, label=model_name)
            
            plt.xlabel('Gaussian Noise (σ)', fontsize=14, fontweight='bold')
            plt.ylabel(metric_labels.get(metric, metric), fontsize=14, fontweight='bold')
            plt.title(f'Gaussian Noise Robustness - {metric_labels.get(metric, metric)}', fontsize=16, fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'gaussian_noise_robustness_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Salt & Pepper Noise Robustness
    if any('salt_pepper_noise' in results for results in results_dict.values()):
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 6))
            
            for model_name, all_results in results_dict.items():
                if 'salt_pepper_noise' not in all_results:
                    continue
                
                noise_results = all_results['salt_pepper_noise']
                
                # Extract noise levels and metric values
                noise_levels = []
                metric_values = []
                
                for noise_key in sorted(noise_results.keys(), key=lambda x: float(x.split('_')[1])):
                    noise_value = float(noise_key.split('_')[1])
                    noise_levels.append(noise_value)
                    
                    if metric in noise_results[noise_key]:
                        metric_values.append(noise_results[noise_key][metric])
                    else:
                        metric_values.append(0)
                
                plt.plot(noise_levels, metric_values, marker='d', linewidth=2, markersize=8, label=model_name)
            
            plt.xlabel('Salt & Pepper Noise Probability', fontsize=14, fontweight='bold')
            plt.ylabel(metric_labels.get(metric, metric), fontsize=14, fontweight='bold')
            plt.title(f'Salt & Pepper Noise Robustness - {metric_labels.get(metric, metric)}', fontsize=16, fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'salt_pepper_noise_robustness_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 5. Combined visualization (all degradation types on one plot)
    for metric in metrics_to_plot:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Overall Robustness Comparison - {metric_labels.get(metric, metric)}', 
                     fontsize=18, fontweight='bold')
        
        # Resolution
        ax = axes[0, 0]
        for model_name, all_results in results_dict.items():
            if 'resolution' in all_results:
                resolution_results = all_results['resolution']
                resolutions = []
                metric_values = []
                for res_key in sorted(resolution_results.keys(), key=lambda x: int(x.split('x')[0]), reverse=True):
                    resolutions.append(int(res_key.split('x')[0]))
                    metric_values.append(resolution_results[res_key].get(metric, 0))
                ax.plot(resolutions, metric_values, marker='o', linewidth=2, markersize=6, label=model_name)
        ax.set_xlabel('Resolution (pixels)', fontweight='bold')
        ax.set_ylabel(metric_labels.get(metric, metric), fontweight='bold')
        ax.set_title('Resolution Degradation', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Blur
        ax = axes[0, 1]
        for model_name, all_results in results_dict.items():
            if 'blur' in all_results:
                blur_results = all_results['blur']
                blur_levels = []
                metric_values = []
                for blur_key in sorted(blur_results.keys(), key=lambda x: int(x.split('_')[1])):
                    blur_levels.append(int(blur_key.split('_')[1]))
                    metric_values.append(blur_results[blur_key].get(metric, 0))
                ax.plot(blur_levels, metric_values, marker='s', linewidth=2, markersize=6, label=model_name)
        ax.set_xlabel('Blur Kernel Size', fontweight='bold')
        ax.set_ylabel(metric_labels.get(metric, metric), fontweight='bold')
        ax.set_title('Blur Degradation', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Gaussian Noise
        ax = axes[1, 0]
        for model_name, all_results in results_dict.items():
            if 'gaussian_noise' in all_results:
                noise_results = all_results['gaussian_noise']
                noise_levels = []
                metric_values = []
                for noise_key in sorted(noise_results.keys(), key=lambda x: float(x.split('_')[1])):
                    noise_levels.append(float(noise_key.split('_')[1]))
                    metric_values.append(noise_results[noise_key].get(metric, 0))
                ax.plot(noise_levels, metric_values, marker='^', linewidth=2, markersize=6, label=model_name)
        ax.set_xlabel('Gaussian Noise (σ)', fontweight='bold')
        ax.set_ylabel(metric_labels.get(metric, metric), fontweight='bold')
        ax.set_title('Gaussian Noise Degradation', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Salt & Pepper Noise
        ax = axes[1, 1]
        for model_name, all_results in results_dict.items():
            if 'salt_pepper_noise' in all_results:
                noise_results = all_results['salt_pepper_noise']
                noise_levels = []
                metric_values = []
                for noise_key in sorted(noise_results.keys(), key=lambda x: float(x.split('_')[1])):
                    noise_levels.append(float(noise_key.split('_')[1]))
                    metric_values.append(noise_results[noise_key].get(metric, 0))
                ax.plot(noise_levels, metric_values, marker='d', linewidth=2, markersize=6, label=model_name)
        ax.set_xlabel('S&P Noise Probability', fontweight='bold')
        ax.set_ylabel(metric_labels.get(metric, metric), fontweight='bold')
        ax.set_title('Salt & Pepper Noise Degradation', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'combined_robustness_{metric}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nVisualization complete! Plots saved to: {output_dir}")


if __name__ == "__main__":
    print("Image Quality Robustness Testing")
    print("Usage:")
    print("  python test_image_quality_robustness.py --model <model_type> --checkpoint <path>")
    print("\nOr integrate into main_pipeline.py with --test-robustness flag")
