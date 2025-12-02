"""
Evaluation Metrics for Face Recognition
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import config


def calculate_accuracy(distances: np.ndarray, labels: np.ndarray, 
                      threshold: float) -> float:
    """
    Calculate accuracy at given threshold
    
    Args:
        distances: Pairwise distances
        labels: Ground truth labels (1 for same, 0 for different)
        threshold: Decision threshold
        
    Returns:
        accuracy: Classification accuracy
    """
    predictions = (distances < threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    return accuracy


def calculate_roc(distances: np.ndarray, labels: np.ndarray) -> Tuple:
    """
    Calculate ROC curve
    
    Args:
        distances: Pairwise distances
        labels: Ground truth labels
        
    Returns:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Decision thresholds
    """
    # Convert distances to similarities (lower distance = higher similarity)
    similarities = 1 - distances
    
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    
    return fpr, tpr, thresholds


def calculate_tar_at_far(distances: np.ndarray, labels: np.ndarray,
                         far_target: float) -> Tuple[float, float]:
    """
    Calculate True Accept Rate (TAR) at specific False Accept Rate (FAR)
    
    Args:
        distances: Pairwise distances
        labels: Ground truth labels
        far_target: Target FAR value (e.g., 0.001 for 0.1%)
        
    Returns:
        tar: True Accept Rate at target FAR
        threshold: Threshold at target FAR
    """
    fpr, tpr, thresholds = calculate_roc(distances, labels)
    
    # Use interpolation for more accurate TAR at exact FAR
    if far_target <= np.min(fpr):
        # If target FAR is smaller than minimum, use first point
        tar = tpr[0]
        threshold = thresholds[0]
    elif far_target >= np.max(fpr):
        # If target FAR is larger than maximum, use last point
        tar = tpr[-1]
        threshold = thresholds[-1]
    else:
        # Interpolate to get exact TAR at target FAR
        tar = np.interp(far_target, fpr, tpr)
        threshold = np.interp(far_target, fpr, thresholds)
    
    return tar, threshold


def calculate_eer(distances: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Equal Error Rate (EER) using interpolation
    
    Args:
        distances: Pairwise distances
        labels: Ground truth labels
        
    Returns:
        eer: Equal Error Rate
        threshold: Threshold at EER
    """
    fpr, tpr, thresholds = calculate_roc(distances, labels)
    
    # EER is where FPR = FNR (or TPR = 1 - FPR)
    fnr = 1 - tpr
    
    # Use interpolation to find exact intersection
    try:
        # Use brentq to find exact EER point
        def diff_func(threshold_val):
            idx = np.searchsorted(thresholds[::-1], threshold_val)
            idx = len(thresholds) - 1 - idx
            if idx >= len(fpr) - 1:
                return fpr[-1] - fnr[-1]
            elif idx <= 0:
                return fpr[0] - fnr[0]
            else:
                # Interpolate
                fpr_interp = np.interp(threshold_val, thresholds[::-1], fpr[::-1])
                fnr_interp = 1 - np.interp(threshold_val, thresholds[::-1], tpr[::-1])
                return fpr_interp - fnr_interp
        
        threshold = brentq(diff_func, thresholds.min(), thresholds.max())
        eer = np.interp(threshold, thresholds[::-1], fpr[::-1])
        
    except (ValueError, RuntimeError):
        # Fallback to closest point method
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        threshold = thresholds[eer_idx]
    
    return eer, threshold


def calculate_auc(distances: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Area Under ROC Curve
    
    Args:
        distances: Pairwise distances
        labels: Ground truth labels
        
    Returns:
        auc_score: Area under ROC curve
    """
    fpr, tpr, _ = calculate_roc(distances, labels)
    auc_score = auc(fpr, tpr)
    return auc_score


def calculate_degradation_rate(tar_t: float, tar_t_delta: float, 
                               time_gap: float) -> float:
    """
    Calculate degradation rate
    
    D = (TAR_t - TAR_{t+Δt}) / Δt
    
    Args:
        tar_t: TAR at time t (baseline)
        tar_t_delta: TAR at time t+Δt
        time_gap: Time gap in years
        
    Returns:
        degradation_rate: Performance degradation per year
    """
    if time_gap == 0:
        return 0.0
    
    degradation_rate = (tar_t - tar_t_delta) / time_gap
    return degradation_rate


def evaluate_verification(embeddings1: torch.Tensor, embeddings2: torch.Tensor,
                         labels: torch.Tensor, distance_metric: str = 'cosine') -> Dict:
    """
    Evaluate face verification performance
    
    Args:
        embeddings1: First set of embeddings (N, D)
        embeddings2: Second set of embeddings (N, D)
        labels: Ground truth labels (N,) - 1 for same, 0 for different
        distance_metric: 'cosine' or 'euclidean'
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Convert to numpy
    if isinstance(embeddings1, torch.Tensor):
        embeddings1 = embeddings1.cpu().numpy()
    if isinstance(embeddings2, torch.Tensor):
        embeddings2 = embeddings2.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Calculate distances
    if distance_metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        similarities = np.sum(embeddings1 * embeddings2, axis=1)
        distances = 1 - similarities
    elif distance_metric == 'euclidean':
        distances = np.linalg.norm(embeddings1 - embeddings2, axis=1)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    # Calculate metrics
    eer, eer_threshold = calculate_eer(distances, labels)
    auc_score = calculate_auc(distances, labels)
    
    # Calculate TAR at different FAR values
    tar_at_far = {}
    thresholds_at_far = {}
    for far in config.EVAL_CONFIG["far_targets"]:
        tar, threshold = calculate_tar_at_far(distances, labels, far)
        tar_at_far[f'tar@far={far}'] = tar
        thresholds_at_far[f'threshold@far={far}'] = threshold
    
    # Remove accuracy metric as it's threshold-dependent and not reliable for comparison
    # TAR@FAR and EER are more stable and meaningful metrics
    
    metrics = {
        'eer': eer,
        'eer_threshold': eer_threshold,
        'auc': auc_score,
        **tar_at_far,
        **thresholds_at_far,
    }
    
    # Add accuracy at EER for backward compatibility (but not recommended for comparison)
    accuracy_at_eer = calculate_accuracy(distances, labels, eer_threshold)
    metrics['accuracy_at_eer'] = accuracy_at_eer
    
    return metrics


def evaluate_by_time_gap(pairs_df, embeddings_dict: Dict, 
                        time_gaps: List[int] = None) -> Dict:
    """
    Evaluate performance degradation across different time gaps with balanced pairs
    
    Args:
        pairs_df: DataFrame with pair information
        embeddings_dict: Dictionary mapping image paths to embeddings
        time_gaps: List of time gaps to evaluate
        
    Returns:
        results: Dictionary with results for each time gap
    """
    if time_gaps is None:
        time_gaps = config.TEMPORAL_CONFIG["time_gaps"]
    
    results = {}
    
    for time_gap in time_gaps:
        # Filter pairs for this time gap
        gap_pairs = pairs_df[
            (pairs_df['time_gap'] >= time_gap - 0.5) & 
            (pairs_df['time_gap'] < time_gap + 0.5)
        ]
        
        if len(gap_pairs) == 0:
            print(f"Warning: No pairs found for time gap {time_gap}y")
            continue
        
        # Check class balance
        positive_count = len(gap_pairs[gap_pairs['label'] == 1])
        negative_count = len(gap_pairs[gap_pairs['label'] == 0])
        
        print(f"Time gap {time_gap}y: {positive_count} positive, {negative_count} negative pairs")
        
        # Balance pairs if significantly imbalanced
        if abs(positive_count - negative_count) > min(positive_count, negative_count) * 0.2:
            print(f"Warning: Imbalanced pairs for gap {time_gap}y. Balancing...")
            min_count = min(positive_count, negative_count)
            positive_pairs = gap_pairs[gap_pairs['label'] == 1].sample(n=min_count, random_state=config.SEED)
            negative_pairs = gap_pairs[gap_pairs['label'] == 0].sample(n=min_count, random_state=config.SEED)
            gap_pairs = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
        
        # Extract embeddings for pairs
        embeddings1 = []
        embeddings2 = []
        labels = []
        
        for _, row in gap_pairs.iterrows():
            if row['enrollment_path'] in embeddings_dict and \
               row['probe_path'] in embeddings_dict:
                embeddings1.append(embeddings_dict[row['enrollment_path']])
                embeddings2.append(embeddings_dict[row['probe_path']])
                labels.append(row['label'])
        
        if len(embeddings1) == 0:
            print(f"Warning: No valid embeddings found for time gap {time_gap}y")
            continue
        
        embeddings1 = torch.stack(embeddings1)
        embeddings2 = torch.stack(embeddings2)
        labels = torch.tensor(labels)
        
        # Final balance check after filtering
        final_positive = (labels == 1).sum().item()
        final_negative = (labels == 0).sum().item()
        print(f"Final count for gap {time_gap}y: {final_positive} positive, {final_negative} negative")
        
        # Evaluate
        metrics = evaluate_verification(
            embeddings1, embeddings2, labels,
            distance_metric=config.EVAL_CONFIG["distance_metric"]
        )
        
        # Add pair statistics to metrics
        metrics['num_pairs'] = len(labels)
        metrics['num_positive'] = final_positive
        metrics['num_negative'] = final_negative
        metrics['balance_ratio'] = final_positive / max(final_negative, 1)
        
        results[f'gap_{time_gap}y'] = metrics
    
    # Calculate degradation rates
    if len(results) > 1:
        baseline_tar = results[f'gap_{time_gaps[0]}y']['tar@far=0.001']
        
        for i, time_gap in enumerate(time_gaps[1:], 1):
            key = f'gap_{time_gap}y'
            if key in results:
                tar = results[key]['tar@far=0.001']
                degradation = calculate_degradation_rate(
                    baseline_tar, tar, time_gap - time_gaps[0]
                )
                results[key]['degradation_rate'] = degradation
    
    return results


def compute_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise similarity matrix
    
    Args:
        embeddings: Embeddings (N, D)
        
    Returns:
        similarity_matrix: Similarity matrix (N, N)
    """
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Compute cosine similarity
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    
    return similarity_matrix


def rank_n_accuracy(query_embeddings: torch.Tensor, 
                   gallery_embeddings: torch.Tensor,
                   query_labels: torch.Tensor,
                   gallery_labels: torch.Tensor,
                   n: int = 1) -> float:
    """
    Calculate Rank-N accuracy for identification
    
    Args:
        query_embeddings: Query embeddings (N_q, D)
        gallery_embeddings: Gallery embeddings (N_g, D)
        query_labels: Query labels (N_q,)
        gallery_labels: Gallery labels (N_g,)
        n: Rank N
        
    Returns:
        accuracy: Rank-N accuracy
    """
    # Compute similarity between queries and gallery
    similarities = torch.mm(
        torch.nn.functional.normalize(query_embeddings, p=2, dim=1),
        torch.nn.functional.normalize(gallery_embeddings, p=2, dim=1).t()
    )
    
    # Get top-N matches for each query
    _, top_n_indices = torch.topk(similarities, k=n, dim=1)
    
    # Check if true label is in top-N
    query_labels = query_labels.unsqueeze(1).expand(-1, n)
    top_n_labels = gallery_labels[top_n_indices]
    
    correct = (query_labels == top_n_labels).any(dim=1).float()
    accuracy = correct.mean().item()
    
    return accuracy


def visualize_time_gap_impact(results_dict: Dict, save_path: str = None, 
                              metric: str = 'tar@far=0.001',
                              title: str = None):
    """
    Visualize the impact of time gap on face recognition accuracy
    
    Args:
        results_dict: Dictionary mapping model names to their time gap results
                     Format: {model_name: {gap_Xy: {metrics...}, ...}, ...}
        save_path: Path to save the plot
        metric: Which metric to plot (default: 'tar@far=0.001')
        title: Custom title for the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers for different models
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        # Extract time gaps and metric values
        time_gaps = []
        metric_values = []
        
        # Extract and sort gap keys by numeric value
        gap_keys = [k for k in results.keys() if k.startswith('gap_')]
        gap_keys_sorted = sorted(gap_keys, key=lambda x: int(x.split('_')[1].replace('y', '')))
        
        for gap_key in gap_keys_sorted:
            # Extract time gap value (e.g., 'gap_0y' -> 0)
            gap_value = int(gap_key.split('_')[1].replace('y', ''))
            time_gaps.append(gap_value)
            
            # Get metric value with appropriate scaling
            if metric in results[gap_key]:
                value = results[gap_key][metric]
                # Convert to percentage for most metrics, but handle EER and AUC differently
                if 'eer' in metric.lower():
                    metric_values.append(value * 100)  # EER as percentage
                elif 'auc' in metric.lower():
                    metric_values.append(value)  # AUC stays as is (0-1)
                else:
                    metric_values.append(value * 100)  # TAR as percentage
            else:
                metric_values.append(0)
        
        # Plot line with markers
        plt.plot(time_gaps, metric_values, 
                marker=markers[idx % len(markers)],
                color=colors[idx],
                linewidth=2.5,
                markersize=10,
                label=model_name,
                alpha=0.8)
        
        # Add value labels on each point
        for x, y in zip(time_gaps, metric_values):
            plt.text(x, y + 1, f'{y:.1f}%', 
                    ha='center', va='bottom', fontsize=9, alpha=0.7)
    
    # Customize plot with appropriate y-label
    plt.xlabel('Time Gap (years)', fontsize=14, fontweight='bold')
    
    # Set y-label based on metric type
    if 'auc' in metric.lower():
        ylabel = f'{metric.upper()}'
    else:
        ylabel = f'{metric.upper().replace("@", " @ ")} (%)'
    
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    
    if title is None:
        title = f'Impact of Time Gap on Face Recognition Performance\n({metric.upper().replace("@", " @ ")})'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    
    return plt.gcf()


def visualize_multiple_metrics(results_dict: Dict, save_dir: str = None):
    """
    Create multiple plots for different metrics
    
    Args:
        results_dict: Dictionary mapping model names to their time gap results
        save_dir: Directory to save the plots
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Metrics to visualize (removed accuracy as it's not reliable)
    metrics_to_plot = [
        ('tar@far=0.001', 'TAR @ FAR=0.1%'),
        ('tar@far=0.01', 'TAR @ FAR=1%'),
        ('tar@far=0.0001', 'TAR @ FAR=0.01%'),
        ('eer', 'Equal Error Rate (EER)'),
        ('auc', 'AUC'),
    ]
    
    for metric_key, metric_label in metrics_to_plot:
        # Check if metric exists in results
        has_metric = False
        for model_results in results_dict.values():
            for gap_results in model_results.values():
                if metric_key in gap_results:
                    has_metric = True
                    break
            if has_metric:
                break
        
        if not has_metric:
            continue
        
        # Create plot with special handling for EER (lower is better)
        save_path = None
        if save_dir:
            safe_filename = metric_key.replace('@', '_at_').replace('=', '').replace('.', '_')
            save_path = os.path.join(save_dir, f'time_gap_impact_{safe_filename}.png')
        
        # For EER, we want to show it as percentage and note that lower is better
        plot_title = f'Impact of Time Gap on {metric_label}'
        if 'eer' in metric_key.lower():
            plot_title += ' (Lower is Better)'
        
        visualize_time_gap_impact(
            results_dict,
            save_path=save_path,
            metric=metric_key,
            title=plot_title
        )


def visualize_degradation_rates(results_dict: Dict, save_path: str = None):
    """
    Visualize degradation rates across models
    
    Args:
        results_dict: Dictionary mapping model names to their time gap results
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    models = []
    degradation_rates = []
    time_gaps = []
    
    for model_name, results in results_dict.items():
        for gap_key, metrics in results.items():
            if 'degradation_rate' in metrics and gap_key.startswith('gap_'):
                gap_value = int(gap_key.split('_')[1].replace('y', ''))
                models.append(f'{model_name}\n(Gap: {gap_value}y)')
                degradation_rates.append(metrics['degradation_rate'] * 100)  # Percentage
                time_gaps.append(gap_value)
    
    if len(models) == 0:
        print("No degradation rate data found")
        return None
    
    # Create bar plot
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(models)))
    bars = plt.bar(range(len(models)), degradation_rates, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, degradation_rates)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.2f}%/year',
                ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Model and Time Gap', fontsize=12, fontweight='bold')
    plt.ylabel('Degradation Rate (%/year)', fontsize=12, fontweight='bold')
    plt.title('Performance Degradation Rate Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved degradation plot to {save_path}")
    
    plt.show()
    
    return plt.gcf()


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Generate dummy data
    n_samples = 1000
    embedding_dim = 512
    
    embeddings1 = torch.randn(n_samples, embedding_dim)
    embeddings2 = torch.randn(n_samples, embedding_dim)
    
    # Create labels (50% positive, 50% negative)
    labels = torch.randint(0, 2, (n_samples,))
    
    # For positive pairs, make embeddings similar
    positive_mask = labels == 1
    embeddings2[positive_mask] = embeddings1[positive_mask] + 0.1 * torch.randn(positive_mask.sum(), embedding_dim)
    
    # Normalize embeddings
    embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
    embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
    
    # Evaluate
    print("Testing evaluation metrics...")
    metrics = evaluate_verification(embeddings1, embeddings2, labels)
    
    print("\nVerification metrics:")
    print(f"  EER: {metrics['eer']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  TAR@FAR=0.1%: {metrics['tar@far=0.001']:.4f}")
    print(f"  TAR@FAR=1%: {metrics['tar@far=0.01']:.4f}")
    
    # Test rank-N accuracy
    n_queries = 100
    n_gallery = 500
    query_emb = torch.randn(n_queries, embedding_dim)
    gallery_emb = torch.randn(n_gallery, embedding_dim)
    query_labels = torch.randint(0, 50, (n_queries,))
    gallery_labels = torch.randint(0, 50, (n_gallery,))
    
    rank1_acc = rank_n_accuracy(query_emb, gallery_emb, query_labels, gallery_labels, n=1)
    rank5_acc = rank_n_accuracy(query_emb, gallery_emb, query_labels, gallery_labels, n=5)
    
    print(f"\nIdentification metrics:")
    print(f"  Rank-1 Accuracy: {rank1_acc:.4f}")
    print(f"  Rank-5 Accuracy: {rank5_acc:.4f}")
