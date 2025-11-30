"""
Visualization utilities for face recognition results
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from typing import Dict, List
import config


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = config.VIS_CONFIG["dpi"]
plt.rcParams['figure.figsize'] = config.VIS_CONFIG["figsize"]


def plot_roc_curve(fpr, tpr, auc_score, save_path=None, title="ROC Curve"):
    """
    Plot ROC curve
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    
    plt.show()


def plot_multiple_roc_curves(roc_data: Dict, save_path=None, title="ROC Curves Comparison"):
    """
    Plot multiple ROC curves for comparison
    
    Args:
        roc_data: Dictionary mapping model names to (fpr, tpr, auc) tuples
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(roc_data)))
    
    for (model_name, (fpr, tpr, auc_score)), color in zip(roc_data.items(), colors):
        plt.plot(fpr, tpr, linewidth=2, color=color, 
                label=f'{model_name} (AUC = {auc_score:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")
    
    plt.show()


def plot_tar_vs_time_gap(results_dict: Dict, save_path=None):
    """
    Plot TAR vs Time Gap (Performance degradation over time)
    
    Args:
        results_dict: Dictionary with time gap results
        save_path: Path to save figure
    """
    time_gaps = []
    tar_values = []
    
    # Extract data
    for gap_key, metrics in results_dict.items():
        if 'gap_' in gap_key:
            gap_years = int(gap_key.split('_')[1].replace('y', ''))
            time_gaps.append(gap_years)
            tar_values.append(metrics.get('tar@far=0.001', 0))
    
    # Sort by time gap
    sorted_data = sorted(zip(time_gaps, tar_values))
    time_gaps, tar_values = zip(*sorted_data)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(time_gaps, tar_values, marker='o', linewidth=2, markersize=8)
    
    plt.xlabel('Time Gap (years)', fontsize=12)
    plt.ylabel('TAR @ FAR = 0.1%', fontsize=12)
    plt.title('Performance Degradation Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(time_gaps, tar_values, 1)
    p = np.poly1d(z)
    plt.plot(time_gaps, p(time_gaps), "--", alpha=0.5, 
            label=f'Trend (slope={z[0]:.4f})')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved TAR vs Time Gap plot to {save_path}")
    
    plt.show()


def plot_multiple_tar_vs_time_gap(models_results: Dict, save_path=None):
    """
    Compare TAR vs Time Gap for multiple models
    
    Args:
        models_results: Dictionary mapping model names to results
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 7))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models_results)))
    
    for (model_name, results_dict), color in zip(models_results.items(), colors):
        time_gaps = []
        tar_values = []
        
        for gap_key, metrics in results_dict.items():
            if 'gap_' in gap_key:
                gap_years = int(gap_key.split('_')[1].replace('y', ''))
                time_gaps.append(gap_years)
                tar_values.append(metrics.get('tar@far=0.001', 0))
        
        # Sort by time gap
        if time_gaps:
            sorted_data = sorted(zip(time_gaps, tar_values))
            time_gaps, tar_values = zip(*sorted_data)
            
            plt.plot(time_gaps, tar_values, marker='o', linewidth=2, 
                    markersize=8, color=color, label=model_name)
    
    plt.xlabel('Time Gap (years)', fontsize=12)
    plt.ylabel('TAR @ FAR = 0.1%', fontsize=12)
    plt.title('Performance Degradation Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()


def plot_embedding_trajectory(embeddings_dict: Dict, identity: str, 
                             ages: List[int], save_path=None):
    """
    Plot embedding trajectory for one person across different ages (t-SNE/UMAP)
    
    Args:
        embeddings_dict: Dictionary mapping (identity, age) to embeddings
        identity: Identity to visualize
        ages: List of ages for this identity
        save_path: Path to save figure
    """
    import torch
    
    # Collect embeddings for this identity
    embeddings = []
    valid_ages = []
    
    for age in sorted(ages):
        key = (identity, age)
        if key in embeddings_dict:
            emb = embeddings_dict[key]
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            embeddings.append(emb)
            valid_ages.append(age)
    
    if len(embeddings) < 2:
        print(f"Not enough embeddings for {identity}")
        return
    
    embeddings = np.array(embeddings)
    
    # Create figure with two subplots (t-SNE and UMAP)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # t-SNE
    if len(embeddings) >= 2:
        tsne = TSNE(n_components=2, random_state=config.SEED, 
                   perplexity=min(5, len(embeddings)-1))
        embeddings_2d_tsne = tsne.fit_transform(embeddings)
        
        # Plot trajectory
        axes[0].plot(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1], 
                    'o-', linewidth=2, markersize=10, alpha=0.7)
        
        # Annotate with ages
        for i, age in enumerate(valid_ages):
            axes[0].annotate(f'{age}', 
                           (embeddings_2d_tsne[i, 0], embeddings_2d_tsne[i, 1]),
                           fontsize=10, ha='center')
        
        axes[0].set_title(f't-SNE: Embedding Trajectory for {identity}', fontsize=12)
        axes[0].set_xlabel('Dimension 1')
        axes[0].set_ylabel('Dimension 2')
        axes[0].grid(True, alpha=0.3)
    
    # UMAP
    if len(embeddings) >= 2:
        reducer = umap.UMAP(n_components=2, random_state=config.SEED,
                          n_neighbors=min(5, len(embeddings)-1))
        embeddings_2d_umap = reducer.fit_transform(embeddings)
        
        # Plot trajectory
        axes[1].plot(embeddings_2d_umap[:, 0], embeddings_2d_umap[:, 1], 
                    's-', linewidth=2, markersize=10, alpha=0.7)
        
        # Annotate with ages
        for i, age in enumerate(valid_ages):
            axes[1].annotate(f'{age}', 
                           (embeddings_2d_umap[i, 0], embeddings_2d_umap[i, 1]),
                           fontsize=10, ha='center')
        
        axes[1].set_title(f'UMAP: Embedding Trajectory for {identity}', fontsize=12)
        axes[1].set_xlabel('Dimension 1')
        axes[1].set_ylabel('Dimension 2')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved embedding trajectory to {save_path}")
    
    plt.show()


def plot_degradation_rates(degradation_data: Dict, save_path=None):
    """
    Plot degradation rates as bar chart
    
    Args:
        degradation_data: Dictionary mapping model names to degradation rates
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    models = list(degradation_data.keys())
    rates = list(degradation_data.values())
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = plt.bar(models, rates, color=colors, alpha=0.8)
    
    plt.ylabel('Degradation Rate (TAR loss per year)', fontsize=12)
    plt.title('Performance Degradation Rate Comparison', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved degradation rates to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Different', 'Same'],
               yticklabels=['Different', 'Same'])
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_quality_score_distribution(quality_scores, save_path=None):
    """
    Plot distribution of face quality scores
    
    Args:
        quality_scores: Array of quality scores
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(quality_scores, bins=50, alpha=0.7, edgecolor='black')
    
    plt.xlabel('Quality Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Face Quality Scores', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_q = np.mean(quality_scores)
    median_q = np.median(quality_scores)
    plt.axvline(mean_q, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_q:.3f}')
    plt.axvline(median_q, color='g', linestyle='--', linewidth=2, label=f'Median: {median_q:.3f}')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved quality distribution to {save_path}")
    
    plt.show()


def create_evaluation_report(results_dict: Dict, output_dir: str):
    """
    Create comprehensive evaluation report with all visualizations
    
    Args:
        results_dict: Dictionary with all evaluation results
        output_dir: Directory to save report and figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating evaluation report in {output_dir}...")
    
    # 1. TAR vs Time Gap
    if any('gap_' in key for key in results_dict.keys()):
        plot_tar_vs_time_gap(
            results_dict,
            save_path=os.path.join(output_dir, 'tar_vs_time_gap.png')
        )
    
    # 2. Save metrics to CSV
    metrics_df = pd.DataFrame(results_dict).T
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'))
    print(f"Saved metrics to {os.path.join(output_dir, 'metrics.csv')}")
    
    # 3. Create summary text report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FACE RECOGNITION EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        for protocol, metrics in results_dict.items():
            f.write(f"\n{protocol}:\n")
            f.write("-"*40 + "\n")
            
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\n")
    
    print(f"Created evaluation report: {report_path}")


if __name__ == "__main__":
    # Test visualizations with dummy data
    print("Testing visualizations...")
    
    # Test ROC curve
    fpr = np.linspace(0, 1, 100)
    tpr = fpr ** 0.5  # Dummy TPR
    auc_score = 0.85
    
    plot_roc_curve(fpr, tpr, auc_score, title="Test ROC Curve")
    
    # Test TAR vs Time Gap
    dummy_results = {
        'gap_1y': {'tar@far=0.001': 0.95},
        'gap_2y': {'tar@far=0.001': 0.92},
        'gap_4y': {'tar@far=0.001': 0.87},
        'gap_6y': {'tar@far=0.001': 0.82},
    }
    
    plot_tar_vs_time_gap(dummy_results)
    
    print("Visualization tests complete!")
