"""
Main pipeline for Age-Invariant Face Recognition
"""
import os
import torch
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

import config
from utils.data_processor import DataProcessor, create_train_val_split, add_identity_from_annotations
from utils.morph2_processor import MORPH2Processor
from utils.pair_generator import PairGenerator, generate_all_protocols
from train import Trainer
from evaluate import Evaluator
from utils.visualization import (
    plot_tar_vs_time_gap,
    plot_multiple_tar_vs_time_gap,
    create_evaluation_report
)
from utils.statistical_analysis import StatisticalAnalyzer, create_statistical_report
from test_image_quality_robustness import RobustnessEvaluator, visualize_robustness_results


def step1_data_preprocessing(dataset='agedb_30'):
    """
    Step 1: Data Preprocessing Pipeline
    - For MORPH-2: Load CSV metadata with age information
    - For AgeDB-30: Use pre-aligned images
    """
    print("\n" + "="*80)
    print("STEP 1: DATA PREPROCESSING")
    print("="*80)
    
    output_folder = os.path.join(config.OUTPUT_ROOT, "processed_images")
    
    if dataset == 'morph_2':
        print(f"\nProcessing MORPH-2 dataset...")
        processor = MORPH2Processor()
        
        # Process train split
        train_df = processor.process_dataset('train', output_folder=os.path.join(output_folder, 'train'))
        
        # Process validation split
        val_df = processor.process_dataset('val', output_folder=os.path.join(output_folder, 'val'))
        
        # Remap identities to be contiguous
        all_identities = pd.concat([train_df['identity'], val_df['identity']]).unique()
        identity_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_identities))}
        
        train_df['identity'] = train_df['identity'].map(identity_map)
        val_df['identity'] = val_df['identity'].map(identity_map)
        
        # Save metadata
        train_df.to_csv(os.path.join(output_folder, 'train_metadata.csv'), index=False)
        val_df.to_csv(os.path.join(output_folder, 'val_metadata.csv'), index=False)
        
        # Combine for overall metadata
        metadata_df = pd.concat([train_df, val_df], ignore_index=True)
        metadata_df.to_csv(os.path.join(output_folder, 'metadata.csv'), index=False)
        
    elif dataset == 'agedb_30':
        print(f"\nProcessing AgeDB-30 dataset...")
        # For agedb_30_112x112, images are already aligned, so skip detection
        processor = DataProcessor(use_quality_filter=False, skip_detection=True)
        
        # Process AgeDB-30 dataset
        image_folder = config.AGEDB_CONFIG["image_folder"]
        
        if not os.path.exists(image_folder):
            print(f"Error: Image folder not found: {image_folder}")
            print("Please download AgeDB-30 dataset first")
            return None
        
        print(f"\nProcessing images from: {image_folder}")
        metadata_df = processor.process_dataset(image_folder, output_folder)
        
        # Add identity information from annotation file
        annotation_file = config.AGEDB_CONFIG["annotation_file"]
        if os.path.exists(annotation_file):
            print(f"\nAdding identity information from: {annotation_file}")
            metadata_df = add_identity_from_annotations(metadata_df, annotation_file)
        else:
            print(f"Warning: Annotation file not found: {annotation_file}")
            print("Assigning unique identity to each image")
            metadata_df['identity'] = range(len(metadata_df))
        
        # Save updated metadata with identity information
        metadata_path = os.path.join(output_folder, 'metadata.csv')
        metadata_df.to_csv(metadata_path, index=False)
        
        # Create train/val split
        train_df, val_df = create_train_val_split(metadata_df, val_ratio=0.2)
        
        # Save splits
        train_df.to_csv(os.path.join(output_folder, 'train_metadata.csv'), index=False)
        val_df.to_csv(os.path.join(output_folder, 'val_metadata.csv'), index=False)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print("\nData preprocessing complete!")
    return metadata_df

def step2_generate_pairs(dataset='agedb_30'):
    """
    Step 2: Pair Generation
    - For MORPH-2: Generate temporal pairs using age information
    - For AgeDB-30: Use existing annotation file
    """
    print("\n" + "="*80)
    print("STEP 2: PAIR GENERATION")
    print("="*80)
    
    output_dir = os.path.join(config.OUTPUT_ROOT, "pairs")
    os.makedirs(output_dir, exist_ok=True)
    
    if dataset == 'morph_2':
        print(f"\nGenerating temporal pairs for MORPH-2...")
        
        # Load metadata
        metadata_path = os.path.join(config.OUTPUT_ROOT, "processed_images", "metadata.csv")
        if not os.path.exists(metadata_path):
            print(f"Error: Metadata not found: {metadata_path}")
            print("Please run step1 (preprocess) first")
            return
        
        metadata_df = pd.read_csv(metadata_path)
        
        # Generate temporal pairs (limit to 128 pairs per gap)
        processor = MORPH2Processor()
        pairs_df = processor.generate_temporal_pairs(
            metadata_df,
            output_path=os.path.join(output_dir, "morph2_temporal_pairs.csv"),
            time_gaps=config.TEMPORAL_CONFIG["time_gaps"],
            pairs_per_gap=128
        )
        
    elif dataset == 'agedb_30':
        print(f"\nGenerating pairs for AgeDB-30...")
        
        annotation_file = config.AGEDB_CONFIG["annotation_file"]
        
        if not os.path.exists(annotation_file):
            print(f"Error: Annotation file not found: {annotation_file}")
            return
        
        print(f"Loading pairs from: {annotation_file}")
        
        # Load metadata to map original paths to processed paths
        metadata_path = os.path.join(config.OUTPUT_ROOT, "processed_images", "metadata.csv")
        if not os.path.exists(metadata_path):
            print(f"Error: Metadata not found: {metadata_path}")
            print("Please run step1 (preprocess) first")
            return
        
        metadata_df = pd.read_csv(metadata_path)
        
        # Create mapping from image_id to aligned_path
        id_to_path = {}
        for _, row in metadata_df.iterrows():
            img_id = str(row['image_id'])
            id_to_path[img_id] = row['aligned_path']
        
        # Read annotation file
        pairs = []
        skipped = 0
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    label = int(parts[0])
                    img1_name = os.path.basename(parts[1])
                    img2_name = os.path.basename(parts[2])
                    
                    # Extract image IDs (remove extension)
                    img1_id = os.path.splitext(img1_name)[0]
                    img2_id = os.path.splitext(img2_name)[0]
                    
                    # Map to processed image paths
                    if img1_id in id_to_path and img2_id in id_to_path:
                        img1_full = id_to_path[img1_id]
                        img2_full = id_to_path[img2_id]
                        
                        pairs.append({
                            'enrollment_path': img1_full,
                            'probe_path': img2_full,
                            'label': label
                        })
                    else:
                        skipped += 1
        
        # Create DataFrame
        pairs_df = pd.DataFrame(pairs)
        
        pairs_path = os.path.join(output_dir, "agedb_30_pairs.csv")
        pairs_df.to_csv(pairs_path, index=False)
        
        print(f"\nPair generation complete:")
        print(f"  Total pairs: {len(pairs_df)}")
        print(f"  Positive pairs (same person): {len(pairs_df[pairs_df['label'] == 1])}")
        print(f"  Negative pairs (different person): {len(pairs_df[pairs_df['label'] == 0])}")
        print(f"  Skipped (images not in processed): {skipped}")
        print(f"  Saved to: {pairs_path}")
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def step3_train_baseline_models(backbone='magface', epochs=None):
    """
    Step 3: Train Baseline Models
    - Train specified backbone model (ArcFace or MagFace)
    - Evaluate on standard benchmarks
    
    Args:
        backbone: 'arcface' or 'magface' - which model to train
        epochs: Number of training epochs (None = use config default)
    """
    print("\n" + "="*80)
    print(f"STEP 3: TRAIN BASELINE MODELS (Backbone: {backbone.upper()})")
    print("="*80)
    
    # Load data
    train_df = pd.read_csv(os.path.join(config.OUTPUT_ROOT, "processed_images", "train_metadata.csv"))
    val_df = pd.read_csv(os.path.join(config.OUTPUT_ROOT, "processed_images", "val_metadata.csv"))
    
    # Remap identity labels to be contiguous from 0 to num_classes-1
    unique_identities = sorted(train_df['identity'].unique())
    identity_map = {old_id: new_id for new_id, old_id in enumerate(unique_identities)}
    train_df['identity'] = train_df['identity'].map(identity_map)
    val_df['identity'] = val_df['identity'].map(identity_map)
    
    # Get number of classes
    num_classes = train_df['identity'].nunique()
    print(f"Number of classes: {num_classes}")
    print(f"Identity range: {train_df['identity'].min()} - {train_df['identity'].max()}")
    
    # Train selected backbone
    num_epochs = epochs if epochs is not None else config.TRAIN_CONFIG["num_epochs"]
    print(f"\n--- Training {backbone.upper()} (Epochs: {num_epochs}) ---")
    trainer = Trainer(model_type=backbone, num_classes=num_classes, use_temporal=False)
    trainer.setup_dataloader(train_df, val_df)
    trainer.train(num_epochs=num_epochs)
    
    print(f"\n{backbone.upper()} baseline training complete!")

def step5_evaluate_models(backbone='magface', evaluate_dataset=None):
    """
    Step 5: Evaluation & Analysis
    - Extract features
    - Calculate TAR @ FAR
    - Measure degradation rate
    - Statistical analysis
    
    Args:
        backbone: 'arcface' or 'magface' - backbone for temporal model
        evaluate_dataset: Which dataset to evaluate on (None = use active dataset)
    """
    print("\n" + "="*80)
    print("STEP 5: EVALUATION & ANALYSIS")
    if evaluate_dataset:
        print(f"Evaluating on: {evaluate_dataset.upper()}")
    print(f"Temporal backbone: {backbone.upper()}")
    print("="*80)
    
    models_to_evaluate = {
        'ArcFace': {
            'type': 'arcface',
            'checkpoint': os.path.join(config.MODEL_ROOT, 'arcface_best_model.pth')
        },
        'MagFace': {
            'type': 'magface',
            'checkpoint': os.path.join(config.MODEL_ROOT, 'magface_best_model.pth')
        }
    }
    
    all_results = {}
    
    for model_name, model_info in models_to_evaluate.items():
        print(f"\n--- Evaluating {model_name} ---")
        
        if not os.path.exists(model_info['checkpoint']):
            print(f"Checkpoint not found: {model_info['checkpoint']}")
            continue
        
        evaluator = Evaluator(
            model_type=model_info['type'],
            checkpoint_path=model_info['checkpoint']
        )
        
        results = evaluator.evaluate_all_protocols(evaluate_dataset=evaluate_dataset)
        all_results[model_name] = results
    
    print("\nEvaluation complete!")
    
    # Create visualizations for time gap analysis
    print("\n" + "="*60)
    print("CREATING TIME GAP VISUALIZATIONS")
    print("="*60)
    
    # Extract time gap results from all_results
    time_gap_results = {}
    for model_name, protocols in all_results.items():
        if protocols:
            for protocol_name, protocol_results in protocols.items():
                if isinstance(protocol_results, dict):
                    # Check if this protocol has time gap results
                    has_time_gaps = any(key.startswith('gap_') for key in protocol_results.keys())
                    if has_time_gaps:
                        # Create a key combining model and protocol
                        result_key = f"{model_name} ({protocol_name})"
                        time_gap_results[result_key] = protocol_results
    
    if time_gap_results:
        from utils.metrics import visualize_time_gap_impact, visualize_multiple_metrics, visualize_degradation_rates
        
        # Create output directory for visualizations
        viz_dir = os.path.join(config.OUTPUT_ROOT, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Visualize multiple metrics
        print("\nGenerating time gap impact plots...")
        visualize_multiple_metrics(time_gap_results, save_dir=viz_dir)
        
        # Visualize degradation rates
        print("\nGenerating degradation rate comparison...")
        degradation_path = os.path.join(viz_dir, "degradation_rates.png")
        visualize_degradation_rates(time_gap_results, save_path=degradation_path)
        
        print(f"\nVisualization complete! Saved to: {viz_dir}")
    else:
        print("\nNo time gap results found for visualization.")
    
    return all_results

def step6_test_robustness(backbone='magface', test_dataset=None):
    """
    Step 6: Test Image Quality Robustness
    - Test resolution degradation
    - Test blur robustness
    - Test noise robustness (Gaussian & Salt-Pepper)
    - Compare ArcFace vs MagFace
    
    Args:
        backbone: 'arcface' or 'magface' - backbone for temporal model
        test_dataset: Dataset to test on (None = use active dataset)
    """
    print("\n" + "="*80)
    print("STEP 6: IMAGE QUALITY ROBUSTNESS TESTING")
    if test_dataset:
        print(f"Testing on: {test_dataset.upper()}")
    print(f"Temporal backbone: {backbone.upper()}")
    print("="*80)
    
    # Determine pairs file
    if test_dataset:
        config.set_active_dataset(test_dataset)
    
    if config.ACTIVE_DATASET == 'morph_2':
        pairs_path = os.path.join(config.OUTPUT_ROOT, "pairs", "morph2_temporal_pairs.csv")
    else:
        pairs_path = os.path.join(config.OUTPUT_ROOT, "pairs", "agedb_30_pairs.csv")
    
    if not os.path.exists(pairs_path):
        print(f"Error: Pairs not found: {pairs_path}")
        print("Please run step2_generate_pairs first")
        return
    
    print(f"Loading pairs from: {pairs_path}")
    pairs_df = pd.read_csv(pairs_path)
    
    # Rename columns if needed (for compatibility)
    if 'img1_path' in pairs_df.columns and 'enrollment_path' not in pairs_df.columns:
        pairs_df = pairs_df.rename(columns={
            'img1_path': 'enrollment_path',
            'img2_path': 'probe_path'
        })
    
    # Limit pairs for faster testing (optional)
    # Use a subset for quick testing, comment out for full evaluation
    # pairs_df = pairs_df.sample(n=min(500, len(pairs_df)), random_state=42)
    print(f"Testing with {len(pairs_df)} pairs")
    
    # Output directory
    output_dir = os.path.join(config.OUTPUT_ROOT, "robustness_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Test models
    models_to_test = {
        'ArcFace': {
            'type': 'arcface',
            'checkpoint': os.path.join(config.MODEL_ROOT, 'arcface_best_model.pth')
        },
        'MagFace': {
            'type': 'magface',
            'checkpoint': os.path.join(config.MODEL_ROOT, 'magface_best_model.pth')
        }
    }
    
    all_robustness_results = {}
    
    for model_name, model_info in models_to_test.items():
        print(f"\n{'='*80}")
        print(f"Testing {model_name} Robustness")
        print(f"{'='*80}")
        
        if not os.path.exists(model_info['checkpoint']):
            print(f"Checkpoint not found: {model_info['checkpoint']}")
            continue
        
        evaluator = RobustnessEvaluator(
            model_type=model_info['type'],
            checkpoint_path=model_info['checkpoint']
        )
        
        # Run all robustness tests
        model_output_dir = os.path.join(output_dir, model_name.replace(' ', '_'))
        results = evaluator.run_all_robustness_tests(pairs_df, output_dir=model_output_dir)
        all_robustness_results[model_name] = results
    
    # Create comparison visualizations
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    viz_dir = os.path.join(output_dir, "comparison_plots")
    visualize_robustness_results(all_robustness_results, output_dir=viz_dir)
    
    print("\nRobustness testing complete!")
    print(f"Results saved to: {output_dir}")
    
    return all_robustness_results

def step7_visualization_and_statistics(all_results):
    """
    Step 7: Visualization & Statistical Analysis
    - Plot ROC curves
    - Plot TAR vs Time Gap
    - Embedding trajectory visualization
    - LME model analysis
    """
    print("\n" + "="*80)
    print("STEP 7: VISUALIZATION & STATISTICAL ANALYSIS")
    print("="*80)
    
    output_dir = os.path.join(config.OUTPUT_ROOT, "final_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualizations
    print("\nCreating visualizations...")
    
    # Compare TAR vs Time Gap for all models
    if all_results:
        plot_multiple_tar_vs_time_gap(
            all_results,
            save_path=os.path.join(output_dir, 'model_comparison.png')
        )
    
    # Statistical analysis
    print("\nPerforming statistical analysis...")
    analyzer = StatisticalAnalyzer(confidence_level=0.95)
    
    # Create comprehensive report
    if all_results:
        for model_name, results in all_results.items():
            report_dir = os.path.join(output_dir, model_name.replace(' ', '_'))
            create_evaluation_report(results, report_dir)
    
    print("\nVisualization and analysis complete!")
    print(f"Results saved to: {output_dir}")

def step8_age_group_analysis(backbone='magface'):
    """
    Step 8: Age Group Analysis
    - Chia data thành 3 nhóm tuổi: 16-29, 30-49, 50-70
    - Generate pairs cho mỗi nhóm tuổi
    - Đánh giá hiệu suất FMR/TAR cho từng nhóm
    - So sánh accuracy và similarity score giữa các nhóm
    """
    print("\n" + "="*80)
    print("STEP 8: AGE GROUP ANALYSIS")
    print("Analyzing performance across different age groups")
    print("Age Groups: 16-29, 30-49, 50-70")
    print("="*80)
    
    # Load metadata
    metadata_path = os.path.join(config.OUTPUT_ROOT, "processed_images", "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata not found: {metadata_path}")
        print("Please run step1 (preprocess) first")
        return
    
    metadata_df = pd.read_csv(metadata_path)
    
    # Extract real age from filename
    def extract_real_age(filename):
        """Extract real age from filename like '305277_05M40.JPG'"""
        try:
            # Get the part after '_' and before '.JPG'
            name_part = os.path.splitext(filename)[0]  # Remove extension
            age_part = name_part.split('_')[-1]  # Get last part after '_'
            # Extract age from pattern like '05M40' - age is the last 2 digits
            age_str = age_part[-2:]  # Last 2 characters
            return int(age_str)
        except:
            return None
    
    # Method 1: Add 16 to existing age (as suggested)
    metadata_df['real_age_method1'] = metadata_df['age'] + 16
    
    # Method 2: Extract from filename
    metadata_df['real_age_method2'] = metadata_df['filename'].apply(extract_real_age)
    
    # Choose method based on which gives more reasonable age range
    method1_range = (metadata_df['real_age_method1'].min(), metadata_df['real_age_method1'].max())
    method2_range = (metadata_df['real_age_method2'].min(), metadata_df['real_age_method2'].max())
    
    print(f"Age extraction methods:")
    print(f"  Method 1 (age + 16): {method1_range}")
    print(f"  Method 2 (from filename): {method2_range}")
    
    # Use method 2 if available and reasonable, otherwise method 1
    if metadata_df['real_age_method2'].notna().sum() > len(metadata_df) * 0.5 and method2_range[1] <= 80:
        metadata_df['real_age'] = metadata_df['real_age_method2']
        print(f"Using method 2 (filename extraction)")
    else:
        metadata_df['real_age'] = metadata_df['real_age_method1']
        print(f"Using method 1 (age + 16)")
    
    # Remove rows with invalid ages
    metadata_df = metadata_df.dropna(subset=['real_age'])
    metadata_df = metadata_df[(metadata_df['real_age'] >= 16) & (metadata_df['real_age'] <= 70)]
    
    print(f"\nFinal age range: {metadata_df['real_age'].min()} - {metadata_df['real_age'].max()}")
    print(f"Total samples: {len(metadata_df)}")
    
    # Define age groups
    def categorize_age(age):
        if 16 <= age <= 29:
            return 'Young (16-29)'
        elif 30 <= age <= 49:
            return 'Middle (30-49)'
        elif 50 <= age <= 70:
            return 'Senior (50-70)'
        else:
            return 'Other'
    
    metadata_df['age_group'] = metadata_df['real_age'].apply(categorize_age)
    
    # Remove 'Other' group
    metadata_df = metadata_df[metadata_df['age_group'] != 'Other']
    
    # Print age group statistics
    print("\nAge group distribution:")
    age_group_stats = metadata_df.groupby('age_group').agg({
        'identity': 'nunique',
        'real_age': ['count', 'mean', 'std']
    })
    age_group_stats.columns = ['Unique_Identities', 'Sample_Count', 'Mean_Age', 'Std_Age']
    print(age_group_stats)
    
    # Find minimum number of unique identities across all groups
    min_identities = metadata_df.groupby('age_group')['identity'].nunique().min()
    print(f"\nBalancing data: Using {min_identities} unique identities per age group")
    
    # Create output directory for age group analysis
    age_analysis_dir = os.path.join(config.OUTPUT_ROOT, "age_group_analysis")
    os.makedirs(age_analysis_dir, exist_ok=True)
    
    # Fixed number of pairs per age group
    pairs_per_group = 1281  # Balance across all groups
    
    # Generate pairs for each age group
    all_age_group_results = {}
    
    for age_group in ['Young (16-29)', 'Middle (30-49)', 'Senior (50-70)']:
        print(f"\n{'-'*60}")
        print(f"Processing {age_group}")
        print(f"{'-'*60}")
        
        # Filter data for this age group
        group_data = metadata_df[metadata_df['age_group'] == age_group]
        
        # Sample min_identities unique identities from this group
        available_identities = group_data['identity'].unique()
        if len(available_identities) > min_identities:
            np.random.seed(42)
            selected_identities = np.random.choice(available_identities, size=min_identities, replace=False)
            group_data = group_data[group_data['identity'].isin(selected_identities)]
            print(f"Sampled {min_identities} identities from {len(available_identities)} available")
        
        if len(group_data) < 10:
            print(f"Skipping {age_group} - insufficient data ({len(group_data)} samples)")
            continue
        
        print(f"Samples: {len(group_data)}, Identities: {group_data['identity'].nunique()}")
        
        # Generate pairs for this age group
        np.random.seed(42)
        
        # Generate positive pairs (same identity within age group)
        positive_pairs = []
        identity_groups = group_data.groupby('identity')
        
        # Generate exactly pairs_per_group positive pairs
        identities_with_pairs = [id for id, data in identity_groups if len(data) >= 2]
        
        if len(identities_with_pairs) == 0:
            print(f"Skipping {age_group} - no identities with multiple samples")
            continue
        
        for _ in range(pairs_per_group):
            # Randomly select an identity with multiple samples
            identity = np.random.choice(identities_with_pairs)
            identity_data = identity_groups.get_group(identity)
            
            # Sample 2 different images from this identity
            sample_pair = identity_data.sample(n=2, replace=False)
            img1, img2 = sample_pair.iloc[0], sample_pair.iloc[1]
            positive_pairs.append({
                'enrollment_path': img1['aligned_path'],
                'probe_path': img2['aligned_path'],
                'label': 1,
                'age_group': age_group,
                'enrollment_age': img1['real_age'],
                'probe_age': img2['real_age'],
                'identity': identity
            })
        
        # Generate negative pairs (different identities within age group)
        negative_pairs = []
        identities = list(identity_groups.groups.keys())
        
        if len(identities) >= 2:
            # Generate exactly pairs_per_group negative pairs
            for _ in range(pairs_per_group):
                id1, id2 = np.random.choice(identities, size=2, replace=False)
                img1 = identity_groups.get_group(id1).sample(n=1).iloc[0]
                img2 = identity_groups.get_group(id2).sample(n=1).iloc[0]
                
                negative_pairs.append({
                    'enrollment_path': img1['aligned_path'],
                    'probe_path': img2['aligned_path'],
                    'label': 0,
                    'age_group': age_group,
                    'enrollment_age': img1['real_age'],
                    'probe_age': img2['real_age'],
                    'identity': f"{id1}_{id2}"
                })
        
        # Combine pairs
        all_pairs = positive_pairs + negative_pairs
        
        if len(all_pairs) == 0:
            print(f"No pairs generated for {age_group}")
            continue
        
        pairs_df = pd.DataFrame(all_pairs)
        
        # Save pairs for this age group
        pairs_path = os.path.join(age_analysis_dir, f"{age_group.replace(' ', '_').replace('(', '').replace(')', '')}_pairs.csv")
        pairs_df.to_csv(pairs_path, index=False)
        
        print(f"Generated {len(pairs_df)} pairs:")
        print(f"  Positive: {len(pairs_df[pairs_df['label'] == 1])}")
        print(f"  Negative: {len(pairs_df[pairs_df['label'] == 0])}")
        
        # Evaluate models on this age group
        print(f"\nEvaluating models on {age_group}...")
        
        models_to_evaluate = {
            'ArcFace': {
                'type': 'arcface',
                'checkpoint': os.path.join(config.MODEL_ROOT, 'arcface_best_model.pth')
            },
            'MagFace': {
                'type': 'magface', 
                'checkpoint': os.path.join(config.MODEL_ROOT, 'magface_best_model.pth')
            }
        }
        
        age_group_results = {}
        
        for model_name, model_info in models_to_evaluate.items():
            if not os.path.exists(model_info['checkpoint']):
                print(f"  Skipping {model_name} - checkpoint not found")
                continue
            
            print(f"  Evaluating {model_name}...")
            
            try:
                model_evaluator = Evaluator(
                    model_type=model_info['type'],
                    checkpoint_path=model_info['checkpoint']
                )
                
                # Evaluate on pairs
                
                # Get similarity scores
                scores = model_evaluator.compute_similarity_scores_from_pairs(pairs_df)
                
                if scores is not None and len(scores) > 0:
                    # Compute metrics
                    labels = pairs_df['label'].values
                    
                    # ROC curve
                    fpr, tpr, thresholds = roc_curve(labels, scores)
                    roc_auc = auc(fpr, tpr)
                    
                    # Find best threshold (EER point)
                    eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
                    eer = (fpr[eer_idx] + (1 - tpr[eer_idx])) / 2
                    best_threshold = thresholds[eer_idx]
                    
                    # Accuracy at best threshold
                    predictions = (scores >= best_threshold).astype(int)
                    accuracy = np.mean(predictions == labels)
                    
                    # TAR at different FARs
                    tar_at_far = {}
                    for target_far in [0.01, 0.1]:
                        idx = np.where(fpr <= target_far)[0]
                        if len(idx) > 0:
                            tar_at_far[f'TAR@FAR={target_far}'] = tpr[idx[-1]]
                    
                    # Similarity score statistics
                    pos_scores = scores[labels == 1]
                    neg_scores = scores[labels == 0]
                    
                    age_group_results[model_name] = {
                        'accuracy': accuracy,
                        'eer': eer,
                        'auc': roc_auc,
                        'tar_at_far': tar_at_far,
                        'pos_score_mean': np.mean(pos_scores),
                        'pos_score_std': np.std(pos_scores),
                        'neg_score_mean': np.mean(neg_scores),
                        'neg_score_std': np.std(neg_scores),
                        'scores': scores,
                        'labels': labels,
                        'fpr': fpr,
                        'tpr': tpr
                    }
                    
                    print(f"    Accuracy: {accuracy:.4f}")
                    print(f"    EER: {eer:.4f}")
                    print(f"    AUC: {roc_auc:.4f}")
                    print(f"    Pos scores: {np.mean(pos_scores):.4f} ± {np.std(pos_scores):.4f}")
                    print(f"    Neg scores: {np.mean(neg_scores):.4f} ± {np.std(neg_scores):.4f}")
                else:
                    print(f"    Error: Could not compute similarity scores")
                    
            except Exception as e:
                print(f"    Error evaluating {model_name}: {str(e)}")
        
        all_age_group_results[age_group] = age_group_results
    
    # Create comparison visualizations
    print(f"\n{'-'*80}")
    print("Creating Age Group Comparison Visualizations")
    print(f"{'-'*80}")
    
    # Create plots comparing metrics across age groups
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    metrics_data = []
    for age_group, results in all_age_group_results.items():
        for model_name, metrics in results.items():
            metrics_data.append({
                'Age Group': age_group,
                'Model': model_name,
                'Accuracy': metrics['accuracy']
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        sns.barplot(data=metrics_df, x='Age Group', y='Accuracy', hue='Model', ax=ax)
        ax.set_title('Accuracy Across Age Groups')
        ax.set_ylim([0, 1])
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Plot 2: EER comparison
    ax = axes[0, 1]
    eer_data = []
    for age_group, results in all_age_group_results.items():
        for model_name, metrics in results.items():
            eer_data.append({
                'Age Group': age_group,
                'Model': model_name,
                'EER': metrics['eer']
            })
    
    if eer_data:
        eer_df = pd.DataFrame(eer_data)
        sns.barplot(data=eer_df, x='Age Group', y='EER', hue='Model', ax=ax)
        ax.set_title('Equal Error Rate Across Age Groups')
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Plot 3: AUC comparison
    ax = axes[0, 2]
    auc_data = []
    for age_group, results in all_age_group_results.items():
        for model_name, metrics in results.items():
            auc_data.append({
                'Age Group': age_group,
                'Model': model_name,
                'AUC': metrics['auc']
            })
    
    if auc_data:
        auc_df = pd.DataFrame(auc_data)
        sns.barplot(data=auc_df, x='Age Group', y='AUC', hue='Model', ax=ax)
        ax.set_title('AUC Across Age Groups')
        ax.set_ylim([0, 1])
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Plot 4: ROC Curves by Age Group
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # Extended color list
    age_groups = list(all_age_group_results.keys())
    
    for i, model_name in enumerate(['ArcFace', 'MagFace']):
        ax = axes[1, i]
        for j, age_group in enumerate(age_groups):
            if age_group in all_age_group_results and model_name in all_age_group_results[age_group]:
                results = all_age_group_results[age_group][model_name]
                fpr = results['fpr']
                tpr = results['tpr']
                auc_score = results['auc']
                ax.plot(fpr, tpr, color=colors[j], 
                       label=f'{age_group} (AUC={auc_score:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(age_analysis_dir, 'age_group_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create similarity score distributions plot
    fig, axes = plt.subplots(len(all_age_group_results), len(['ArcFace', 'MagFace']), 
                           figsize=(10, 5*len(all_age_group_results)))
    
    if len(all_age_group_results) == 1:
        axes = axes.reshape(1, -1)
    
    for i, age_group in enumerate(all_age_group_results.keys()):
        for j, model_name in enumerate(['ArcFace', 'MagFace']):
            if model_name in all_age_group_results[age_group]:
                ax = axes[i, j] if len(all_age_group_results) > 1 else axes[j]
                results = all_age_group_results[age_group][model_name]
                
                scores = results['scores']
                labels = results['labels']
                
                # Plot distributions
                pos_scores = scores[labels == 1]
                neg_scores = scores[labels == 0]
                
                ax.hist(neg_scores, bins=50, alpha=0.6, label='Different Person', color='red', density=True)
                ax.hist(pos_scores, bins=50, alpha=0.6, label='Same Person', color='blue', density=True)
                
                ax.set_xlabel('Similarity Score')
                ax.set_ylabel('Density')
                ax.set_title(f'{model_name} - {age_group}')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(age_analysis_dir, 'similarity_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for age_group, results in all_age_group_results.items():
        json_results[age_group] = {}
        for model_name, metrics in results.items():
            json_results[age_group][model_name] = {
                'accuracy': float(metrics['accuracy']),
                'eer': float(metrics['eer']),
                'auc': float(metrics['auc']),
                'tar_at_far': metrics['tar_at_far'],
                'pos_score_mean': float(metrics['pos_score_mean']),
                'pos_score_std': float(metrics['pos_score_std']),
                'neg_score_mean': float(metrics['neg_score_mean']),
                'neg_score_std': float(metrics['neg_score_std'])
            }
    
    with open(os.path.join(age_analysis_dir, 'age_group_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print summary
    print(f"\n{'-'*80}")
    print("AGE GROUP ANALYSIS SUMMARY")
    print(f"{'-'*80}")
    
    for age_group, results in all_age_group_results.items():
        print(f"\n{age_group}:")
        for model_name, metrics in results.items():
            print(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}, EER={metrics['eer']:.4f}, AUC={metrics['auc']:.4f}")
    
    print(f"\nResults and visualizations saved to: {age_analysis_dir}")
    print(f"Generated files:")
    print(f"  - age_group_comparison.png: Overall comparison metrics")
    print(f"  - similarity_distributions.png: Score distribution plots")
    print(f"  - age_group_results.json: Numerical results")
    print(f"  - [Age_Group]_pairs.csv: Generated pairs for each group")
    
    return all_age_group_results

def main():
    """Main pipeline"""
    parser = argparse.ArgumentParser(description='Age-Invariant Face Recognition Pipeline')
    parser.add_argument('--step', type=str, default='all',
                      choices=['all', 'preprocess', 'pairs', 'train_baseline', 
                              'evaluate', 'test_robustness', 'visualize', 'age_analysis'],
                      help='Which step to run')
    parser.add_argument('--dataset', type=str, default='morph_2',
                      choices=['agedb_30', 'morph_2'],
                      help='Which dataset to use (default: morph_2)')
    parser.add_argument('--backbone', type=str, default='magface',
                      choices=['arcface', 'magface'],
                      help='Backbone model for temporal training (default: magface)')
    parser.add_argument('--eval-backbone', type=str, default=None,
                      choices=['arcface', 'magface'],
                      help='Backbone for temporal model evaluation (default: same as --backbone)')
    parser.add_argument('--eval-dataset', type=str, default=None,
                      choices=['agedb_30', 'morph_2'],
                      help='Dataset to evaluate on (default: same as --dataset)')
    parser.add_argument('--test-dataset', type=str, default=None,
                      choices=['agedb_30', 'morph_2'],
                      help='Dataset to test robustness on (default: same as --dataset)')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of training epochs (default: from config.TRAIN_CONFIG["num_epochs"])')
    
    args = parser.parse_args()
    
    # Set active dataset in config
    config.set_active_dataset(args.dataset)
    
    print("\n" + "="*80)
    print("AGE-INVARIANT FACE RECOGNITION")
    print("Deep Learning Approach")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Backbone: {args.backbone.upper()}")
    print("="*80)
    
    # if args.step == 'all' or args.step == 'preprocess':
    #     metadata_df = step1_data_preprocessing(args.dataset)
    
    if args.step == 'all' or args.step == 'pairs':
        step2_generate_pairs(args.dataset)
    
    if args.step == 'all' or args.step == 'train_baseline':
        step3_train_baseline_models(backbone=args.backbone, epochs=args.epochs)
    
    if args.step == 'all' or args.step == 'evaluate':
        eval_backbone = args.eval_backbone if args.eval_backbone else args.backbone
        eval_dataset = args.eval_dataset if args.eval_dataset else args.dataset
        # Set active dataset for evaluation if different
        if eval_dataset != args.dataset:
            config.set_active_dataset(eval_dataset)
        all_results = step5_evaluate_models(backbone=eval_backbone, evaluate_dataset=eval_dataset)
    
    if args.step == 'all' or args.step == 'test_robustness':
        test_backbone = args.eval_backbone if args.eval_backbone else args.backbone
        test_dataset = args.test_dataset if args.test_dataset else args.dataset
        # Set active dataset for testing if different
        if test_dataset != args.dataset:
            config.set_active_dataset(test_dataset)
        robustness_results = step6_test_robustness(backbone=test_backbone, test_dataset=test_dataset)
    
    if args.step == 'all' or args.step == 'visualize':
        # Load results if not already loaded
        try:
            import json
            results_path = os.path.join(config.OUTPUT_ROOT, "results")
            all_results = {}
            
            for model_file in os.listdir(results_path):
                if model_file.endswith('_evaluation_results.json'):
                    model_name = model_file.replace('_evaluation_results.json', '')
                    with open(os.path.join(results_path, model_file), 'r') as f:
                        all_results[model_name] = json.load(f)
            
            step7_visualization_and_statistics(all_results)
        except Exception as e:
            print(f"Error loading results: {e}")
            step7_visualization_and_statistics(all_results)
        except Exception as e:
            print(f"Error loading results: {e}")
    
    if args.step == 'all' or args.step == 'age_analysis':
        age_backbone = args.eval_backbone if args.eval_backbone else args.backbone
        age_group_results = step8_age_group_analysis(backbone=age_backbone)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
