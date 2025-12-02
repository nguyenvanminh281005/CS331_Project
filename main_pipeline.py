"""
Main pipeline for Age-Invariant Face Recognition
"""
import os
import torch
import argparse
import pandas as pd

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
        
        # Generate temporal pairs
        processor = MORPH2Processor()
        pairs_df = processor.generate_temporal_pairs(
            metadata_df,
            output_path=os.path.join(output_dir, "morph2_temporal_pairs.csv"),
            time_gaps=config.TEMPORAL_CONFIG["time_gaps"],
            pairs_per_gap=1000
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

def step3_train_baseline_models(backbone='magface'):
    """
    Step 3: Train Baseline Models
    - Train specified backbone model (ArcFace or MagFace)
    - Evaluate on standard benchmarks
    
    Args:
        backbone: 'arcface' or 'magface' - which model to train
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
    print(f"\n--- Training {backbone.upper()} ---")
    trainer = Trainer(model_type=backbone, num_classes=num_classes, use_temporal=False)
    trainer.setup_dataloader(train_df, val_df)
    trainer.train(num_epochs=config.TRAIN_CONFIG["num_epochs"])
    
    print(f"\n{backbone.upper()} baseline training complete!")

def step4_train_temporal_model(backbone='magface'):
    """
    Step 4: Train Temporal-Aware Model
    - Implement Temporal Contrastive Loss
    - Train with temporal triplets
    
    Args:
        backbone: 'arcface' or 'magface' - which model to use as backbone
    """
    print("\n" + "="*80)
    print(f"STEP 4: TRAIN TEMPORAL-AWARE MODEL (Backbone: {backbone.upper()})")
    print("="*80)
    
    # Determine pairs file based on active dataset
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
    print(f"Loaded {len(pairs_df)} pairs")
    
    # Check if pairs already have identity column (MORPH-2 case)
    if 'identity' not in pairs_df.columns:
        # Load metadata to get identity information (AgeDB-30 case)
        metadata_path = os.path.join(config.OUTPUT_ROOT, "processed_images", "metadata.csv")
        if not os.path.exists(metadata_path):
            print(f"Error: Metadata not found: {metadata_path}")
            return
        
        metadata_df = pd.read_csv(metadata_path)
        
        # Create path to identity mapping
        path_to_identity = dict(zip(metadata_df['aligned_path'], metadata_df['identity']))
        
        # Rename columns to match TemporalTripletDataset expectations
        if 'img1_path' in pairs_df.columns:
            pairs_df = pairs_df.rename(columns={
                'img1_path': 'enrollment_path',
                'img2_path': 'probe_path'
            })
        
        # Add actual identity column from metadata
        pairs_df['identity'] = pairs_df['enrollment_path'].map(path_to_identity)
        
        # Remove pairs where identity couldn't be found
        pairs_df = pairs_df.dropna(subset=['identity'])
        pairs_df['identity'] = pairs_df['identity'].astype(int)
        print(f"After adding identity: {len(pairs_df)} pairs")
    
    # Load train metadata to get number of classes
    train_df = pd.read_csv(os.path.join(config.OUTPUT_ROOT, "processed_images", "train_metadata.csv"))
    
    # Get number of classes from train data
    num_classes = train_df['identity'].nunique()
    print(f"Number of classes: {num_classes}")
    
    # Create mapping from original identities to 0-based class indices
    if 'identity' in pairs_df.columns:
        # Get unique identities in pairs
        unique_identities = sorted(pairs_df['identity'].unique())
        
        # Create mapping: original identity -> 0-based index
        identity_to_idx = {orig_id: idx for idx, orig_id in enumerate(unique_identities)}
        
        # Remap identities in pairs_df
        pairs_df['identity'] = pairs_df['identity'].map(identity_to_idx)
        
        print(f"Final pairs count: {len(pairs_df)}")
        print(f"Identity range in pairs: {pairs_df['identity'].min()} - {pairs_df['identity'].max()}")
        print(f"Unique identities in pairs: {pairs_df['identity'].nunique()}")
        
        # Update num_classes to match the pairs dataset
        num_classes = pairs_df['identity'].nunique()
        print(f"Updated num_classes to: {num_classes}")
    
    print(f"\n--- Training Temporal-Aware Model (Backbone: {backbone.upper()}) ---")
    temporal_trainer = Trainer(
        model_type='temporal', 
        num_classes=num_classes, 
        use_temporal=True,
        backbone=backbone  # Pass backbone choice to Trainer
    )
    temporal_trainer.setup_dataloader(pairs_df, use_temporal_triplets=True)
    temporal_trainer.train(num_epochs=config.TRAIN_CONFIG["num_epochs"])
    
    print("\nTemporal-aware model training complete!")

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
        },
        'Temporal-Aware': {
            'type': 'temporal',
            'checkpoint': os.path.join(config.MODEL_ROOT, 'temporal_best_model.pth')
        }
    }
    
    all_results = {}
    
    for model_name, model_info in models_to_evaluate.items():
        print(f"\n--- Evaluating {model_name} ---")
        
        if not os.path.exists(model_info['checkpoint']):
            print(f"Checkpoint not found: {model_info['checkpoint']}")
            continue
        
        # Pass backbone for temporal model
        if model_info['type'] == 'temporal':
            evaluator = Evaluator(
                model_type=model_info['type'],
                checkpoint_path=model_info['checkpoint'],
                backbone=backbone
            )
        else:
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
        },
    }
    
    # Optionally add temporal model if trained
    temporal_checkpoint = os.path.join(config.MODEL_ROOT, 'temporal_best_model.pth')
    if os.path.exists(temporal_checkpoint):
        models_to_test['Temporal-Aware'] = {
            'type': 'temporal',
            'checkpoint': temporal_checkpoint,
            'backbone': backbone
        }
    
    all_robustness_results = {}
    
    for model_name, model_info in models_to_test.items():
        print(f"\n{'='*80}")
        print(f"Testing {model_name} Robustness")
        print(f"{'='*80}")
        
        if not os.path.exists(model_info['checkpoint']):
            print(f"Checkpoint not found: {model_info['checkpoint']}")
            continue
        
        # Create evaluator
        if model_info['type'] == 'temporal':
            evaluator = RobustnessEvaluator(
                model_type=model_info['type'],
                checkpoint_path=model_info['checkpoint'],
                backbone=model_info['backbone']
            )
        else:
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

def main():
    """Main pipeline"""
    parser = argparse.ArgumentParser(description='Age-Invariant Face Recognition Pipeline')
    parser.add_argument('--step', type=str, default='all',
                      choices=['all', 'preprocess', 'pairs', 'train_baseline', 
                              'train_temporal', 'evaluate', 'test_robustness', 'visualize'],
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
    
    args = parser.parse_args()
    
    # Set active dataset in config
    config.set_active_dataset(args.dataset)
    
    print("\n" + "="*80)
    print("AGE-INVARIANT FACE RECOGNITION")
    print("Temporal-Aware Deep Learning Approach")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Backbone: {args.backbone.upper()}")
    print("="*80)
    
    # if args.step == 'all' or args.step == 'preprocess':
    #     metadata_df = step1_data_preprocessing(args.dataset)
    
    # if args.step == 'all' or args.step == 'pairs':
    #     step2_generate_pairs(args.dataset)
    
    if args.step == 'all' or args.step == 'train_baseline':
        step3_train_baseline_models(backbone=args.backbone)
    
    # if args.step == 'all' or args.step == 'train_temporal':
    #     step4_train_temporal_model(backbone=args.backbone)
    
    if args.step == 'all' or args.step == 'evaluate':
        eval_backbone = args.eval_backbone if args.eval_backbone else args.backbone
        eval_dataset = args.eval_dataset if args.eval_dataset else args.dataset
        # Set active dataset for evaluation if different
        if eval_dataset != args.dataset:
            config.set_active_dataset(eval_dataset)
        all_results = step5_evaluate_models(backbone=eval_backbone, evaluate_dataset=eval_dataset)
    
    # if args.step == 'all' or args.step == 'test_robustness':
    #     test_backbone = args.eval_backbone if args.eval_backbone else args.backbone
    #     test_dataset = args.test_dataset if args.test_dataset else args.dataset
    #     # Set active dataset for testing if different
    #     if test_dataset != args.dataset:
    #         config.set_active_dataset(test_dataset)
    #     robustness_results = step6_test_robustness(backbone=test_backbone, test_dataset=test_dataset)
    
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
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
