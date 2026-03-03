"""
Evaluation script for testing model stability under perturbed conditions.
Tests performance with varying levels of Gaussian noise and dropout.
author: Auto-generated for perturbation testing
"""
import torch_geometric as pyg
from torch import nn
import pandas as pd
import math
import os
import json
import torch
from lightning import Trainer, seed_everything
import torch.nn as nn
import numpy as np
from pandas import read_csv
from torch import Tensor
from torch.utils.data import DataLoader, ConcatDataset
import wandb
from torch_geometric.utils import negative_sampling
import src.datamodules.grn_dataset_test_perturb as dt
from argparse import ArgumentParser
from typing import Optional, Tuple, List, Dict
from torch_geometric.loader import DataListLoader
from src.models.grnformer.model_single_out import GRNFormerLitModule
import psutil
import time
from contextlib import contextmanager
from datetime import datetime

def get_memory_usage():
    """Get current memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024  # MB
    else:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

def get_peak_memory_usage():
    """Get peak memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        return get_memory_usage()

@contextmanager
def memory_monitor(operation_name):
    """Context manager to monitor memory usage for an operation"""
    start_memory = get_memory_usage()
    start_time = time.time()
    
    print(f"Starting {operation_name} - Memory: {start_memory:.2f} MB")
    
    try:
        yield
    finally:
        end_memory = get_memory_usage()
        peak_memory = get_peak_memory_usage()
        end_time = time.time()
        
        print(f"Completed {operation_name}")
        print(f"  - Time: {end_time - start_time:.2f} seconds")
        print(f"  - Memory before: {start_memory:.2f} MB")
        print(f"  - Memory after: {end_memory:.2f} MB")
        print(f"  - Memory delta: {end_memory - start_memory:.2f} MB")
        print(f"  - Peak memory: {peak_memory:.2f} MB")
        
        # Reset peak memory counter
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

AVAIL_GPUS = [0]
NUM_NODES = 1
BATCH_SIZE = 1
DATALOADERS = 1
ACCELERATOR = "gpu"

DATASET_DIR = os.path.abspath("./")
EPS = 1e-15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_evaluation_with_perturbation(
    exp_file: str,
    tf_file: str,
    net_file: str,
    output_file: str,
    ckpt_path: str,
    noise_std: float = 0.0,
    dropout_fraction: float = 0.0,
    random_seed: int = None,
    coexpression_threshold: float = 0.1,
    max_subgraph_size: int = 100,
    scale_noise: bool = True
) -> Dict:
    """
    Run evaluation with specified perturbation parameters.
    
    Returns:
        Dictionary containing evaluation results and metadata
    """
    print(f"\n{'='*80}")
    print(f"Running evaluation with perturbations:")
    print(f"  - Noise std: {noise_std}")
    print(f"  - Dropout fraction: {dropout_fraction}")
    print(f"  - Random seed: {random_seed}")
    print(f"{'='*80}\n")
    
    # Set seed if provided
    if random_seed is not None:
        seed_everything(random_seed)
    
    results = {
        'noise_std': noise_std,
        'dropout_fraction': dropout_fraction,
        'random_seed': random_seed,
        'scale_noise': scale_noise,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Load data with perturbations
        with memory_monitor(f"Data Loading (noise={noise_std}, dropout={dropout_fraction})"):
            root = [os.path.dirname(os.path.abspath(exp_file))]
            gene_expression_file = [os.path.abspath(exp_file)]
            numnodes = len(pd.read_csv(os.path.abspath(exp_file), index_col=0))
            print(f"Number of nodes: {numnodes}")
            
            tffile = os.path.abspath(tf_file)
            tf = pd.read_csv(tffile, header=None)[0].to_list()
            TF_list = [tf]
            regulation_file = [os.path.abspath(net_file)]
            
            All_test_dataset = []
            for i in range(len(root)):
                dataset = dt.GeneExpressionDataset(
                    root[i],
                    gene_expression_file[i],
                    TF_list[i],
                    regulation_file[i],
                    coexpression_threshold=coexpression_threshold,
                    max_subgraph_size=max_subgraph_size,
                    noise_std=noise_std,
                    dropout_fraction=dropout_fraction,
                    random_seed=random_seed,
                    scale_noise=scale_noise
                )
                All_test_dataset.append(dataset)
            
            TestDatasets = ConcatDataset(All_test_dataset)
            test_loader = DataListLoader(
                dataset=TestDatasets,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=1
            )
        
        # Create unique output file for this perturbation
        base_output = os.path.abspath(output_file)
        base_name, ext = os.path.splitext(base_output)
        # Add perturbation info to output filename
        perturb_suffix = ""
        if noise_std > 0.0 or dropout_fraction > 0.0:
            noise_str = f"{noise_std:.3f}".replace('.', 'p')
            dropout_str = f"{dropout_fraction:.3f}".replace('.', 'p')
            perturb_suffix = f"_noise{noise_str}_dropout{dropout_str}"
        unique_output_file = base_name + perturb_suffix + ext
        
        # Create model
        with memory_monitor("Model Creation"):
            model = GRNFormerLitModule(
                totalnodes=numnodes,
                tf_file=tffile,
                exp_file=os.path.abspath(exp_file),
                net_file=os.path.abspath(net_file),
                output_file=unique_output_file
            )
            print("Model loaded")
        
        # Run inference
        with memory_monitor("Model Inference"):
            trainer = Trainer(
                devices=[0],
                num_nodes=1,
                accelerator=ACCELERATOR,
                detect_anomaly=True,
                enable_model_summary=True
            )
            
            # Run test and capture results
            test_results = trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt_path)
            
            # Store test results
            if test_results and len(test_results) > 0:
                results['test_metrics'] = test_results[0]
            else:
                results['test_metrics'] = {}
            
            # Read metrics from CSV file saved by the model
            metrics_file = unique_output_file.rsplit('.', 1)[0] + "-metrics.csv"
            if os.path.exists(metrics_file):
                try:
                    metrics_df = pd.read_csv(metrics_file)
                    # Extract AUROC and AUPRC
                    for _, row in metrics_df.iterrows():
                        metric_name = str(row['Metric'])
                        metric_value = float(row['Value'])
                        
                        # Map metric names to standard names
                        # Handle BinaryAUROC -> AUROC
                        if 'AUROC' in metric_name or 'auroc' in metric_name.lower():
                            results['AUROC'] = metric_value
                            results['test_metrics']['AUROC'] = metric_value
                        # Handle BinaryAveragePrecision or AUPRC -> AUPRC
                        elif 'AveragePrecision' in metric_name or 'aupr' in metric_name.lower() or 'AUPRC' in metric_name:
                            results['AUPRC'] = metric_value
                            results['test_metrics']['AUPRC'] = metric_value
                        # Also store other metrics with original names
                        results['test_metrics'][metric_name] = metric_value
                    
                    print(f"Loaded metrics from {metrics_file}")
                    if 'AUROC' in results:
                        print(f"  AUROC: {results['AUROC']:.4f}")
                    if 'AUPRC' in results:
                        print(f"  AUPRC: {results['AUPRC']:.4f}")
                except Exception as e:
                    print(f"Warning: Could not read metrics file {metrics_file}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: Metrics file not found at {metrics_file}")
        
        results['status'] = 'success'
        results['memory_peak_mb'] = get_peak_memory_usage()
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        results['status'] = 'error'
        results['error'] = str(e)
    
    return results


def run_perturbation_sweep(
    exp_file: str,
    tf_file: str,
    net_file: str,
    output_file: str,
    ckpt_path: str,
    noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    dropout_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    random_seed: int = 42,
    coexpression_threshold: float = 0.1,
    max_subgraph_size: int = 100,
    output_dir: str = None,
    scale_noise: bool = True
) -> Dict:
    """
    Run evaluation across multiple perturbation levels.
    
    Args:
        noise_levels: List of Gaussian noise standard deviations to test
        dropout_levels: List of dropout fractions to test
        output_dir: Directory to save results. If None, creates timestamped directory.
    
    Returns:
        Dictionary containing all results
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(exp_file), f"perturbation_results_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {
        'config': {
            'exp_file': exp_file,
            'tf_file': tf_file,
            'net_file': net_file,
            'ckpt_path': ckpt_path,
            'noise_levels': noise_levels,
            'dropout_levels': dropout_levels,
            'random_seed': random_seed,
            'coexpression_threshold': coexpression_threshold,
            'max_subgraph_size': max_subgraph_size,
            'scale_noise': scale_noise
        },
        'results': []
    }
    
    # Test baseline (no perturbations)
    print("\n" + "="*80)
    print("BASELINE EVALUATION (No Perturbations)")
    print("="*80)
    baseline_results = run_evaluation_with_perturbation(
        exp_file=exp_file,
        tf_file=tf_file,
        net_file=net_file,
        output_file=output_file,
        ckpt_path=ckpt_path,
        noise_std=0.0,
        dropout_fraction=0.0,
        random_seed=random_seed,
        coexpression_threshold=coexpression_threshold,
        max_subgraph_size=max_subgraph_size,
        scale_noise=scale_noise
    )
    all_results['baseline'] = baseline_results
    all_results['results'].append(baseline_results)
    
    # Test with varying noise levels (no dropout)
    print("\n" + "="*80)
    print("TESTING WITH GAUSSIAN NOISE (No Dropout)")
    print("="*80)
    for noise_std in noise_levels:
        if noise_std == 0.0:
            continue  # Already tested as baseline
        
        result = run_evaluation_with_perturbation(
            exp_file=exp_file,
            tf_file=tf_file,
            net_file=net_file,
            output_file=output_file,
            ckpt_path=ckpt_path,
            noise_std=noise_std,
            dropout_fraction=0.0,
            random_seed=random_seed,
            coexpression_threshold=coexpression_threshold,
            max_subgraph_size=max_subgraph_size,
            scale_noise=scale_noise
        )
        all_results['results'].append(result)
    
    # Test with varying dropout levels (no noise)
    print("\n" + "="*80)
    print("TESTING WITH DROPOUT (No Noise)")
    print("="*80)
    for dropout_fraction in dropout_levels:
        if dropout_fraction == 0.0:
            continue  # Already tested as baseline
        
        result = run_evaluation_with_perturbation(
            exp_file=exp_file,
            tf_file=tf_file,
            net_file=net_file,
            output_file=output_file,
            ckpt_path=ckpt_path,
            noise_std=0.0,
            dropout_fraction=dropout_fraction,
            random_seed=random_seed,
            coexpression_threshold=coexpression_threshold,
            max_subgraph_size=max_subgraph_size,
            scale_noise=scale_noise
        )
        all_results['results'].append(result)
    
    # Test combined perturbations (moderate levels)
    print("\n" + "="*80)
    print("TESTING WITH COMBINED PERTURBATIONS")
    print("="*80)
    combined_configs = [
        (0.1, 0.1),
        (0.2, 0.1),
        (0.1, 0.2),
        (0.2, 0.2),
    ]
    
    for noise_std, dropout_fraction in combined_configs:
        result = run_evaluation_with_perturbation(
            exp_file=exp_file,
            tf_file=tf_file,
            net_file=net_file,
            output_file=output_file,
            ckpt_path=ckpt_path,
            noise_std=noise_std,
            dropout_fraction=dropout_fraction,
            random_seed=random_seed,
            coexpression_threshold=coexpression_threshold,
            max_subgraph_size=max_subgraph_size,
            scale_noise=scale_noise
        )
        all_results['results'].append(result)
    
    # Save results
    results_file = os.path.join(output_dir, "perturbation_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary CSV
    summary_data = []
    for result in all_results['results']:
        row = {
            'noise_std': result.get('noise_std', 0.0),
            'dropout_fraction': result.get('dropout_fraction', 0.0),
            'status': result.get('status', 'unknown'),
            'timestamp': result.get('timestamp', ''),
            'memory_peak_mb': result.get('memory_peak_mb', 0.0)
        }
        
        # Extract AUROC and AUPRC (prioritize direct values, then from test_metrics)
        # Try multiple possible key names
        auroc_value = result.get('AUROC')
        if auroc_value is None:
            test_metrics = result.get('test_metrics', {})
            auroc_value = test_metrics.get('AUROC') or test_metrics.get('BinaryAUROC') or test_metrics.get('test_AUROC')
        
        auprc_value = result.get('AUPRC')
        if auprc_value is None:
            test_metrics = result.get('test_metrics', {})
            auprc_value = test_metrics.get('AUPRC') or test_metrics.get('BinaryAveragePrecision') or test_metrics.get('test_AUPRC') or test_metrics.get('Average Precision')
        
        row['AUROC'] = auroc_value if auroc_value is not None else np.nan
        row['AUPRC'] = auprc_value if auprc_value is not None else np.nan
        
        # Add other test metrics if available
        test_metrics = result.get('test_metrics', {})
        for key, value in test_metrics.items():
            # Skip AUROC and AUPRC as they're already added above
            if key not in ['AUROC', 'AUPRC', 'BinaryAUROC', 'BinaryAveragePrecision']:
                row[f'metric_{key}'] = value
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "perturbation_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"Perturbation evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"  - Full results: {results_file}")
    print(f"  - Summary CSV: {summary_file}")
    print(f"{'='*80}\n")
    
    # Print summary
    print("\nSUMMARY OF RESULTS:")
    print(summary_df.to_string())
    
    return all_results


if __name__ == "__main__":
    seed_everything(123)
    parser = ArgumentParser(description="Evaluate model stability under perturbations")
    
    parser.add_argument('--exp_file', type=str, required=True,
                        help="Expression file path (e.g., 'Data/sc-RNA-seq/hESC/hESC_nonspecific_chipseq_500-ExpressionData.csv')")
    parser.add_argument('--tf_file', type=str, required=True,
                        help="TF file path (single column CSV file)")
    parser.add_argument('--net_file', type=str, required=True,
                        help="Ground truth network file path")
    parser.add_argument('--output_file', type=str, required=True,
                        help="Output file path for predictions")
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument('--noise_levels', type=float, nargs='+', 
                        default=[0.0, 0.1, 0.2, 0.3, 0.5],
                        help="Gaussian noise standard deviations to test")
    parser.add_argument('--dropout_levels', type=float, nargs='+',
                        default=[0.0, 0.1, 0.2, 0.3, 0.5],
                        help="Dropout fractions to test (0.0 to 1.0)")
    parser.add_argument('--random_seed', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--coexpression_threshold', type=float, default=0.1,
                        help="Coexpression threshold for graph construction")
    parser.add_argument('--max_subgraph_size', type=int, default=100,
                        help="Maximum subgraph size")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory to save results (default: timestamped directory)")
    parser.add_argument('--single_test', action='store_true',
                        help="Run single test with specified noise_std and dropout_fraction")
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help="Single test: Gaussian noise standard deviation")
    parser.add_argument('--dropout_fraction', type=float, default=0.0,
                        help="Single test: Dropout fraction")
    parser.add_argument('--absolute_noise', action='store_true',
                        help="Use absolute noise values instead of scaled (default: scaled relative to data std)")
    
    args = parser.parse_args()
    
    # Determine noise scaling mode (default: scaled, unless --absolute_noise is specified)
    scale_noise = not args.absolute_noise
    
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
    print(f"Noise scaling mode: {'SCALED (relative to data std)' if scale_noise else 'ABSOLUTE'}")
    
    if args.single_test:
        # Run single evaluation
        result = run_evaluation_with_perturbation(
            exp_file=args.exp_file,
            tf_file=args.tf_file,
            net_file=args.net_file,
            output_file=args.output_file,
            ckpt_path=args.ckpt_path,
            noise_std=args.noise_std,
            dropout_fraction=args.dropout_fraction,
            random_seed=args.random_seed,
            coexpression_threshold=args.coexpression_threshold,
            max_subgraph_size=args.max_subgraph_size,
            scale_noise=scale_noise
        )
        print("\nResult:", json.dumps(result, indent=2))
    else:
        # Run full perturbation sweep
        all_results = run_perturbation_sweep(
            exp_file=args.exp_file,
            tf_file=args.tf_file,
            net_file=args.net_file,
            output_file=args.output_file,
            ckpt_path=args.ckpt_path,
            noise_levels=args.noise_levels,
            dropout_levels=args.dropout_levels,
            random_seed=args.random_seed,
            coexpression_threshold=args.coexpression_threshold,
            max_subgraph_size=args.max_subgraph_size,
            output_dir=args.output_dir,
            scale_noise=scale_noise
        )
    
    print(f"Final memory usage: {get_memory_usage():.2f} MB")
    print(f"Peak memory usage: {get_peak_memory_usage():.2f} MB")

