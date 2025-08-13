import json
import pandas as pd
import os
import glob
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict


def flatten_confusion_matrix(cm: List[List[int]], prefix: str) -> Dict[str, Optional[int]]:
    """
    Flatten confusion matrix into separate columns.
    
    Args:
        cm: 2x2 confusion matrix
        prefix: Prefix for column names
        
    Returns:
        Dictionary with flattened confusion matrix values
    """
    if cm and len(cm) == 2 and len(cm[0]) == 2:
        return {
            f'{prefix}_tn': cm[0][0],
            f'{prefix}_fp': cm[0][1],
            f'{prefix}_fn': cm[1][0],
            f'{prefix}_tp': cm[1][1]
        }
    return {
        f'{prefix}_tn': None,
        f'{prefix}_fp': None,
        f'{prefix}_fn': None,
        f'{prefix}_tp': None
    }


def extract_history_stats(history: Dict[str, List[float]]) -> Dict[str, Optional[float]]:
    """
    Extract key statistics from training history.
    
    Args:
        history: Training history dictionary
        
    Returns:
        Dictionary with history statistics
    """
    stats = {}
    
    # Metrics to extract from history
    metrics = ['loss', 'val_loss', 'f1_score_metric', 'val_f1_score_metric', 'accuracy', 'val_accuracy']
    
    for metric in metrics:
        if metric in history and history[metric]:
            values = history[metric]
            stats[f'history_{metric}_final'] = values[-1] if values else None
            stats[f'history_{metric}_best'] = min(values) if 'loss' in metric else max(values)
            stats[f'history_{metric}_mean'] = np.mean(values) if values else None
            stats[f'history_{metric}_std'] = np.std(values) if values else None
        else:
            stats[f'history_{metric}_final'] = None
            stats[f'history_{metric}_best'] = None
            stats[f'history_{metric}_mean'] = None
            stats[f'history_{metric}_std'] = None
    
    return stats


def get_algorithm_name(file_path: str) -> str:
    """
    Extract algorithm name from the immediate parent folder of the JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Algorithm name (folder name)
    """
    return os.path.basename(os.path.dirname(file_path))


def process_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Process a single JSON result file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary with extracted data or None if processing fails
    """
    # Skip files with 'summary' in path
    if 'summary' in file_path:
        return None
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract basic information
        result = {
            'file_name': os.path.basename(file_path),
            'algorithm': get_algorithm_name(file_path),  # NEW: Algorithm name from folder
            'trial_number': data.get('trial_number'),
            'architecture': data.get('hyperparameters', {}).get('architecture'),
            'learning_rate': data.get('hyperparameters', {}).get('learning_rate'),
            'batch_size': data.get('hyperparameters', {}).get('batch_size'),
            'dropout_rate': data.get('hyperparameters', {}).get('dropout_rate'),
            'optimizer': data.get('hyperparameters', {}).get('optimizer'),
            'epochs': data.get('hyperparameters', {}).get('epochs'),
            'freeze_base': data.get('hyperparameters', {}).get('freeze_base'),
        }
        
        # Handle optimized parameters/factors
        hyperparams = data.get('hyperparameters', {})
        if 'optimized_parameter' in hyperparams:
            result['optimized_parameter'] = hyperparams['optimized_parameter']
            result['optimization_type'] = 'single_factor'
            result['num_optimized_factors'] = 1
            result['optimized_factors'] = hyperparams['optimized_parameter']
        elif 'optimized_factors' in hyperparams:
            factors = hyperparams['optimized_factors']
            result['optimized_parameter'] = '-'.join(factors) if factors else 'unknown'
            result['optimization_type'] = 'multi_factor'
            result['num_optimized_factors'] = len(factors) if factors else 0
            result['optimized_factors'] = ', '.join(factors) if factors else 'unknown'
        else:
            print(f"\nWarning: No optimized parameters found in {file_path}. Defaulting to 'unknown'.")
            result['optimized_parameter'] = 'unknown'
            result['optimization_type'] = 'unknown'
            result['num_optimized_factors'] = 0
            result['optimized_factors'] = ''
        
        # Extract metrics for train, validation, and test sets
        metric_types = ['train', 'val', 'test']
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        for metric_type in metric_types:
            metrics_data = data.get(f'{metric_type}_metrics', {})
            for metric_name in metric_names:
                result[f'{metric_type}_{metric_name}'] = metrics_data.get(metric_name)
            
            # Extract confusion matrices
            result.update(flatten_confusion_matrix(
                metrics_data.get('confusion_matrix'), f'{metric_type}_cm'
            ))
        
        # Extract additional performance metrics
        performance_metrics = [
            'min_train_loss', 'min_val_loss', 'max_train_f1', 'max_val_f1',
            'max_train_accuracy', 'max_val_accuracy'
        ]
        for metric in performance_metrics:
            result[metric] = data.get(metric)
        
        # Extract training information
        result['training_time'] = data.get('training_time')
        result['final_epoch'] = data.get('final_epoch')
        
        # Extract experiment setup
        experiment_setup = data.get('experiment_setup', {})
        result['input_shape'] = str(experiment_setup.get('input_shape', ''))
        result['num_classes'] = experiment_setup.get('num_classes')
        
        # Extract history statistics
        if 'history' in data:
            result.update(extract_history_stats(data['history']))
        
        return result
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def get_all_json_files(base_dir: str) -> List[str]:
    """Get all JSON files recursively from base directory."""
    return glob.glob(os.path.join(base_dir, "**", "*.json"), recursive=True)


def get_immediate_folder(base_dir: str, file_path: str) -> str:
    """Get the immediate folder name relative to base directory."""
    rel_path = os.path.relpath(file_path, base_dir)
    parts = rel_path.split(os.sep)
    return parts[0] if len(parts) > 1 else "(root)"


def count_jsons_per_folder(json_files: List[str]) -> Dict[str, int]:
    """Count JSON files per folder (excluding summary files)."""
    json_files = [f for f in json_files if 'summary' not in f]
    folder_count = defaultdict(int)
    for path in json_files:
        folder = os.path.dirname(path)
        folder_count[folder] += 1
    return folder_count


def analyze_json_structure(base_dir: str) -> List[Dict[str, Any]]:
    """Analyze the structure of JSON files in the directory."""
    json_files = get_all_json_files(base_dir)
    folder_counts = count_jsons_per_folder(json_files)

    result = []
    for path in json_files:
        info = {
            "file": os.path.abspath(path),
            "immediate_folder": get_immediate_folder(base_dir, path),
            "json_count_in_last_folder": folder_counts[os.path.dirname(path)]
        }
        result.append(info)
    return result


def combine_experiment_results(results_directory: str, output_file: str = 'combined_results.csv') -> Optional[pd.DataFrame]:
    """
    Combine all JSON experiment results into a single CSV file.
    
    Args:
        results_directory: Path to directory containing JSON result files
        output_file: Name of output CSV file
        
    Returns:
        Combined DataFrame or None if no valid results
    """
    json_info = analyze_json_structure(results_directory)

    if not json_info:
        print(f"No JSON files found in {results_directory}")
        return None
    
    print(f"Found {len(json_info)} JSON files to process...")
    
    # Process all files
    all_results = []
    failed_files = []
    
    for i, entry in enumerate(json_info):
        file_path = entry['file']
        result = process_json_file(file_path)
        
        # Debug output for specific cases
        json_count = entry['json_count_in_last_folder']
        if result and result.get('num_optimized_factors') == 2:
            if json_count not in [10, 25]:
                print(f"\nProcessing file {i+1}/{len(json_info)}")
                print(f"  Folder: {entry['immediate_folder']}")
                print(f"  JSONs in same folder: {json_count}")
                print(f"  Algorithm: {result.get('algorithm')}")

        if result and result.get('optimized_parameter') == 'EPOCHS-DROPOUT_RATE':
            if json_count != 25:
                print(f"\nProcessing file {i+1}/{len(json_info)}")
                print(f"  File: {file_path}")
                print(f"  Folder: {entry['immediate_folder']}")
                print(f"  JSONs in same folder: {json_count}")
                print(f"  Algorithm: {result.get('algorithm')}")
                
        if result:
            all_results.append(result)
        else:
            failed_files.append(file_path)
    
    # Report failed files
    if not failed_files: # negate to ensure it doesn't run
        print(f"\nFailed to process {len(failed_files)} files:")
        for file_path in failed_files[:10]:  # Show first 10 failed files
            print(f"  - {file_path}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    # Create DataFrame and save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        print(f"\nSuccessfully combined {len(all_results)} experiments!")
        print(f"Results saved to: {output_file}")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Print summary statistics
        print("\n--- SUMMARY STATISTICS ---")
        print(f"Algorithms: {df['algorithm'].value_counts().to_dict()}")
        print(f"Architectures: {df['architecture'].value_counts().to_dict()}")
        print(f"Optimization types: {df['optimization_type'].value_counts().to_dict()}")
        print(f"Number of optimized factors: {df['num_optimized_factors'].value_counts().to_dict()}")
        
        # Performance summary
        if 'test_accuracy' in df.columns and df['test_accuracy'].notna().any():
            print(f"\nBest test accuracy: {df['test_accuracy'].max():.4f}")
            print(f"Mean test accuracy: {df['test_accuracy'].mean():.4f}")
        
        if 'test_f1_score' in df.columns and df['test_f1_score'].notna().any():
            print(f"Best test F1: {df['test_f1_score'].max():.4f}")
            print(f"Mean test F1: {df['test_f1_score'].mean():.4f}")
        
        return df
    else:
        print("No valid results to combine!")
        return None


def analyze_results(df: pd.DataFrame) -> None:
    """Perform additional analysis on the combined results."""
    
    print("\n--- DETAILED ANALYSIS ---")
    
    # Best performing experiments
    if 'test_accuracy' in df.columns and df['test_accuracy'].notna().any():
        print("\nTop 10 experiments by test accuracy:")
        top_acc = df.nlargest(10, 'test_accuracy')[
            ['trial_number', 'algorithm', 'architecture', 'optimization_type', 
             'optimized_factors', 'test_accuracy', 'test_f1_score']
        ]
        print(top_acc.to_string(index=False))
    
    if 'test_f1_score' in df.columns and df['test_f1_score'].notna().any():
        print("\nTop 10 experiments by test F1 score:")
        top_f1 = df.nlargest(10, 'test_f1_score')[
            ['trial_number', 'algorithm', 'architecture', 'optimization_type', 
             'optimized_factors', 'test_accuracy', 'test_f1_score']
        ]
        print(top_f1.to_string(index=False))
    
    # Performance by algorithm
    print("\nPerformance by algorithm:")
    algo_perf = df.groupby('algorithm').agg({
        'test_accuracy': ['count', 'mean', 'std', 'max'],
        'test_f1_score': ['count', 'mean', 'std', 'max'],
        'training_time': ['mean', 'std']
    }).round(4)
    print(algo_perf)
    
    # Performance by architecture
    print("\nPerformance by architecture:")
    arch_perf = df.groupby('architecture').agg({
        'test_accuracy': ['mean', 'std', 'max'],
        'test_f1_score': ['mean', 'std', 'max'],
        'training_time': ['mean', 'std']
    }).round(4)
    print(arch_perf)
    
    # Performance by optimization type
    print("\nPerformance by optimization strategy:")
    opt_perf = df.groupby('optimization_type').agg({
        'test_accuracy': ['count', 'mean', 'std', 'max', 'min', 'median'],
        'test_f1_score': ['count', 'mean', 'std', 'max', 'min', 'median'],
        'training_time': ['mean', 'std', 'max', 'min']
    }).round(4)
    print(opt_perf)


def save_summary_file(df: pd.DataFrame, output_file: str) -> None:
    """Save a summary text file with key statistics."""
    summary_file = output_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Experiment Results Summary\n")
        f.write("========================\n\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Total columns: {len(df.columns)}\n")
        f.write(f"Algorithms: {df['algorithm'].value_counts().to_dict()}\n")
        f.write(f"Architectures: {df['architecture'].value_counts().to_dict()}\n")
        f.write(f"Optimization types: {df['optimization_type'].value_counts().to_dict()}\n")
        f.write(f"Optimization Factors: {df['optimized_parameter'].value_counts().to_dict()}\n")
        
        if 'test_accuracy' in df.columns and df['test_accuracy'].notna().any():
            f.write(f"Best test accuracy: {df['test_accuracy'].max():.4f}\n")
            f.write(f"Mean test accuracy: {df['test_accuracy'].mean():.4f}\n")
        
        if 'test_f1_score' in df.columns and df['test_f1_score'].notna().any():
            f.write(f"Best test F1: {df['test_f1_score'].max():.4f}\n")
            f.write(f"Mean test F1: {df['test_f1_score'].mean():.4f}\n")
    
    print(f"Summary saved to: {summary_file}")


def main():
    """Main execution function."""
    # Configuration
    RESULTS_DIR = "./ALL_RESULTS"
    OUTPUT_FILE = "combined_results.csv"
    
    # Check if directory exists
    if not os.path.exists(RESULTS_DIR):
        print(f"Directory {RESULTS_DIR} does not exist!")
        print("Please update RESULTS_DIR variable with the correct path to your results folder.")
        return
    
    # Combine results
    df = combine_experiment_results(RESULTS_DIR, OUTPUT_FILE)
    
    # Perform analysis if successful
    if df is not None:
        analyze_results(df)
        save_summary_file(df, OUTPUT_FILE)


if __name__ == "__main__":
    main()