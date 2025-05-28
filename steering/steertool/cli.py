#!/usr/bin/env python

import argparse
import sys
import torch
import logging
from .dataset_generator import generate_datasets
from .visualizer import visualize_dataset, visualize_all_datasets
from .experiments import run_steering_experiment, run_separability_analysis

def main():
    """Main entry point for the steertool CLI"""
    parser = argparse.ArgumentParser(description="Steertool CLI for time series analysis and manipulation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Dataset generation command
    dataset_parser = subparsers.add_parser("generate", help="Generate time series datasets")
    dataset_parser.add_argument("--config_dir", type=str, required=True, 
                              help="Directory containing config files")
    dataset_parser.add_argument("--seed", type=int, default=42,
                              help="Random seed for reproducibility")
    dataset_parser.add_argument("--only-default", action="store_true",
                            help="Generate only from the default config.yaml file")
    dataset_parser.add_argument("--visualize", action="store_true",
                            help="Visualize datasets after generation")
    dataset_parser.add_argument("--output-dir", type=str, default="data_visualizations",
                            help="Directory to save visualizations")
    
    # Dataset visualization command (renamed)
    viz_parser = subparsers.add_parser("visualize-data", help="Visualize time series datasets")
    viz_parser.add_argument("--dataset", type=str,
                          help="Path to a specific dataset file to visualize")
    viz_parser.add_argument("--dataset-dir", type=str, default="datasets",
                          help="Directory containing dataset files (.parquet)")
    viz_parser.add_argument("--output-dir", type=str, default="data_visualizations",
                          help="Directory to save visualizations")
    viz_parser.add_argument("--samples", type=int, default=3,
                          help="Number of random samples to plot")
    viz_parser.add_argument("--skip-ecg", action="store_true",
                          help="Skip ECG dataset files")
    
    # Steering command
    steer_parser = subparsers.add_parser("steer", help="Run steering experiments on time series data")
    steer_parser.add_argument("--source-dataset", type=str, required=True,
                            help="Path to the source dataset (parquet)")
    steer_parser.add_argument("--target-dataset", type=str, required=True,
                            help="Path to the target dataset (parquet)")
    steer_parser.add_argument("--input-sample", type=str, required=True,
                            help="Path to the dataset containing the sample to steer")
    steer_parser.add_argument("--input-sample-index", type=int, default=0,
                            help="Index of the sample to steer")
    steer_parser.add_argument("--model", type=str, choices=["moment", "chronos"], default="moment",
                            help="Model to use for steering")
    steer_parser.add_argument("--method", type=str, choices=["mean", "median", "lda"], default="mean",
                            help="Method to use for steering vector computation")
    steer_parser.add_argument("--samples", type=int, default=20,
                            help="Number of samples to use from each dataset for steering vector computation")
    steer_parser.add_argument("--alpha", type=float, default=1.0,
                            help="Steering strength")
    steer_parser.add_argument("--second-target-dataset", type=str,
                            help="Path to second target dataset for compositional steering")
    steer_parser.add_argument("--beta", type=float,
                            help="Second steering strength (for compositional steering)")
    steer_parser.add_argument("--output-dir", type=str, default="steering_results",
                            help="Directory to save steering results")
    steer_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                            help="Device to run on ('cpu' or 'cuda')")
    steer_parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                            help="Logging level")
    
    # Separability analysis command
    analyze_parser = subparsers.add_parser("analyze", help="Run separability analysis on time series data")
    analyze_parser.add_argument("--dataset1", type=str, required=True,
                              help="Path to the first dataset (parquet)")
    analyze_parser.add_argument("--dataset2", type=str, required=True,
                              help="Path to the second dataset (parquet)")
    analyze_parser.add_argument("--type", type=str, choices=["constant-sine", "trend", "periodicity"], required=True,
                              help="Type of analysis to run")
    analyze_parser.add_argument("--model", type=str, choices=["moment", "chronos"], default="moment",
                              help="Model to use for analysis")
    analyze_parser.add_argument("--samples", type=int, default=20,
                              help="Number of samples to use from each dataset")
    analyze_parser.add_argument("--output-dir", type=str, default="analysis_results",
                              help="Directory to save analysis results")
    analyze_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                              help="Device to run on ('cpu' or 'cuda')")
    analyze_parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                              help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    if hasattr(args, 'log_level'):
        logging.basicConfig(level=getattr(logging, args.log_level), 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if args.command == "generate":
        generate_datasets(config_dir=args.config_dir, random_seed=args.seed, 
                          only_default=args.only_default, visualize=args.visualize,
                          output_dir=args.output_dir)
    elif args.command == "visualize-data":
        if args.dataset:
            visualize_dataset(args.dataset, args.output_dir, args.samples)
        else:
            visualize_all_datasets(args.dataset_dir, args.output_dir, args.samples, args.skip_ecg)
    elif args.command == "steer":
        run_steering_experiment(
            source_dataset_path=args.source_dataset,
            target_dataset_path=args.target_dataset,
            input_sample_path=args.input_sample,
            input_sample_index=args.input_sample_index,
            model_type=args.model,
            method=args.method,
            num_samples=args.samples,
            alpha=args.alpha,
            beta=args.beta,
            second_target_dataset_path=args.second_target_dataset,
            output_dir=args.output_dir,
            device=args.device
        )
        logging.info(f"Steering experiment completed. Results saved to {args.output_dir}")
    elif args.command == "analyze":
        run_separability_analysis(
            dataset1_path=args.dataset1,
            dataset2_path=args.dataset2,
            analysis_type=args.type,
            model_type=args.model,
            num_samples=args.samples,
            output_dir=args.output_dir,
            device=args.device
        )
        logging.info(f"Separability analysis completed. Results saved to {args.output_dir}")
    elif not args.command:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 