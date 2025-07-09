#!/usr/bin/env python3

import warnings
# Silence specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="momentfm")

import torch
import time
import argparse
import json
import os
import numpy as np
from datetime import datetime
from torch.profiler import profile, record_function, ProfilerActivity

# Model imports
try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    print("Warning: Chronos not available")

try:
    from momentfm import MOMENTPipeline
    MOMENT_AVAILABLE = True
except ImportError:
    MOMENT_AVAILABLE = False
    print("Warning: MOMENT not available")

# Configuration
DEFAULT_RUNS = 10
WARMUP_RUNS = 5
PROFILE_RUNS = 5
PROFILE_WARMUP_RUNS = 5

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Skip transformer blocks in time series models.')
    
    # Model selection
    parser.add_argument('--model_type', type=str, choices=['chronos', 'moment'], required=True,
                        help='Type of model to evaluate (chronos or moment)')
    
    # Block skipping configuration
    parser.add_argument('--blocks_to_skip', type=str, default='2-4,10-17,20-22',
                        help='Blocks to skip, format: start-end,start-end,...')
    parser.add_argument('--skip_name', type=str, default='default',
                        help='Name for this skipping configuration')
    
    # Evaluation settings
    parser.add_argument('--num_inference_samples', type=int, default=40,
                        help='Number of samples for inference speed measurement')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--sequence_length', type=int, default=512,
                        help='Input sequence length')
    
    # Device and saving options
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (e.g., "cpu", "cuda", "mps"). Auto-detect if not specified')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the modified model weights')
    parser.add_argument('--save_metadata', action='store_true', default=True,
                        help='Save evaluation metadata')
    parser.add_argument('--no_skip', action='store_true',
                        help='Evaluate original model without skipping (baseline)')
    
    # Profiling options
    parser.add_argument('--profile', action='store_true',
                        help='Run detailed profiling')
    parser.add_argument('--output_dir', type=str, default='results/pruning',
                        help='Directory to save results')
    
    return parser.parse_args()

def setup_device(device_arg=None):
    """Setup and return the appropriate device."""
    if device_arg is not None:
        device = torch.device(device_arg)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    print(f"Using device: {device}")
    return device

def parse_blocks_to_skip(blocks_str):
    """Parse blocks string into list of block indices."""
    blocks = []
    for part in blocks_str.split(','):
        start_end = part.split('-')
        if len(start_end) == 2:
            start, end = map(int, start_end)
        else:
            start = end = int(start_end[0])
        blocks.extend(list(range(start, end + 1)))
    return sorted(blocks)

def skip_chronos_blocks(model_pipeline, blocks_to_skip):
    """Skip specified transformer blocks in Chronos model."""
    print(f"Skipping Chronos blocks: {blocks_to_skip}")
    
    # Create a new ModuleList with only the blocks we want to keep
    kept_blocks = torch.nn.ModuleList()
    for i, block in enumerate(model_pipeline.model.model.encoder.block):
        if i not in blocks_to_skip:
            kept_blocks.append(block)
    
    # Replace the blocks list with our filtered list
    model_pipeline.model.model.encoder.block = kept_blocks
    
    # Calculate statistics
    # Chronos has encoder-decoder architecture (24 encoder + 24 decoder blocks)
    # Only encoder blocks are being pruned, so compute reduction is relative to total model
    original_encoder_blocks = 24  # T5 encoder has 24 blocks
    total_blocks = 48  # 24 encoder + 24 decoder blocks
    kept_blocks_count = len(kept_blocks)
    skipped_blocks = original_encoder_blocks - kept_blocks_count
    # Compute reduction is relative to the entire model (encoder + decoder)
    compute_reduction = skipped_blocks / total_blocks
    
    return model_pipeline, {
        "model_type": "chronos",
        "blocks_skipped": blocks_to_skip,
        "original_blocks": original_encoder_blocks,
        "remaining_blocks": kept_blocks_count,
        "blocks_skipped_count": skipped_blocks,
        "compute_reduction_ratio": compute_reduction,
        "expected_speedup": 1 / (1 - compute_reduction) if compute_reduction < 1 else float('inf')
    }

def skip_moment_blocks(model_pipeline, blocks_to_skip):
    """Skip specified transformer blocks in MOMENT model."""
    print(f"Skipping MOMENT blocks: {blocks_to_skip}")
    
    # Create a new ModuleList with only the blocks we want to keep
    kept_blocks = torch.nn.ModuleList()
    for i, block in enumerate(model_pipeline.encoder.block):
        if i not in blocks_to_skip:
            kept_blocks.append(block)
    
    # Replace the blocks list with our filtered list
    model_pipeline.encoder.block = kept_blocks
    
    # Calculate statistics
    original_blocks = 24  # MOMENT encoder has 24 blocks
    kept_blocks_count = len(kept_blocks)
    skipped_blocks = original_blocks - kept_blocks_count
    compute_reduction = skipped_blocks / original_blocks
    
    return model_pipeline, {
        "model_type": "moment",
        "blocks_skipped": blocks_to_skip,
        "original_blocks": original_blocks,
        "remaining_blocks": kept_blocks_count,
        "blocks_skipped_count": skipped_blocks,
        "compute_reduction_ratio": compute_reduction,
        "expected_speedup": 1 / (1 - compute_reduction) if compute_reduction < 1 else float('inf')
    }

def load_chronos_model_pipeline(device):
    """Load and initialize Chronos model."""
    if not CHRONOS_AVAILABLE:
        raise ImportError("Chronos is not available")
    
    model_pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large",
        device_map=device.type
    )
    
    model = model_pipeline.model.to(device)
    model_pipeline.tokenizer.boundaries = model_pipeline.tokenizer.boundaries.to(device)
    model_pipeline.model = model  # Update the model in the pipeline
    
    return model_pipeline

def load_moment_model_pipeline(device):
    """Load and initialize MOMENT model."""
    if not MOMENT_AVAILABLE:
        raise ImportError("MOMENT is not available")
    
    model_pipeline = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={"task_name": "embedding"},
    )
    model_pipeline.init()
    model_pipeline.to(device)
    
    return model_pipeline

def create_sample_input(model_type, batch_size, sequence_length, device):
    """Create sample input for the specified model type."""
    if model_type == "chronos":
        return torch.randn(batch_size, sequence_length, device=device)
    elif model_type == "moment":
        return torch.randn(batch_size, 1, sequence_length, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def run_single_inference(model_pipeline, model_type, input_sample):
    """Run a single inference for the specified model type."""
    if model_type == "chronos":
        return model_pipeline.predict(input_sample, prediction_length=1, num_samples=1)
    elif model_type == "moment":
        return model_pipeline.embed(x_enc=input_sample)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def measure_inference_speed(model_pipeline, model_type, input_sample, num_samples=100):
    """Measure average inference speed."""
    print(f"Measuring inference speed with {num_samples} samples...")
    
    if model_type == "chronos":
        model_pipeline.model.eval()
        device = next(model_pipeline.model.parameters()).device
    else:  # moment
        model_pipeline.eval()
        device = next(model_pipeline.parameters()).device
    
    times = []
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            run_single_inference(model_pipeline, model_type, input_sample)
    
    # Measure inference time
    print("Measuring...")
    with torch.no_grad():
        for i in range(num_samples * 2):
            start_time = time.time()
            run_single_inference(model_pipeline, model_type, input_sample)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_samples * 2} measurements")
    
    # Remove warmup samples
    times = times[num_samples:]
    
    average_time = sum(times) / len(times)
    standard_deviation = np.std(times)
    
    return average_time, standard_deviation, times

def profile_inference(model_pipeline, model_type, input_sample, num_runs=PROFILE_RUNS, warmup_runs=PROFILE_WARMUP_RUNS):
    """Run detailed PyTorch profiling."""
    print(f"\nRunning detailed PyTorch profiling ({num_runs} runs)")
    
    # Warmup
    print(f"Performing {warmup_runs} warmup runs...")
    for _ in range(warmup_runs):
        run_single_inference(model_pipeline, model_type, input_sample)
    
    # Determine device for profiling
    if model_type == "chronos":
        device = next(model_pipeline.model.parameters()).device
    else:
        device = next(model_pipeline.parameters()).device
    
    # Setup profiler activities
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    
    print("Starting profiler...")
    try:
        with profile(
            activities=activities,
            with_stack=True,
            record_shapes=True,
            profile_memory=device.type == 'cuda'
        ) as prof:
            for i in range(num_runs):
                with record_function(f"inference_{i}"):
                    run_single_inference(model_pipeline, model_type, input_sample)
        
        return prof
    except Exception as e:
        print(f"Profiling error: {e}")
        return None

def analyze_profile_results(prof, run_name, output_dir):
    """Analyze and save profiling results."""
    if prof is None:
        print("No profiling results available.")
        return None
    
    print(f"\nProfiling Results for {run_name}")
    print("=" * 50)
    print("\nTop Operations by CPU Time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    # Save trace file
    os.makedirs(f"{output_dir}/traces", exist_ok=True)
    trace_filename = f"{output_dir}/traces/trace_{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    prof.export_chrome_trace(trace_filename)
    print(f"\nChrome trace exported to {trace_filename}")
    
    return prof

def calculate_model_stats(model_pipeline, model_type):
    """Calculate model statistics like parameter count."""
    if model_type == "chronos":
        params = model_pipeline.model.parameters()
    else:
        params = model_pipeline.parameters()
    
    total_params = 0
    
    for param in params:
        total_params += param.numel()
    
    return {
        "total_parameters": total_params
    }

def estimate_model_size_mb(model_pipeline, model_type):
    """Estimate model size in MB."""
    if model_type == "chronos":
        params = model_pipeline.model.parameters()
        buffers = model_pipeline.model.buffers()
    else:
        params = model_pipeline.parameters()
        buffers = model_pipeline.buffers()
    
    param_size = sum(param.nelement() * param.element_size() for param in params)
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in buffers)
    
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)
    return total_size_mb

def save_model_weights(model_pipeline, model_type, save_path):
    """Save model weights."""
    print(f"Saving model to {save_path}")
    
    if model_type == "chronos":
        model_pipeline.model.model.config.num_layers = len(model_pipeline.model.model.encoder.block)
        model_pipeline.model.model.save_pretrained(save_path)
    else:
        model_pipeline.config.t5_config["num_layers"] = len(model_pipeline.encoder.block)
        model_pipeline.save_pretrained(save_path)

def evaluate_model(model_pipeline, model_type, input_sample, model_name, args, skip_stats=None):
    """Comprehensive model evaluation."""
    print(f"\nEvaluating {model_name} model...")
    print("=" * 50)
    
    # Calculate model statistics
    model_stats = calculate_model_stats(model_pipeline, model_type)
    model_size_mb = estimate_model_size_mb(model_pipeline, model_type)
    
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Total parameters: {model_stats['total_parameters']:,}")
    
    # Measure inference speed
    avg_time, std_time, all_times = measure_inference_speed(
        model_pipeline, model_type, input_sample, args.num_inference_samples
    )
    print(f"Inference speed: {avg_time:.4f} ± {std_time:.4f} seconds")
    
    # Run profiling if requested
    prof_results = None
    if args.profile:
        prof = profile_inference(model_pipeline, model_type, input_sample)
        prof_results = analyze_profile_results(prof, model_name, args.output_dir)
    
    # Prepare results dictionary
    results = {
        "model_name": model_name,
        "model_type": model_type,
        "model_stats": model_stats,
        "model_size_mb": model_size_mb,
        "inference_speed": {
            "average_time": avg_time,
            "standard_deviation": std_time,
            "all_measurements": all_times
        },
        "evaluation_config": {
            "num_inference_samples": args.num_inference_samples,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "device": str(input_sample.device)
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Add skip statistics if provided
    if skip_stats is not None:
        results["skip_stats"] = skip_stats
    
    # Save model weights if requested (but not during profiling)
    if args.save_model and not args.profile:
        os.makedirs(f"{args.output_dir}/models", exist_ok=True)
        save_path = f"{args.output_dir}/models/{model_name}"
        save_model_weights(model_pipeline, model_type, save_path)
        results["saved_model_path"] = save_path
    elif args.save_model and args.profile:
        print("Note: Model weights not saved during profiling runs")
    
    return results

def save_results(results, filename):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj
    
    serializable_results = convert_for_json(results)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filename}")

def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Setup
    device = setup_device(args.device)
    torch.set_default_device(device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Skip Blocks Evaluation")
    print(f"Model: {args.model_type}")
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    
    # Load model
    print(f"\nLoading {args.model_type} model...")
    if args.model_type == "chronos":
        model_pipeline = load_chronos_model_pipeline(device)
    elif args.model_type == "moment":
        model_pipeline = load_moment_model_pipeline(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Create sample input
    input_sample = create_sample_input(
        args.model_type, args.batch_size, args.sequence_length, device
    )
    
    all_results = {}
    
    # Evaluate original model if requested
    if args.no_skip:
        print("\nEvaluating original model (no skipping)...")
        original_results = evaluate_model(
            model_pipeline, args.model_type, input_sample, f"{args.model_type}_original", args
        )
        all_results["original"] = original_results
    else:
        # Skip blocks and evaluate
        blocks_to_skip = parse_blocks_to_skip(args.blocks_to_skip)
        print(f"\nSkipping blocks: {blocks_to_skip}")
        
        if args.model_type == "chronos":
            model_pipeline, skip_stats = skip_chronos_blocks(model_pipeline, blocks_to_skip)
        else:
            model_pipeline, skip_stats = skip_moment_blocks(model_pipeline, blocks_to_skip)
        
        print(f"Expected speedup: {skip_stats['expected_speedup']:.2f}x")
        print(f"Compute reduction: {skip_stats['compute_reduction_ratio']:.2f}")
        
        # Evaluate modified model
        model_name = f"{args.model_type}_skipped_{args.skip_name}"
        skip_results = evaluate_model(
            model_pipeline, args.model_type, input_sample, model_name, args, skip_stats
        )
        all_results["skipped"] = skip_results
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"{args.output_dir}/evaluation_results_{args.model_type}_{timestamp}.json"
    save_results(all_results, results_filename)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {results_filename}")

if __name__ == '__main__':
    main() 