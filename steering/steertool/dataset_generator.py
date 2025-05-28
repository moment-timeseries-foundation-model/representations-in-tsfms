import os
import logging
from .data_generator import generate_and_save_dataset
from .utils import seed_everything
from .visualizer import visualize_dataset

def generate_datasets(config_dir=None, random_seed=42, only_default=False, visualize=False, output_dir="data_visualizations"):
    """
    Generate time series datasets from configuration files.
    
    Args:
        config_dir (str, optional): Directory containing configuration files
        random_seed (int): Random seed for reproducibility
        only_default (bool): Whether to generate only from the default config.yaml file
        visualize (bool): Whether to visualize the generated datasets
        output_dir (str): Directory to save visualizations (if visualize=True)
    """
    seed_everything(random_seed)
    logging.info(f"Using random seed: {random_seed}")
    
    if config_dir is None:
        logging.info("No config directory specified, please provide a config directory")
        return
    
    if not os.path.exists(os.path.join(os.getcwd(), config_dir)):
        logging.error(f"Config directory {config_dir} does not exist")
        return
    
    config_path = os.path.join(os.getcwd(), config_dir)
    all_files = os.listdir(config_path)
    
    if only_default:
        if "config.yaml" in all_files:
            configs = [os.path.join(config_dir, "config.yaml")]
            logging.info("Generating only from default config.yaml file")
        else:
            logging.error("Default config.yaml file not found in the config directory")
            return
    else:
        if "config.yaml" in all_files:
            all_files.remove("config.yaml")
            logging.info("Skipping default config.yaml file")
        
        configs = [os.path.join(config_dir, file) for file in all_files]
    
    logging.info(f"Found {len(configs)} configuration files to process in {config_path}")
    
    generated_datasets = []
    
    for file in configs:
        logging.info(f"Processing config file: {file}")
        
        if os.path.basename(file) == "config.yaml":
            dataset_filename = "default.parquet"
        else:
            dataset_filename = file.split('/')[-1].split('.')[0] + '.parquet'
            
        output_path = os.path.join("datasets", dataset_filename)
        logging.info(f"Will save dataset to: {os.path.abspath(output_path)}")
        
        generate_and_save_dataset(config_file=file, dataset_name=dataset_filename)
        logging.info(f"Successfully generated dataset from {file}")
        
        generated_datasets.append(output_path)
    
    if visualize and generated_datasets:
        logging.info(f"Visualizing {len(generated_datasets)} generated datasets")
        for dataset_path in generated_datasets:
            visualize_dataset(dataset_path, output_dir)
        logging.info(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    generate_datasets() 