import time
import logging
from src.util import set_seed, extract_param
from src.dataloader import AMLData
from src.training import train_gnn

import random
from datetime import datetime
import os
import sys
import yaml
import argparse
import torch
from types import SimpleNamespace

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="/mnt/cagri/data", type=str, required=True)
    parser.add_argument("--output_dir", default="/home/cbilgi/cse3000/megagnn/results", type=str, required=True)
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")
    
    #Model parameters
    parser.add_argument("--batch_size", default=8192, type=int, help="Select the batch size for GNN training")
    parser.add_argument("--n_epochs", default=100, type=int, help="Select the number of epochs for GNN training")
    parser.add_argument('--num_neighs', nargs='+', default=[100,100], help='Pass the number of neighors to be sampled in each hop (descending).')

    #Misc
    parser.add_argument("--device", default="cuda:0", type=str, help="Select a GPU", required=False)
    parser.add_argument("--seed", default=1, type=int, help="Select the random seed for reproducability")
    parser.add_argument("--data", default=None, type=str, help="Select the AML dataset. Needs to be either small or medium.", required=True)
    parser.add_argument("--model", default=None, type=str, help="Select the model architecture. Needs to be one of [gin, pna]", required=True)
    parser.add_argument("--testing", action='store_true', help="Disable wandb logging while running the script in 'testing' mode.")
    parser.add_argument("--save_model", action='store_true', help="Save the best model.")

    return parser

def logger_setup(log_dir: str):
    """Setup logging to file and stdout"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "logs.log")),  # Log to run-specific log file
            logging.StreamHandler(sys.stdout)                        # Log also to stdout
        ]
    )

def setup_config(args):
    """Setup configuration and logging directories, consolidating all args into config
    
    Args:
        args: Command line arguments
        
    Returns:
        SimpleNamespace: Configuration object with attribute-style access
    """
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{args.data}_{timestamp}")
    log_dir = os.path.join(run_dir, "logs")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger_setup(log_dir)   
    
    # Create config dictionary with all args included
    config_dict = {
        # Training configuration
        "epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "model": args.model,
        "data": args.data,
        "num_neighs": args.num_neighs,
        "lr": extract_param("lr", args),
        "n_hidden": extract_param("n_hidden", args),
        "n_gnn_layers": extract_param("n_gnn_layers", args),
        "loss": "ce",
        "w_ce1": extract_param("w_ce1", args),
        "w_ce2": extract_param("w_ce2", args),
        "dropout": extract_param("dropout", args),
        "final_dropout": extract_param("final_dropout", args),
        
        # Directories
        "run_dir": run_dir,
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
        "output_dir": args.output_dir,
        "data_path": args.data_path,
        
        # Other args
        "seed": args.seed,
        "device": torch.device(args.device if torch.cuda.is_available() else "cpu"),
        "emlps": args.emlps,
        "save_model": args.save_model if hasattr(args, 'save_model') else False
    }
    
    # Add any other args that weren't explicitly handled
    for key, value in vars(args).items():
        if key not in config_dict:
            config_dict[key] = value

    # Print the config 
    print("----- CONFIG -----")
    for key, value in config_dict.items():
        print(f"{key}: {value}")
    print(2*"------------------")

    # Save config to file
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    logging.info(f"Configuration saved to {config_path}")
    
    # Convert dictionary to SimpleNamespace for attribute-style access
    config = SimpleNamespace(**config_dict)
    
    return config

def main():
    parser = create_parser()
    args = parser.parse_args()
    args.num_neighs = [int(t) for t in args.num_neighs]

    # Setup configuration 
    config = setup_config(args)

    # Set seed
    if args.seed == 1:
        args.seed = random.randint(2, 256000)
    set_seed(args.seed)

    # Get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()
    dataset = AMLData(config)  # Use config instead of args
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = dataset.get_data()
    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")

    # Training (only passing config, not args)
    logging.info(f"Running Training")
    train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, config)


if __name__ == "__main__":
    main()
