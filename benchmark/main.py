import torch
from tqdm import tqdm
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

from config.config import init_experiments_from_config
from utils.utils import create_log_dirs
from benchmark.benchmark.benchmark import run_experiments
from post_process.post_process import PostProcessor
from pre_process.dataset import split_dataset_for_grid_search

#TODO: Pre and Post processing of the audio could be improved honestly, maybe in a class?
#TODO: Handling of parameters is dirty af
#TODO: Potentially cleaner to create a Factory SD for the VADs

AVAILABLE_VAD_MODELS = ["webrtc", "silero", "pyannote"] 

def run(args):
    log_dir = create_log_dirs(args.log_dir, args.xp_name, args.models)
    grid_search_dataset, inference_dataset = split_dataset_for_grid_search(args.path_audio_files, args.grid_search_dataset_percentage, shuffle=False)    
    experiments = init_experiments_from_config(args.config_file, args.models, log_dir)
    
    if grid_search_dataset:
        best_models, experiments = run_experiments(grid_search_dataset, experiments, log_dir, grid_search=True)
    predictions = run_experiments(inference_dataset, experiments, log_dir)

    # if predictions:
    #     result_processor = PostProcessor(args.models, predictions)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models", required=True, nargs='*', choices=AVAILABLE_VAD_MODELS,
        help="List of models to benchmark.",
    )

    parser.add_argument(
        "--xp_name", required=False, type=str, default="unknown_xp",
        help="Name of experiment being launched.",
    )

    parser.add_argument(
        "--grid_search_dataset_percentage", required=False, type=int, default=25,
        help="Percentage of the dataset that is used for the gridsearch.",
    )

    parser.add_argument(
        "--path_audio_files", required=False, type=list[list],
        help="Path of folder containing .wav audio and corresponding labels."
    )

    parser.add_argument( #TODO: Turn off logging if we want?
        "--log_dir", required=False, type=Path, default="../logs/",
        help="Path to dir where we store xp results."
    )

    parser.add_argument(
        "--config_file", required=False, type=Path, 
        help="JSON file containing parameter grids (optionnal) for each model."
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

