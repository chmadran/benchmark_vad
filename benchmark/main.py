import torch
from tqdm import tqdm
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

from config.config import init_experiments_from_config
from utils.utils import create_log_dirs
from benchmark.benchmark import benchmark_and_log_models
from post_process.post_process import PostProcessor

#TODO: Pre and Post processing of the audio could be improved honestly, maybe in a class?
#TODO: Handling of parameters is dirty af
#TODO: Potentially cleaner to create a Factory SD for the VADs

AVAILABLE_VAD_MODELS = ["webrtc", "silero", "pyannote"] 


def run(args):
    log_dir = create_log_dirs(args.log_dir, args.models)

    experiments = init_experiments_from_config(args.config_file, args.models)
    if not experiments:
            print("No valid experiments were generated.")
            return
    results = benchmark_and_log_models(args.audio_files, experiments, log_dir)
    
    if results:
        result_processor = PostProcessor(args.models, results)
        result_processor.print_mean_results()
        result_processor.compute_print_best_model(model_specific="silero", metric="F1")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models", required=True, nargs='*', choices=AVAILABLE_VAD_MODELS,
        help="List of models to benchmark.",
    )

    parser.add_argument(
        "--audio_files", required=False, type=list[list],
        help="List of path with .wav audio and corresponding labels."
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

