import torch
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

from utils.utils import create_log_dirs
from post_process.post_process import compare_model_metrics
from config.config import init_experiments_from_config
from benchmark.benchmark import benchmark_and_log_models

#TODO: Pre and Post processing of the audio could be improved honestly, maybe in a class?
#TODO: Handling of parameters is dirty af
#TODO: Potentially cleaner to create a Factory SD for the VADs

AVAILABLE_VAD_MODELS = ["webrtc", "silero", "pyannote"] 

def round_df(df):
    return df.round({"f1_mean": 2, "precision_mean": 2, "recall_mean": 2})


def print_output(models: list, results: list):
    all_df = compare_model_metrics(models, results)
    print("\nMEAN METRICS FOR ALL PREDICTIONS:\n")
    print(tabulate(round_df(all_df), headers="keys", tablefmt="pretty"))


def print_best_models(models: list, best_models: dict):
    best_results = list(best_models.values())
    best_df = compare_model_metrics(models, best_results)
    print("\nBEST MODEL METRICS:\n")
    print(tabulate(round_df(best_df), headers="keys", tablefmt="pretty"))

    print("\nBEST MODEL CONFIGS (based on f1):\n")
    for model, config in best_models.items():
        print(f"\n→ {model.upper()}")
        for k, v in config.items():
            if k not in {"preds_ms", "confidence", "preds_s"}:
                print(f"  {k}: {v}")

def run(args):
    log_dir = create_log_dirs(args.log_dir, args.models)

    experiments = init_experiments_from_config(args.config_file, args.models)
    if not experiments:
            print("No valid experiments were generated.")
            return
    
    results, best_models = benchmark_and_log_models(args.audio_files, experiments, log_dir)
    
    if results:
        print_output(args.models, results)
        print_best_models(args.models, best_models)

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

