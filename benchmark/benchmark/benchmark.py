from collections import defaultdict
from tabulate import tabulate
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os
import traceback
import torch
import time
import json

from utils.utils import convert_predictions_to_seconds, get_memory_usage_mb
from pre_process.loader import load_audio, load_labels

from models.vad import VADFactory

def round_df(df: pd.DataFrame, threshold: int = 2) -> pd.DataFrame:
    """
    Round the values in a DataFrame containing individual evaluation metrics.

    Args:
        df (pd.DataFrame): DataFrame with individual model results.
        threshold (int): Number of decimal places to round to (default is 2).

    Returns:
        pd.DataFrame: Rounded DataFrame.
    """
    return df.round({"F1": threshold, "Precision": threshold, "Recall": threshold})

def print_best_model(best_model: dict, metric: str) -> None:
    """
    Print the hyperparameters and evaluation metrics for a single best-performing model.

    Args:
        best_model (dict): Dictionary containing model configuration and evaluation metrics.
        metric (str): The metric used to select the best model (e.g., "F1").
    """
    df = pd.DataFrame([best_model]) 
    print("\n")
    print(f"Best hyperparameters for {best_model["model"].upper()} based on metric : {metric}\n")
    print(tabulate(round_df(df, threshold=2), headers="keys", tablefmt="fancy_grid"))
    print("\n")

def save_results_global(log_dir: Path, model: str, all_results: list):
    """
    Save benchmarking results for a specific model to a JSON file.

    Args:
        log_dir (Path): Directory where the results should be saved.
        model (str): Name of the model whose results should be saved.
        all_results (list): List of result dictionaries from all benchmarks.
    """
    model_log_path = os.path.join(log_dir, model, "inference", "global_results.json")
    os.makedirs(os.path.dirname(model_log_path), exist_ok=True)
    with open(model_log_path, "w") as f:
        json.dump([r for r in all_results if r["model"] == model], f, indent=2)

def save_results_per_model_csv(results: dict, log_dir: Path):
    """
    Save benchmarking results for a specific model to a CSV file.

    Args:
        log_dir (Path): Directory where the results should be saved.
        results (dict): Dict with keys = model name and values results dictionaries from all benchmarks.
    """

    for model, results in results.items():
        df = pd.DataFrame(results)
        param_df = df["parameters"].apply(pd.Series)
        df= df.drop(columns=["model", "parameters"])
        df = pd.concat([df, param_df], axis=1)
        csv_filename = os.path.join(log_dir, model, "grid_search", "benchmark_results.csv")
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        df.to_csv(csv_filename, index=False)
    

def save_results_per_audio(log_dir: Path, model: str, all_results: list):
    """
    Save all benchmarking results grouped by (model, audio) pair.

    Args:
        log_dir (Path): Directory to save result files.
        all_results (list): List of individual result dictionaries.
    """
    grouped = defaultdict(list)

    for result in all_results:
        if result["model"] != model:
            continue
        audio_name, _ = os.path.splitext(result["audio"])
        grouped[audio_name].append(result)

    for audio_name, results in grouped.items():
        file_path = os.path.join(log_dir, model, "inference", f"{audio_name}.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)


def run_benchmark(model_name: str, model_params: dict, audio: dict):
    
    preds = {}
    model = VADFactory.create(model_name, model_params, audio)
    start = time.time()
    mem_start = get_memory_usage_mb()
    
    preds = model.predict()

    end = time.time()
    mem_end = get_memory_usage_mb()

    preds["inference_time"] = end - start
    preds["rtf"] = (end - start) / audio["duration"]
    preds["memory_KB"] = mem_end - mem_start

    return preds


def run_experiments(dataset:list, experiments:dict, log_dir:Path, grid_search: bool=False) -> tuple[list, dict]:
    """
    Run benchmarking for multiple models across parameter combinations and audio files, 
    and log the results.

    Args:
        audio_paths (list): List of (audio_path, label_path) tuples. Falls back to DEFAULT_AUDIO_FILES if empty.
        experiments (dict): A dictionary mapping model names to lists of parameter combinations.
        log_dir (Path): Directory where benchmark results will be stored per model.

    Returns:
        list: 
            - A list of all benchmarking results (one per audio and param combination).
    """
    results = {}
    best_models = {}
    new_experiments = {}

    for model, params in experiments.items():
        best_score = -1
        all_results = []
        if grid_search:
            print(f"\n RUNNING GRID SEARCH FOR {model.upper()}")
        else:
            print(f"\n RUNNING INFERENCE FOR {model.upper()}")

        for param in tqdm(params):
            for raw_audio_path, label_audio_path in dataset:
                try:
                    audio = load_audio(raw_audio_path, label_audio_path)
                except Exception as e:
                    print(f"Error while loading audio {raw_audio_path} : {e}")
                    continue
                try:
                    predictions = run_benchmark(model, param, audio)
                    result = {
                        "audio": os.path.basename(raw_audio_path),
                        "model": model,
                        "parameters" : param, 
                        **predictions["metrics"],
                        "Inference_time": predictions["inference_time"],
                        "RTF": predictions["rtf"],
                        "Memory_KB": predictions["memory_KB"] * 1024,
                    }
                    all_results.append(result)
                    if result["F1"] > best_score:
                        best_score =  result["F1"] 
                        best_model = result

                except Exception as e:
                    traceback.print_exc()
                    print(f"[ERROR] Model: {model}, Params: {param}, audio: {raw_audio_path}, Reason: {e}")

        new_experiments[model] = [best_model["parameters"]]
        results[model] = all_results

        if grid_search:
            best_models[model] = [best_model]
            print_best_model(best_model, metric="F1")
            save_results_per_model_csv(results, log_dir)
        else:
            save_results_global(log_dir, model, all_results)
            save_results_per_audio(log_dir, model, all_results)

    return new_experiments
