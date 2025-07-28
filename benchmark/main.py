import torch
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path
import argparse
import json
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np

from utils.utils import process_data, load_labels, convert_predictions_to_seconds, get_memory_usage_mb, create_log_dirs
from post_process.post_process import match_segments, compare_model_metrics
from grid_search.grid_search import init_grid_search

from models.silero import SileroVAD
from models.webrtc import WebRTCVAD
from IPython.display import Audio

AUDIO_FILES = [["../data/en_example_v0.wav", "../data/en_example_labels_v0.json"],
               ["../data/en_example_v1.wav", "../data/en_example_labels_v1.json"]
               ]

AVAILABLE_VAD_MODELS = ["webrtc", "silero"] 

def run_benchmark(model_name, model_params, audio_path, label_audio_path):
    preds = {}
    sr, wav, wav_bytes, wav_tensor = process_data(audio_path)
    audio_duration = len(wav) / sr

    if model_name == "silero":
        silero = SileroVAD(sampling_rate=sr, **model_params)
        start = time.time()
        mem_start = get_memory_usage_mb()
        confidence = silero.get_confidence(wav_tensor[0])
        timestamps = silero.get_predictions(wav_tensor[0])
        end = time.time()
        mem_end = get_memory_usage_mb()
        preds = {
            "preds_ms": timestamps,
            "confidence": confidence,
            "inference_time": end - start,
            "rtf": (end - start) / audio_duration,
            "memory_KB": mem_end - mem_start
        }

    elif model_name == "webrtc":
        webrtc = WebRTCVAD(sampling_rate=sr, **model_params)
        start = time.time()
        mem_start = get_memory_usage_mb()
        frames = webrtc.predict(wav, wav_bytes)
        end = time.time()
        mem_end = get_memory_usage_mb()
        frames = webrtc.merge_speech_segments(frames)
        preds = {
            "preds_ms": frames,
            "inference_time": end - start,
            "rtf": (end - start) / audio_duration,
            "memory_KB": mem_end - mem_start
        }

    truth = load_labels(label_audio_path)
    preds_in_seconds = convert_predictions_to_seconds(preds["preds_ms"], sr)
    preds["metrics"] = match_segments(preds_in_seconds, truth, threshold=0.5)
    preds["preds_s"] = preds_in_seconds

    return preds

def benchmark_and_log_models(audio_paths, model_param_grid_map, log_dir):
    all_results = []
    best_models = {} 

    if not audio_paths:
        audio_paths = AUDIO_FILES

    for model_name, param_grid in model_param_grid_map.items():
        best_model_result = None
        best_f1 = -1
        print(f"\n RUNNING BENCHMARK FOR {model_name.upper()}")
        for params in tqdm(param_grid):
            for audio_path, label_path in audio_paths:
                try:
                    stats = run_benchmark(model_name, params, audio_path, label_path)
                    result = {
                        "audio": os.path.basename(audio_path),
                        "model": model_name,
                        **params,
                        **stats["metrics"],
                        "inference_time": stats["inference_time"],
                        "rtf": stats["rtf"],
                        "memory_KB": stats["memory_KB"] * 1024,
                    }
                    all_results.append(result)

                    if result["f1"] > best_f1:
                        best_f1 = result["f1"]
                        best_model_result = result

                except Exception as e:
                    print(f"[ERROR] Model: {model_name}, Params: {params}, Audio: {audio_path}, Reason: {e}")

        if best_model_result:
            best_models[model_name] = best_model_result

        model_log_path = os.path.join(log_dir, model_name, "results.json")
        os.makedirs(os.path.dirname(model_log_path), exist_ok=True)
        with open(model_log_path, "w") as f:
            json.dump([r for r in all_results if r["model"] == model_name], f, indent=2)

    return all_results, best_models

def print_output(models, results, best_models):

    def round_df(df):
        return df.round({"f1_mean": 2, "precision_mean": 2, "recall_mean": 2})

    all_df = compare_model_metrics(models, results)
    print("\nMEAN METRICS FOR ALL PREDICTIONS:\n")
    print(tabulate(round_df(all_df), headers="keys", tablefmt="pretty"))

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
    params_grids = init_grid_search(args.grid_search, args.grid_file, args.models)
    results, best_models = benchmark_and_log_models(args.audio_files, params_grids, log_dir)
    print_output(args.models, results, best_models)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models", required=True, nargs='*', choices=AVAILABLE_VAD_MODELS,
        help="List of models that can be benchmarked.",
    )

    parser.add_argument(
        "--audio_files", required=False, type=list[list],
        help="Path to dir where we store the dataset results."
    )

    parser.add_argument(
        "--log_dir", required=False, type=Path, default="../logs/",
        help="Path to dir where we store xp results."
    )

    parser.add_argument(
        "--grid_search", default=False, action="store_true",
        help="JSON file containing parameter grids for each model."
    )

    parser.add_argument(
        "--grid_file", required=False, type=Path, 
        help="JSON file containing parameter grids for each model."
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

