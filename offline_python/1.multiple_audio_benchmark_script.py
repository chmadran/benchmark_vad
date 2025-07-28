import torch
from pathlib import Path
import argparse
import json
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
from utils import process_data, load_labels, convert_predictions_to_seconds, get_memory_usage_mb
from post_process import match_segments, compare_model_metrics
from models.silero import SileroVAD
from models.webrtc import WebRTCVAD
from IPython.display import Audio

AUDIO_PATH = [["../data/en_example_v0.wav", "../data/en_example_labels_v0.json"], 
              ["../data/en_example_v1.wav", "../data/en_example_labels_v1.json"]]
AVAILABLE_VAD_MODELS = ["webrtc", "silero"] 

def run_benchmark(models, audio_path, label_audio_path):
    preds = {}
    sr, y, wav_bytes, wav_tensor = process_data(audio_path)
    audio_duration = len(y) / sr
    if "silero" in models: 
        silero = SileroVAD(sr, threshold=0.3)    
        start = time.time()
        mem_start = get_memory_usage_mb()
        confidence_silero = silero.predict(wav_tensor[0])
        predictions_silero = silero.get_timestamps(wav_tensor[0])
        print("Predictions silero : ")
        print(predictions_silero)
        end = time.time()
        mem_end = get_memory_usage_mb()
        preds["silero"] = {"preds_ms" : predictions_silero,
                            "inference_time" : end - start,
                             "confidence" : confidence_silero }
        preds["silero"]["rtf"] = preds["silero"]["inference_time"] / audio_duration
        preds["silero"]["memory_MB"] = mem_end - mem_start

    if "webrtc" in models:
        webrtc = WebRTCVAD(sampling_rate=sr, window_size_samples=0.03, mode=1)
        start = time.time()
        mem_start = get_memory_usage_mb()
        predictions_webrtc = webrtc.predict(y, wav_bytes)
        end = time.time()
        mem_end = get_memory_usage_mb()
        predictions_webrtc = webrtc.merge_speech_segments(predictions_webrtc)
        preds["webrtc"] = {"preds_ms" : predictions_webrtc,
                        "inference_time" : end - start}    
        preds["webrtc"]["rtf"] = preds["webrtc"]["inference_time"] / audio_duration
        preds["webrtc"]["memory_MB"] = mem_end - mem_start

    truth = load_labels(label_audio_path)
    for model, pred in preds.items():
        preds_in_seconds = convert_predictions_to_seconds(pred["preds_ms"], sr)
        preds[model]["metrics"]= match_segments(preds_in_seconds, truth)
        preds[model]["preds_s"] = preds_in_seconds

    return preds, truth, audio_duration


def benchmark_and_log_models(audio_paths, models, log_dir):
    benchmark_results = {model: [] for model in models}

    for audio_path, label_path in audio_paths:
        model_predictions, _, duration = run_benchmark(models, audio_path, label_path)

        for model, stats in model_predictions.items():
            result_entry = {
                "audio": os.path.basename(audio_path),
                "model": model,
                **stats["metrics"],
                "inference_time": stats["inference_time"],
                "rtf": stats["rtf"],
                "memory_MB": stats["memory_MB"],
            }
            benchmark_results[model].append(result_entry)

    for model, results in benchmark_results.items():
        result_file_path = os.path.join(log_dir, model, "results.json")
        with open(result_file_path, "w") as f:
            json.dump(results, f, indent=2)

    return benchmark_results


def create_log_dirs(args_log_dir, args_models):
    log_dir = os.path.join(str(args_log_dir), datetime.now().strftime("%Y%m%d_%H:%M:%S"))
    os.makedirs(log_dir, exist_ok=True)
    for model in args_models:
        model_log_dir = os.path.join(log_dir, model)
        try :
            os.makedirs(model_log_dir, exist_ok=True)
        except Exception as e:
            print(f"An error occurred: {e}")
    return log_dir

def run(args):
    log_dir = create_log_dirs(args.log_dir, args.models)
    benchmark_results = benchmark_and_log_models(AUDIO_PATH, args.models, log_dir)
    df = compare_model_metrics(args.models, benchmark_results)
    print(df.round(2))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models", required=True, nargs='*', choices=AVAILABLE_VAD_MODELS,
        help="List of models that can be benchmarked.",
    )

    parser.add_argument(
        "--log_dir", required=True, type=Path,
        help="Path to dir where we store xp results."
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

