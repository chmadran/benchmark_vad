from pathlib import Path
from collections import defaultdict
import os
import time
import json
from tqdm import tqdm

from utils.utils import process_data, load_labels, convert_predictions_to_seconds, get_memory_usage_mb

from models.silero import SileroVAD
from models.webrtc import WebRTCVAD
from models.pyannote import PyAnnoteVAD

DEFAULT_AUDIO_FILES = [["../data/en_example_v0.wav", "../data/en_example_labels_v0.json"],
               ["../data/en_example_v1.wav", "../data/en_example_labels_v1.json"]
               ]

def save_results_global(log_dir: Path, model: str, all_results: list):
    """
    Save benchmarking results for a specific model to a JSON file.

    Args:
        log_dir (Path): Directory where the results should be saved.
        model (str): Name of the model whose results should be saved.
        all_results (list): List of result dictionaries from all benchmarks.
    """
    model_log_path = os.path.join(log_dir, model, "results.json")
    os.makedirs(os.path.dirname(model_log_path), exist_ok=True)
    with open(model_log_path, "w") as f:
        json.dump([r for r in all_results if r["model"] == model], f, indent=2)


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
        audio_name = result["audio"]
        grouped[audio_name].append(result)

    for audio_name, results in grouped.items():
        file_path = os.path.join(log_dir, model, f"{audio_name}.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)


def compute_overlap(seg1: tuple, seg2: tuple) -> float:
    """
    Compute the "Intersection over Union" (IoU) between two time segments.

    Each segment is represented as a tuple or list of two floats: (start_time, end_time).
    IoU is calculated as the ratio of the duration of the intersection to the duration of the union.

    Args:
        seg1 (tuple): First segment (start_time, end_time) in seconds.
        seg2 (tuple): Second segment (start_time, end_time) in seconds.

    Returns:
        float: IoU score between 0 and 1. Returns 0 if the segments do not overlap.
    """
    start = max(seg1[0], seg2[0])
    end = min(seg1[1], seg2[1])
    intersection = max(0, end - start)
    union = max(seg1[1], seg2[1]) - min(seg1[0], seg2[0])
    return intersection / union if union > 0 else 0

def match_segments(predicted: list[tuple[float, float]], ground_truth: tuple[float, float], threshold: float) -> dict:
    """
    Match predicted speech segments to ground truth segments using an IoU threshold,
    and compute evaluation metrics: precision, recall, and F1-score.

    A predicted segment is considered a true positive if it overlaps with any ground truth
    segment with IoU ≥ threshold. Remaining unmatched predicted segments are false positives,
    and unmatched ground truth segments are false negatives.

    Args:
        predicted (list of tuple): List of predicted segments, each as (start_time, end_time) in seconds.
        ground_truth (list of tuple): List of ground truth segments, each as (start_time, end_time) in seconds.
        threshold (float): IoU threshold for determining a match (between 0 and 1).

    Returns:
        dict: Dictionary with evaluation metrics:
            - "precision": Precision percentage (TP / (TP + FP)) × 100
            - "recall": Recall percentage (TP / (TP + FN)) × 100
            - "f1": F1-score percentage (harmonic mean of precision and recall)
    """
    TP = 0
    for pred in predicted:
        if any(compute_overlap(pred, gt) >= threshold for gt in ground_truth):
            TP += 1
    FP = len(predicted) - TP
    FN = len(ground_truth) - TP
    precision = TP / (TP + FP + 1e-6) * 100
    recall = TP / (TP + FN + 1e-6) * 100
    f1 = 2 * precision * recall / (precision + recall + 1e-6) 
    return {
        "Precision" : precision, 
        "Recall" : recall, 
        "F1": f1
        }

def run_benchmark(model_name: str, model_params: dict, audio_path: Path, label_audio_path: Path):
    """
    Run a VAD benchmark for a given model and parameter configuration on a single audio file.

    Args:
        model_name (str): Name of the VAD model to benchmark ("silero", "webrtc", or "pyannote").
        model_params (dict): Dictionary of hyperparameters to initialize the model with.
        audio_path (Path): Path to the input audio file.
        label_audio_path (Path): Path to the label file for the audio.

    Returns:
        dict: Dictionary containing:
            - "experiment_id" : Id of the experiment (the hyperparameters tried).
            - "preds_s": List of predicted speech segments in seconds.
            - "inference_time": Total inference time in seconds.
            - "rtf": Real-time factor (inference_time / audio duration).
            - "memory_KB": Memory usage during inference in KB.
            - "metrics": Evaluation metrics computed by `match_segments` (e.g F1 score, precision, recall).
            - (optional) "confidence": Model confidence score.
            - (optional) "preds_ms": Raw prediction output in milliseconds.
    """
    preds = {}
    sr, wav, wav_bytes, wav_tensor = process_data(audio_path)
    audio_duration = len(wav) / sr

    if model_name == "silero":
        silero = SileroVAD(sampling_rate=sr, **model_params)
        start = time.time()
        mem_start = get_memory_usage_mb()
        confidence = silero.get_confidence(wav_tensor[0])
        frames = silero.get_predictions(wav_tensor[0])
        end = time.time()
        mem_end = get_memory_usage_mb()
        preds = {
            "preds_ms": frames,
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

    elif model_name == "pyannote":
        pyannotevad = PyAnnoteVAD(**model_params)
        start = time.time()
        mem_start = get_memory_usage_mb()
        frames = pyannotevad.predict(audio_path)
        end = time.time()
        mem_end = get_memory_usage_mb()
        preds = {
            "preds_s": frames,
            "inference_time": end - start,
            "rtf": (end - start) / audio_duration,
            "memory_KB": mem_end - mem_start
        }

    truth = load_labels(label_audio_path)
    if "preds_s" not in preds:
        preds_in_seconds = convert_predictions_to_seconds(preds["preds_ms"], sr)
        preds["preds_s"] = preds_in_seconds
    preds["metrics"] = match_segments(preds["preds_s"], truth, threshold=0.5)

    return preds

def benchmark_and_log_models(audio_paths:list, experiments:dict, log_dir:Path) -> tuple[list, dict]:
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
    all_results = []

    if not audio_paths:
        audio_paths = DEFAULT_AUDIO_FILES

    for model, params in experiments.items():
        print(f"\n RUNNING BENCHMARK FOR {model.upper()}")
        for param in tqdm(params):
            for audio_path, label_path in audio_paths:
                try:
                    predictions = run_benchmark(model, param, audio_path, label_path)
                    result = {
                        "audio": os.path.basename(audio_path),
                        "model": model,
                        **param,
                        **predictions["metrics"],
                        "Inference_time": predictions["inference_time"],
                        "RTF": predictions["rtf"],
                        "Memory_KB": predictions["memory_KB"] * 1024,
                    }
                    all_results.append(result)

                except Exception as e:
                    print(f"[ERROR] Model: {model}, Params: {params}, audio: {audio_path}, Reason: {e}")

        save_results_global(log_dir, model, all_results)
        save_results_per_audio(log_dir, model, all_results)

    return all_results
