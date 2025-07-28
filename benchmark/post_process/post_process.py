import pandas as pd
import numpy as np

def compute_overlap(seg1, seg2):
    """Computes intersection over union (IoU) between two segments"""
    start = max(seg1[0], seg2[0])
    end = min(seg1[1], seg2[1])
    intersection = max(0, end - start)
    union = max(seg1[1], seg2[1]) - min(seg1[0], seg2[0])
    return intersection / union if union > 0 else 0

def match_segments(predicted, ground_truth, threshold):
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
        "precision" : precision, 
        "recall" : recall, 
        "f1": f1
        }

def compare_model_metrics(models: list, all_results: list):
    mean_data = {}

    for model in models:
        model_results = [r for r in all_results if r["model"] == model]

        if not model_results:
            continue 

        f1_scores = [r["f1"] for r in model_results]
        precision_scores = [r["precision"] for r in model_results]
        recall_scores = [r["recall"] for r in model_results]

        mean_data[model] = {
            "f1_mean": np.mean(f1_scores),
            "precision_mean": np.mean(precision_scores),
            "recall_mean": np.mean(recall_scores),
        }

    df = pd.DataFrame(mean_data).T
    return df

