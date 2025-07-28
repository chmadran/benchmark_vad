import pandas as pd
import numpy as np

def compute_overlap(seg1, seg2):
    """Computes intersection over union (IoU) between two segments"""
    start = max(seg1[0], seg2[0])
    end = min(seg1[1], seg2[1])
    intersection = max(0, end - start)
    union = max(seg1[1], seg2[1]) - min(seg1[0], seg2[0])
    return intersection / union if union > 0 else 0

def match_segments(predicted, ground_truth, threshold=0.5):
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


def compare_model_metrics(models, all_results):
    f1 = []
    precision = []
    recall = []
    mean_data = {}

    for model in models:
        for result in all_results[model]:
            f1.append(result["f1"])
            precision.append(result["precision"])
            recall.append(result["recall"])
        
        mean_data[model] = {
            "f1_mean" : np.mean(f1),
            "precision_mean" : np.mean(precision),
            "recall_mean" : np.mean(recall),
        }

    df = pd.DataFrame(mean_data).T
    return df
