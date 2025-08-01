from abc import ABC, abstractmethod
import traceback

class BaseVAD(ABC):
    def __init__(self, 
                 audio: dict, 
                 experiment_id: int = 0):
        self.audio = audio
        self.experiment_id = experiment_id

    @abstractmethod
    def predict(self) -> dict:
        pass

    @staticmethod
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

    @staticmethod
    def match_segments(predicted: list, ground_truth: list, threshold: float) -> dict:
        """
        Match predicted speech segments to ground truth segments using an IoU threshold,
        and compute evaluation metrics: precision, recall, and F1-score.

        A predicted segment is considered a true positive if it overlaps with any ground truth
        segment with IoU ≥ threshold. Remaining unmatched predicted segments are false positives,
        and unmatched ground truth segments are false negatives.
        """
        try:
            TP = 0
            for pred in predicted:
                if any(BaseVAD.compute_overlap(pred, gt) >= threshold for gt in ground_truth):
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
        except Exception as e:
            traceback.print_exc()
            print("ERROR: ", e)

    @staticmethod
    def convert_ms_to_seconds(timestamps_ms: list[dict], sampling_rate: int) -> list[dict]:
        return [
            (seg["start"] / sampling_rate, seg["end"] / sampling_rate) 
            for seg in timestamps_ms
        ]
