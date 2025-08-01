import webrtcvad
import collections
import numpy as np
from models.base import BaseVAD


#https://github.com/wiseman/py-webrtcvad/blob/master/example.py
#TODO: add sliding window from wiseman

class WebRTCVAD(BaseVAD):
    def __init__(self,
                    audio: dict,
                    mode: int,
                    experiment_id: int,
                    window_size_sample: float = 0.03,
                    padding_duration_ms: int = 300, 
                    threshold_trigger: float = 0.9
                    ): 

        self.audio = audio
        super().__init__(audio, experiment_id=experiment_id)

        if audio["sample_rate"] not in {8000, 16000, 32000, 44000}:
                raise ValueError(f"Invalid sampling_rate: {audio["sample_rate"]}, must be 8kHz, 16kHz, 32kHz or 44kHz.")
        if window_size_sample not in {0.01, 0.02, 0.03}:
            raise ValueError(f"Invalid window_size_samples: {window_size_sample}, must be 10, 20, or 30ms.")
        if mode not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid mode: {mode}, must be 0, 1, 2, or 3.")
        if padding_duration_ms > 500 or padding_duration_ms < 0:
            raise ValueError(f"Invalid padding_duration_ms: {padding_duration_ms}, must be reasonable (less than 500ms).") 
        if threshold_trigger > 1 or threshold_trigger < 0.1:
            raise ValueError(f"Invalid threshold_trigger: {threshold_trigger}, must be between 1 and 0.1.") 

        self.sampling_rate = audio["sample_rate"]
        self.experiment_id = experiment_id
        self.window_size_sample = window_size_sample
        self.padding_duration_ms = padding_duration_ms
        self.threshold_trigger = threshold_trigger
        self.model = webrtcvad.Vad()
        self.model.set_mode(mode)

    def get_predictions(self, wav, wav_bytes):
        segments = []
        bytes_per_sample = 2 
        samples_per_window = int(self.window_size_sample * self.sampling_rate)

        for i, start in enumerate(np.arange(0, len(wav), samples_per_window)):
            stop = min(start + samples_per_window, len(wav))
            loc_raw_sample = wav_bytes[start * bytes_per_sample: stop * bytes_per_sample]
            try:
                is_speech = self.model.is_speech(loc_raw_sample, 
                                    sample_rate = self.sampling_rate)
                segments.append(dict(
                        start = start,
                        stop = stop,
                        is_speech = is_speech))
            except Exception as e:
                print(f"Failed for step {i}, reason: {e}")
        return segments
    
    def merge_speech_segments(self, segments):
        merged = []
        current_start = None
        for seg in segments:
            if seg["is_speech"]:
                if current_start is None:
                    current_start = seg["start"]
                current_end = seg["stop"]
            else:
                if current_start is not None:
                    merged.append({"start": current_start, "end": current_end})
                    current_start = None
        if current_start is not None:
            merged.append({"start": current_start, "end": current_end})
        return merged

    def predict(self) -> dict:
        frames = self.get_predictions(self.audio["signal"], self.audio["signal_bytes"])
        frames = self.merge_speech_segments(frames)
        frames_s = BaseVAD.convert_ms_to_seconds(frames, self.audio["sample_rate"])
        metrics = BaseVAD.match_segments(frames_s, self.audio["labels"], threshold=0.5)
        return {
            "preds_ms": frames,
            "preds_s" : frames_s,
            "metrics" : metrics
        }