import webrtcvad
import numpy as np

class WebRTCVAD():
    def __init__(self,
                    sampling_rate: int,
                    mode: int,
                    window_size_samples: float = 0.03, #30ms per default, can be 10, 20 or 30
                    ): 

        self.sampling_rate = sampling_rate
        self.window_size_samples = window_size_samples
        self.model = webrtcvad.Vad()
        self.model.set_mode(mode)

    def predict(self, y, wav):
        segments = []
        bytes_per_sample = 2 
        samples_per_window = int(self.window_size_samples * self.sampling_rate)

        for i, start in enumerate(np.arange(0, len(y), samples_per_window)):
            stop = min(start + samples_per_window, len(y))
            loc_raw_sample = wav[start * bytes_per_sample: stop * bytes_per_sample]
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
        # Catch final segment
        if current_start is not None:
            merged.append({"start": current_start, "end": current_end})
        return merged
