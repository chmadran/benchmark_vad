import webrtcvad
import numpy as np

#https://github.com/wiseman/py-webrtcvad/blob/master/example.py
#TODO: add sliding window from wiseman

class WebRTCVAD():
    def __init__(self,
                    mode: int,
                    sampling_rate: int,
                    window_size_sample: float = 0.03,
                    ): 

        if sampling_rate not in {8000, 16000, 32000, 44000}:
                raise ValueError(f"Invalid sampling_rate: {sampling_rate}, must be 8kHz, 16kHz, 32kHz or 44kHz.")
        if window_size_sample not in {0.01, 0.02, 0.03}:
            raise ValueError(f"Invalid window_size_samples: {window_size_sample}, must be 10, 20, or 30.")
        if mode not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid mode: {mode}, must be 0, 1, 2, or 3.")

        self.window_size_sample = window_size_sample
        self.sampling_rate = sampling_rate
        self.model = webrtcvad.Vad()
        self.model.set_mode(mode)

    def predict(self, wav, wav_bytes):
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

    