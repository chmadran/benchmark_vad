from silero_vad import (load_silero_vad, get_speech_timestamps)
import torch 

#https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py
#TODO: Add input sanitization
class SileroVAD():

    def __init__(self, 
                 sampling_rate: int,
                 threshold: float = 0.3, 
                 window_size_samples : int = 512, 
                 neg_threshold: float = None,
                 visualize_probs: bool = False,
                 time_resolution: bool = 1, 
                 return_seconds: bool = False,
                 speech_pad_ms: int = 30,
                 min_silence_duration_ms: float = 100,
                 max_speech_duration_s: float = float('inf'),
                 min_speech_duration_ms: float = 250,
                 ):
        
        
        if sampling_rate not in {8000, 16000}:
            raise ValueError(f"Invalid sampling_rate: {sampling_rate}, must be 8kHz or 16kHz")

        if threshold < 0 or threshold > 1:
            raise ValueError(f"Invalid threshold: {threshold}, cannot be below 0 or above 1.")

        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.model = load_silero_vad()

        self.window_size_samples = window_size_samples
        self.neg_threshold = neg_threshold
        self.visualize_probs = visualize_probs
        self.time_resolution = time_resolution
        self.return_seconds = return_seconds
        self.speech_pad_ms = speech_pad_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.max_speech_duration_s = max_speech_duration_s
        self.min_speech_duration_ms = min_speech_duration_ms

    def get_confidence(self, wav):
        predict = self.model.audio_forward(wav, sr=self.sampling_rate)
        return predict
    
    def get_predictions(self, wav):
        timestamps = get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=self.sampling_rate,
            threshold=self.threshold,
            window_size_samples= self.window_size_samples,
            neg_threshold = self.neg_threshold, 
            visualize_probs = self.visualize_probs,
            return_seconds = self.return_seconds,
            speech_pad_ms = self.speech_pad_ms,
            min_silence_duration_ms = self.min_silence_duration_ms,
            max_speech_duration_s = self.max_speech_duration_s,
            min_speech_duration_ms = self.min_speech_duration_ms
        )
        return timestamps



    



