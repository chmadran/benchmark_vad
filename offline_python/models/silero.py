from silero_vad import (load_silero_vad, get_speech_timestamps)
import torch 

class SileroVAD():

    def __init__(self, 
                 sampling_rate: int,
                 threshold: float = 0.3):
        self.sampling_rate = sampling_rate
        self.model = load_silero_vad()
        self.threshold = threshold

    def predict(self, wav):
        predict = self.model.audio_forward(wav, sr=self.sampling_rate)
        return predict
    
    def get_timestamps(self, wav):
        timestamps = get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=self.sampling_rate,
            threshold=self.threshold,
        )
        return timestamps
