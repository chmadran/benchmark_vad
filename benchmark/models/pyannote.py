import torch
import warnings
import logging
import numpy as np
from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

HF_TOKEN = "to add"

class PyAnnoteVAD():
    
    def __init__(self, 
                 onset: float = 0.5,
                 offset: float = 0.5,
                 min_duration_on: float = 0.0,
                 min_duration_off: float = 0.0,
                 ):
        model = Model.from_pretrained("pyannote/segmentation", 
                                    use_auth_token=HF_TOKEN)
        
        self.hyperparameters = {
            "onset": onset,
            "offset": offset,
            "min_duration_on": min_duration_on,
            "min_duration_off": min_duration_off
        }
        pipeline = VoiceActivityDetection(segmentation=model)

        pipeline.instantiate(self.hyperparameters)
        self.pipeline = pipeline


    def predict(self, wav_file):
        
        output = self.pipeline(wav_file)
        return [(segment.start, segment.end) for segment in output.get_timeline().support()]

    def add_start_end(self, preds: list[float, float]):
        formatted_preds = []

        for pred in preds:
            formatted_preds.append({
            "start" : pred[0],
            "end": pred[1]
            })

        return formatted_preds