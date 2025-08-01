import torch
import os
import warnings
import logging
import numpy as np
from models.base import BaseVAD
from pyannote.audio import Model, Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class PyAnnoteVAD(BaseVAD):
    
    def __init__(self, 
                 audio: dict,
                 experiment_id: int,
                 onset: float = 0.5,
                 offset: float = 0.5,
                 min_duration_on: float = 0.0,
                 min_duration_off: float = 0.0,
                 ):
        
        self.audio = audio
        super().__init__(audio, experiment_id=experiment_id)

        model = Model.from_pretrained("pyannote/segmentation", 
                                    use_auth_token=os.environ['HF_TOKEN'])
        
        self.hyperparameters = {
            "onset": onset,
            "offset": offset,
            "min_duration_on": min_duration_on,
            "min_duration_off": min_duration_off
        }
        pipeline = VoiceActivityDetection(segmentation=model)

        pipeline.instantiate(self.hyperparameters)
        self.pipeline = pipeline
        self.experiment_id = experiment_id

    def add_start_end(self, preds: list[float, float]):
        formatted_preds = []

        for pred in preds:
            formatted_preds.append({
            "start" : pred[0],
            "end": pred[1]
            })

        return formatted_preds

    def predict(self) -> dict:
        preds = self.pipeline(self.audio["file_path"])
        frames = [(segment.start, segment.end) for segment in preds.get_timeline().support()]
        metrics = BaseVAD.match_segments(frames, self.audio["labels"], threshold=0.5)

        return {
            "preds_s": frames,
            "metrics": metrics
        }
    
