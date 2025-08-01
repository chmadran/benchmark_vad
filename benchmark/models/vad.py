from abc import ABC, abstractmethod
from models.silero import SileroVAD
from models.webrtc import WebRTCVAD
from models.pyannote import PyAnnoteVAD

class VADFactory():
    @staticmethod
    def create(model_name: str, model_params: dict, audio: dict):
        if model_name == "silero":
            return SileroVAD(audio, **model_params)

        elif model_name == "webrtc":
            return WebRTCVAD(audio, **model_params)

        elif model_name == "pyannote":
            return PyAnnoteVAD(audio, **model_params)

        else:
            raise ValueError(f"Unknown model: {model_name}")
