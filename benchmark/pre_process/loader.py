import numpy as np
from scipy.io import wavfile
import json
import torchaudio
from pathlib import Path
import torch
from typing import Union

def convert_to_mono(signal: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(signal, torch.Tensor):
        if signal.dim() == 2:
            return signal.mean(dim=0)
        return signal
    else:  
        if signal.ndim == 2:
            return signal.mean(axis=1)
        return signal

def load_labels(path_to_json):
    with open(path_to_json, 'r') as f:
        data = json.load(f)

    speech_segments = []

    for task in data:
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                if "Speech" in result["value"]["labels"]:
                    start = result["value"]["start"]
                    end = result["value"]["end"]
                    speech_segments.append((start, end))

    return speech_segments

def load_audio(path_audio, label_path):
    sr, wav = wavfile.read(path_audio)
    wav_tensor, _ = torchaudio.load(path_audio)
    wav = convert_to_mono(wav)
    wav_tensor = convert_to_mono(wav_tensor)
    wav_bytes = wav.tobytes()
    labels = load_labels(label_path)

    audio_obj = {
        "signal": wav,
        "signal_bytes": wav_bytes,
        "signal_tensor": wav_tensor,
        "sample_rate": sr,
        "duration": len(wav) / sr,
        "file_path": str(path_audio),
        "labels": labels
    }
    return audio_obj

