import psutil
import os
import json
import librosa
import torchaudio
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from scipy.io import wavfile

def process_data(path_audio):
    sr, wav = wavfile.read(path_audio)
    wav_tensor = torchaudio.load(path_audio)
    assert wav.ndim == 1, "Audio must be mono." #TODO: add downmixing?
    wav_bytes = wav.tobytes()
    return sr, wav, wav_bytes, wav_tensor

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

def convert_predictions_to_seconds(predictions: list, sampling_rate: int):
    return [
        (seg["start"] / sampling_rate, seg["end"] / sampling_rate) 
        for seg in predictions
    ]


def plot_waveform_ground_truth(audio_path, preds, ground_truth):
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    times = np.linspace(0, duration, len(y))
    nb_models = len(preds)

    fig, axs = plt.subplots(2 + nb_models, 1, figsize=(14, 2 + 2 * nb_models), sharex=True,
                            gridspec_kw={'height_ratios': [3] + [0.5] * (1 + nb_models)})

    fig.suptitle("Waveform with Ground Truth and Predictions", fontsize=14)

    # Plot waveform
    axs[0].plot(times, y, color='blue', linewidth=1)
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Normalized Waveform")

    # Ground Truth
    for (start, end) in ground_truth:
        axs[1].axvspan(start, end, color='green', alpha=0.6)
    axs[1].set_ylabel("labels")
    axs[1].set_yticks([])

    # Predictions per model
    for i, (model_name, pred_data) in enumerate(preds.items()):
        ax = axs[2 + i]
        for (start, end) in pred_data["preds_s"]:
            ax.axvspan(start, end, color='orange', alpha=0.6)
        ax.set_ylabel(model_name)
        ax.set_yticks([])

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  
    return mem_bytes / (1024 ** 2)