from pathlib import Path
import os
import random
import math

DEFAULT_DATASET = "../data"
SEED = 42

#TODO: nb There is no input sanitization on whether the data being added is .wav and .json
def split_dataset_for_grid_search(path_audio_files: Path, 
                                  grid_search_dataset_percentage: int,
                                  shuffle=False, 
                                  include_grid_in_predict=False) -> dict:
    
    inference_dataset = []
    grid_search_dataset = []

    if not path_audio_files:
        path_audio_files =  DEFAULT_DATASET
    
    raw_dir = os.path.join(path_audio_files, "raw")
    label_dir = os.path.join(path_audio_files, "labels")

    raw_files = os.listdir(raw_dir)
    label_files = os.listdir(label_dir)
    for raw_file in raw_files:
        raw_name, _ = os.path.splitext(raw_file)
        for label_file in label_files:
            label_name, _ = os.path.splitext(label_file)
            if raw_name == label_name:
                full_raw_path = os.path.join(raw_dir, raw_file)
                full_label_path = os.path.join(label_dir, label_file)
                inference_dataset.append([full_raw_path, full_label_path])
    if shuffle : 
        random.seed(SEED)
        random.shuffle(inference_dataset)
    
    if grid_search_dataset_percentage > 0:
        amount = math.ceil(grid_search_dataset_percentage * len(inference_dataset) / 100)
        grid_search_dataset = inference_dataset[:amount]
        if not include_grid_in_predict:
            inference_dataset = inference_dataset[amount:]

    return grid_search_dataset, inference_dataset

