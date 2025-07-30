from pathlib import Path
from itertools import product
import json


DEFAULT_PARAMS = {
    "webrtc": {
        "mode": [0],  
        "window_size_samples": [0.01], 
    },
    "silero": {
        "threshold": [0.2], 
    },
    "pyannote": {
        "onset": [0.5], 
        "offset": [0.5],
        "min_duration_on": [0.0],
        "min_duration_off": [0.0]
    }
}

def load_config(config_file: Path, model):
    """
    Load the configuration for a specific model from a JSON config file.

    Args:
        config_file (Path): Path to the JSON configuration file.
        model (str): Name of the model whose parameters should be loaded.

    Returns:
        dict: Dictionary of parameters to test for the specified model.
    """
     
    params = DEFAULT_PARAMS[model]

    if config_file:
        try:
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            with open(config_file, "r") as f:
                params = json.load(f)
            if model not in params:
                raise KeyError(f"Model '{model}' not found in config file.")
            return params[model]

        except (FileNotFoundError, KeyError) as e:
            print(f"Error: {e} — using default parameters for '{model}'.")
    return params

def generate_experiments(params: dict):
    """
    Generate all combinations of parameters for hyperparameter search.

    Args:
        params (dict): Dictionary where keys are parameter names and values are lists of possible values.

    Returns:
        list[dict]: A list of dictionaries, each representing a unique combination of parameters.
    """
    if not params:
        return [{}]
    
    keys = list(params.keys())
    values = list(params.values())
    experiments = []
    
    for combination in product(*values):
        param_dict = dict(zip(keys, combination))
        experiments.append(param_dict)
    
    return experiments

def init_experiments_from_config(config_file: Path, models: list) -> dict:
    """
    Initialize list of experiments to run for each model based on config.

    Args:
        config_file (Path): Path to the configuration file.
        models (list): List of model names or instances.

    Returns:
        dict: Dictionary containing all experiment configurations.
    """

    params = {}
    experiments = {}

    for model in models:
        params = load_config(config_file, model)  
        experiments[model] = generate_experiments(params)

    return experiments
