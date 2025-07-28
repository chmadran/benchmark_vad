from pathlib import Path
from itertools import product
import json

DEFAULT_PARAM_GRIDS = {
    "webrtc": {
        "mode": [0, 1, 2, 3],  
        "window_size_samples": [0.01, 0.02, 0.03, 0.04], 
    },
    "silero": {
        "threshold": [0.2, 0.3, 0.4, 0.5, 0.6], 
    }
}

def load_param_grid_from_file(grid_file: Path):
    with open(grid_file, "r") as f:
        param_grids = json.load(f)
    return param_grids

def init_grid_search(args_grid_search: bool, args_params_grid_file: Path, args_models: list):
    param_grids = {}
    if args_grid_search:
        if args_params_grid_file:
            params = load_param_grid_from_file(args_params_grid_file)
        else:
            params = {model: DEFAULT_PARAM_GRIDS.get(model, {}) for model in args_models}
        for model in params:
            param_grids[model] = generate_model_param_grid(params[model])
    else:
        param_grids = {}
    return param_grids

def generate_model_param_grid(params: dict):
    if not params:
        return [{}]
    
    keys = list(params.keys())
    values = list(params.values())
    params_grid = []
    
    for combination in product(*values):
        param_dict = dict(zip(keys, combination))
        params_grid.append(param_dict)
    
    return params_grid