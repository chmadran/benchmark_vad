# TODO 

* TODO: Add Tan https://github.com/ten-framework/ten-vad
* TODO: Post processing is shit, should add cool plotting
* TODO: Look further into each chosen model's parameters to create a cool (and full) grid search config file 

# Benchark Voice Activity Detectors 

This repository provides a benchmarking pipeline to evaluate different Voice Activity Detection (VAD) models on pre-annotated audio samples. The models currently supported are : 

* Silero
* WebRTC
* PyAnnote

## How to run 

Install dependencies : `pip install -r requirements.txt`
Run a benchmark : `python main.py --models silero webrtc --audio_files <../data/> --log_dir <../logs>`

Run with grid search : 

```
python main.py --models webrtc silero 
--audio_files <../data/> \
--log_dir <../logs/> \
--grid_search \
--grid_file <grid_search/config.json>`
```

