import webrtcvad
import collections
import numpy as np

#https://github.com/wiseman/py-webrtcvad/blob/master/example.py
#TODO: add sliding window from wiseman

class WebRTCVAD():
    def __init__(self,
                    mode: int,
                    sampling_rate: int,
                    window_size_sample: float = 0.03,
                    padding_duration_ms: int = 300, 
                    threshold_trigger: float = 0.9
                    ): 

        if sampling_rate not in {8000, 16000, 32000, 44000}:
                raise ValueError(f"Invalid sampling_rate: {sampling_rate}, must be 8kHz, 16kHz, 32kHz or 44kHz.")
        if window_size_sample not in {0.01, 0.02, 0.03}:
            raise ValueError(f"Invalid window_size_samples: {window_size_sample}, must be 10, 20, or 30ms.")
        if mode not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid mode: {mode}, must be 0, 1, 2, or 3.")
        if padding_duration_ms > 500 or padding_duration_ms < 0:
            raise ValueError(f"Invalid padding_duration_ms: {padding_duration_ms}, must be reasonable (less than 500ms).") 
        if threshold_trigger > 1 or threshold_trigger < 0.1:
            raise ValueError(f"Invalid threshold_trigger: {threshold_trigger}, must be between 1 and 0.1.") 

        self.window_size_sample = window_size_sample
        self.sampling_rate = sampling_rate
        self.padding_duration_ms = padding_duration_ms
        self.threshold_trigger = threshold_trigger
        self.model = webrtcvad.Vad()
        self.model.set_mode(mode)

    def predict(self, wav, wav_bytes):
        segments = []
        bytes_per_sample = 2 
        samples_per_window = int(self.window_size_sample * self.sampling_rate)

        for i, start in enumerate(np.arange(0, len(wav), samples_per_window)):
            stop = min(start + samples_per_window, len(wav))
            loc_raw_sample = wav_bytes[start * bytes_per_sample: stop * bytes_per_sample]
            try:
                is_speech = self.model.is_speech(loc_raw_sample, 
                                    sample_rate = self.sampling_rate)
                segments.append(dict(
                        start = start,
                        stop = stop,
                        is_speech = is_speech))
            except Exception as e:
                print(f"Failed for step {i}, reason: {e}")
        return segments
    
    # def predict_sliding_window(self, wav_bytes):
    #     """
    #     Runs VAD with sliding window and padded buffering over wav bytes.

    #     Returns: list of segments with start/end sample indices:
    #     [{"start": start_sample, "end": end_sample}, ...]
    #     """
        
    #     bytes_per_sample = 2
    #     frame_size = int(self.sampling_rate * self.window_size_sample) * bytes_per_sample
    #     samples_per_frame = int(self.sampling_rate * self.window_size_sample)
    #     num_padding_frames = int(self.padding_duration_ms / self.window_size_sample * 1000)

    #     ring_buffer = collections.deque(maxlen=num_padding_frames)
    #     triggered = False
    #     voiced_frames = []
    #     segments = []

    #     frames = [
    #         wav_bytes[i:i + frame_size]
    #         for i in range(0, len(wav_bytes), frame_size)
    #         if len(wav_bytes[i:i + frame_size]) == frame_size
    #     ]

    #     for i, frame_bytes in enumerate(frames):
    #         is_speech = self.model.is_speech(frame_bytes, self.sampling_rate)

    #         if not triggered:
    #             ring_buffer.append((i, frame_bytes, is_speech))
    #             num_voiced = len([f for _, _, speech in ring_buffer if speech])
    #             if num_voiced > self.threshold_trigger * ring_buffer.maxlen:
    #                 triggered = True
    #                 # start of voiced segment is index of first frame in ring buffer
    #                 segment_start_frame = ring_buffer[0][0]
    #                 for f in ring_buffer:
    #                     voiced_frames.append(f)
    #                 ring_buffer.clear()
    #         else:
    #             voiced_frames.append((i, frame_bytes, is_speech))
    #             ring_buffer.append((i, frame_bytes, is_speech))
    #             num_unvoiced = len([f for _, _, speech in ring_buffer if not speech])
    #             if num_unvoiced > self.threshold_trigger * ring_buffer.maxlen:
    #                 triggered = False
    #                 segment_end_frame = i
    #                 # convert frame indices to sample indices for start and end
    #                 segments.append({
    #                     "start": segment_start_frame * samples_per_frame,
    #                     "end": segment_end_frame * samples_per_frame
    #                 })
    #                 ring_buffer.clear()
    #                 voiced_frames = []

    #     if triggered:
    #         segments.append({
    #             "start": segment_start_frame * samples_per_frame,
    #             "end": len(frames) * samples_per_frame
    #         })
    #     return segments
    
    def merge_speech_segments(self, segments):
        merged = []
        current_start = None
        for seg in segments:
            if seg["is_speech"]:
                if current_start is None:
                    current_start = seg["start"]
                current_end = seg["stop"]
            else:
                if current_start is not None:
                    merged.append({"start": current_start, "end": current_end})
                    current_start = None
        if current_start is not None:
            merged.append({"start": current_start, "end": current_end})
        return merged

    