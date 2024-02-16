import json
import numpy as np
import torch
from transformers import pipeline

class InferlessPythonModel:

    def initialize(self):
        self.generator = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v2", chunk_length_s=30, batch_size=24, stride_length_s=5,
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )

    def infer(self, inputs, tstamps: bool = False):
        audio_url = inputs["audio_url"]
        pipeline_output = self.generator(audio_url, return_timestamps=tstamps)
        if tstamps:
            return {"transcribed_output": pipeline_output["text"], "timestamps": pipeline_output["timestamp"]}
        else:
            return {"transcribed_output": pipeline_output["text"] }

    def finalize(self):
        self.generator = None
