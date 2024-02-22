import json
import numpy as np
import torch
from transformers import pipeline

from pydub import AudioSegment

class InferlessPythonModel:

    def convert_to_mp3(self, file_path):
        # Check the file extension
        file_extension = file_path.split('.')[-1]

        if file_extension.lower() not in ['mp3', 'wav', 'flac']:
            # Load the audio file
            audio = AudioSegment.from_file(file_path, format=file_extension)

            # Define the new file path
            new_file_path = file_path.rsplit('.', 1)[0] + '.mp3'

            # Export the audio file in MP3 format
            audio.export(new_file_path, format='mp3')

            return new_file_path

        # If the file is already in MP3 format, return the original file path
        return file_path

    def initialize(self):
        self.generator = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v2", chunk_length_s=30, batch_size=24, stride_length_s=5,
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )

    def infer(self, inputs):
        audio_url = inputs["audio_url"]
        new_url = self.convert_to_mp3(audio_url)

        pipeline_output = self.generator(new_url)
        return {"transcribed_output": pipeline_output["text"] }

    def finalize(self):
        self.generator = None
