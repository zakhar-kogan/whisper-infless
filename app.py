import json
import numpy as np
import torch
from transformers import pipeline

from pydub import AudioSegment

class InferlessPythonModel:
    """
    A class representing an Inferless Python Model.

    This class provides methods for converting audio files to MP3 format,
    initializing the model, performing inference on audio inputs, and finalizing the model.

    Attributes:
        generator: The automatic speech recognition pipeline generator.
    """

    def convert_to_mp3(self, file_path):
        """
        Convert an audio file to MP3 format.

        Args:
            file_path (str): The path to the audio file.

        Returns:
            str: The path to the converted MP3 file.
        """
        # Check the file extension
        file_extension = file_path.split(".")[-1]

        if file_extension.lower() not in ["mp3", "wav", "flac"]:
            # Load the audio file
            audio = AudioSegment.from_file(file_path, format=file_extension)

            # Define the new file path
            new_file_path = file_path.rsplit(".", 1)[0] + ".mp3"

            # Export the audio file in MP3 format
            pth = audio.export(new_file_path, format="mp3")

            return new_file_path

        # If the file is already in MP3 format, return the original file path
        return file_path

    def initialize(self):
        """
        Initialize the model.

        This method initializes the automatic speech recognition model.
        """
        self.generator = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v2",
            chunk_length_s=30,
            batch_size=24,
            stride_length_s=5,
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )

    def infer(self, inputs):
        """
        Perform inference on audio inputs.

        Args:
            inputs (dict): A dictionary containing the audio URL.

        Returns:
            dict: A dictionary containing the transcribed output.
        """
        audio_url = inputs["audio_url"]
        new_url = self.convert_to_mp3(audio_url)

        pipeline_output = self.generator(new_url)
        return {"transcribed_output": pipeline_output["text"]}

    def finalize(self):
        """
        Finalize the model.

        This method cleans up the model by setting the generator attribute to None.
        """
        self.generator = None
