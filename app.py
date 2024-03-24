import json
import numpy as np
import torch
from transformers import pipeline

import requests

from pydub import AudioSegment
import subprocess

class InferlessPythonModel:
    """
    A class representing an Inferless Python Model.

    This class provides methods for converting audio files to MP3 format,
    initializing the model, performing inference on audio inputs, and finalizing the model.

    Attributes:
        generator: The automatic speech recognition pipeline generator.
    """

    def download_file(self, url):
        """
        Downloads a file from the given URL and saves it locally.

        Args:
            url (str): The URL of the file to download.

        Returns:
            str: The filename of the downloaded file.

        Raises:
            requests.HTTPError: If the download request fails.
        """
        local_filename = url.split('/')[-1]
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    #if chunk: 
                    f.write(chunk)
        return local_filename

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

                # Convert oga to wav using ffmpeg
                subprocess.run(['ffmpeg', '-i', file_path, 'temp.wav'])

                # Load the audio file
                audio = AudioSegment.from_file('temp.wav', format='wav')
                subprocess.run(['rm', 'temp.wav'])
                # Export the audio file in MP3 format
                audio.export("output.mp3", format="mp3")

                return "output.mp3"

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
        local = self.download_file(audio_url)
        file = self.convert_to_mp3(local)

        pipeline_output = self.generator(file)
        return {"transcribed_output": pipeline_output["text"]}

    def finalize(self):
        """
        Finalize the model.

        This method cleans up the model by setting the generator attribute to None.
        """
        self.generator = None
