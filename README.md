# Whisper Large V2 @ Inferless

## Overview
Fast and efficient speech-to-text service using OpenAI's Whisper Large V2 model, optimized for Inferless deployment.

## Features
- High-accuracy speech recognition
- Multiple audio format support (MP3, WAV, FLAC)
- Automatic format conversion
- GPU-accelerated inference
- Inferless-ready deployment

## Technical Requirements
- GPU: NVIDIA T4
- CUDA support
- FFmpeg for audio processing
- Python packages:
  - PyTorch
  - Transformers
  - Accelerate
  - Pydub

## Usage
Input format:
```json
{
    "audio_url": "https://example.com/audio.mp3"
}
```

## Environment Configuration
- Minimum replicas: 0
- Maximum replicas: 1
- Inference timeout: 180s
- Memory: 16GB
- GPU: T4

## Setup
- Clone the repository
- Install dependencies: ```pip install -r requirements.txt```
- Deploy to Inferless using provided ```inferless.yaml```

## License
This project uses OpenAI's Whisper model under the MIT license.
