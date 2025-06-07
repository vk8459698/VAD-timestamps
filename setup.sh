#!/bin/bash

echo " Setting up VAD Audio Processor..."

# Update system packages
echo " Updating system packages..."
sudo apt-get update -qq

# Install system dependencies for audio processing
echo " Installing system dependencies..."
sudo apt-get install -y libsndfile1 ffmpeg sox libsox-fmt-all

# Install Python dependencies
echo " Installing Python packages..."
pip install -r requirements.txt

echo " Setup complete!"
echo ""
echo " Usage: python app.py your_audio_file.wav"
echo " Example: python app.py 41918_song.wav"
