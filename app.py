import torch
import torchaudio
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import sys

def load_vad_model():
    """Load the Silero VAD model"""
    print("Loading Silero VAD model...")
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)
    return model, utils

def process_audio(audio_path, model, utils):
    """Process audio file with VAD"""
    print(f"Processing audio file: {audio_path}")
    
    # Extract utilities
    get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = utils
    
    # Read audio file
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
    except Exception as e:
        print(f"Error reading audio with torchaudio: {e}")
        print("Trying Silero's read_audio utility...")
        waveform = read_audio(audio_path, sampling_rate=16000)
        sample_rate = 16000
    
    # Ensure sample rate is 16000 Hz
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate} Hz to 16000 Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # Flatten to 1D array if needed
    if len(waveform.shape) > 1:
        waveform = waveform.squeeze()
    
    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(waveform, model, threshold=0.5, sampling_rate=16000)
    
    # Calculate statistics
    total_duration_seconds = len(waveform) / sample_rate
    speech_duration = sum([t['end'] - t['start'] for t in speech_timestamps]) / sample_rate
    speech_percentage = (speech_duration / total_duration_seconds) * 100
    
    print(f"\nResults:")
    print(f"Total audio duration: {total_duration_seconds:.2f} seconds")
    print(f"Speech detected: {speech_duration:.2f} seconds ({speech_percentage:.2f}%)")
    print(f"Number of speech segments: {len(speech_timestamps)}")
    
    return waveform, sample_rate, speech_timestamps, total_duration_seconds, speech_duration, speech_percentage

def save_results(audio_filename, speech_timestamps, sample_rate, total_duration_seconds, speech_duration, speech_percentage):
    """Save VAD results to JSON and create speech segments"""
    
    # Convert timestamps to seconds
    vad_timestamps_seconds = []
    for i, ts in enumerate(speech_timestamps):
        start_time = ts['start'] / sample_rate
        end_time = ts['end'] / sample_rate
        duration = end_time - start_time
        
        vad_timestamps_seconds.append({
            'segment_id': i + 1,
            'start_sample': ts['start'],
            'end_sample': ts['end'],
            'start_time_seconds': start_time,
            'end_time_seconds': end_time,
            'duration_seconds': duration
        })
    
    # Set reference point (first speech segment)
    reference_start_sample = 0
    reference_start_time = 0.0
    
    if speech_timestamps:
        reference_timestamp = speech_timestamps[0]
        reference_start_sample = reference_timestamp['start']
        reference_start_time = reference_start_sample / sample_rate
        
        print(f"\nReference Point (First Speech Segment):")
        print(f"Reference start time: {reference_start_time:.3f} seconds")
        print(f"Reference start sample: {reference_start_sample}")
    else:
        print(f"\nWarning: No speech detected in the audio file!")
    
    # Create VAD data structure
    vad_data = {
        'file_info': {
            'filename': audio_filename,
            'total_duration_seconds': total_duration_seconds,
            'sample_rate': sample_rate,
            'processing_timestamp': datetime.now().isoformat()
        },
        'speech_statistics': {
            'total_speech_duration_seconds': speech_duration,
            'speech_percentage': speech_percentage,
            'number_of_segments': len(speech_timestamps),
            'silence_duration_seconds': total_duration_seconds - speech_duration,
            'silence_percentage': 100 - speech_percentage
        },
        'reference_point': {
            'start_sample': reference_start_sample,
            'start_time_seconds': reference_start_time,
            'is_auto_selected': True,
            'description': 'Automatically selected first speech segment as reference'
        },
        'vad_segments': vad_timestamps_seconds,
        'raw_timestamps': speech_timestamps
    }
    
    # Save to JSON
    base_name = os.path.splitext(audio_filename)[0]
    vad_filename = f"vad_analysis_{base_name}.json"
    
    with open(vad_filename, 'w') as f:
        json.dump(vad_data, f, indent=2, default=str)
    
    print(f"\nVAD timestamps saved to: {vad_filename}")
    
    # Display detailed segment information
    print(f"\nDetailed Speech Segments:")
    print(f"{'Segment':<8} {'Start (s)':<10} {'End (s)':<10} {'Duration (s)':<12} {'Rel. Start (s)':<15}")
    print("-" * 65)
    
    for segment in vad_timestamps_seconds:
        relative_start = segment['start_time_seconds'] - reference_start_time
        print(f"{segment['segment_id']:<8} {segment['start_time_seconds']:<10.3f} "
              f"{segment['end_time_seconds']:<10.3f} {segment['duration_seconds']:<12.3f} "
              f"{relative_start:<15.3f}")
    
    return vad_filename, vad_data

def save_speech_segments(waveform, speech_timestamps, sample_rate, audio_filename):
    """Save individual speech segments as audio files"""
    print("\nExtracting speech segments...")
    
    # Create directory for segments
    base_name = os.path.splitext(audio_filename)[0]
    segments_dir = f"speech_segments_{base_name}"
    
    if not os.path.exists(segments_dir):
        os.makedirs(segments_dir)
    
    # Save each segment
    for i, ts in enumerate(speech_timestamps):
        segment = waveform[ts['start']:ts['end']]
        segment_filename = f"{segments_dir}/segment_{i+1}.wav"
        torchaudio.save(segment_filename, segment.unsqueeze(0), sample_rate)
        duration = (ts['end'] - ts['start']) / sample_rate
        print(f"Saved segment {i+1}: {duration:.2f} seconds -> {segment_filename}")
    
    return segments_dir

def create_visualization(waveform, speech_timestamps, sample_rate, audio_filename):
    """Create and save visualization plots"""
    print("\nCreating visualization...")
    
    base_name = os.path.splitext(audio_filename)[0]
    
    # Set reference point
    reference_start_time = 0.0
    if speech_timestamps:
        reference_start_time = speech_timestamps[0]['start'] / sample_rate
    
    plt.figure(figsize=(15, 8))
    
    # Plot original waveform
    plt.subplot(3, 1, 1)
    time_axis = np.arange(len(waveform)) / sample_rate
    plt.plot(time_axis, waveform.numpy())
    if speech_timestamps:
        plt.axvline(x=reference_start_time, color='red', linestyle='--',
                    label=f'Reference Point ({reference_start_time:.3f}s)', linewidth=2)
        plt.legend()
    plt.title('Original Audio Waveform with Reference Point')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    
    # Plot VAD markers
    plt.subplot(3, 1, 2)
    vad_mask = torch.zeros_like(waveform)
    for ts in speech_timestamps:
        vad_mask[ts['start']:ts['end']] = 1
    plt.plot(time_axis, vad_mask.numpy())
    if speech_timestamps:
        plt.axvline(x=reference_start_time, color='red', linestyle='--',
                    label=f'Reference Point', linewidth=2)
        plt.legend()
    plt.title('Voice Activity Detection')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speech Detected')
    plt.yticks([0, 1], ['No', 'Yes'])
    
    # Plot timeline relative to reference point
    plt.subplot(3, 1, 3)
    relative_time_axis = time_axis - reference_start_time
    plt.plot(relative_time_axis, vad_mask.numpy())
    plt.axvline(x=0, color='red', linestyle='--', label='Reference Point (t=0)', linewidth=2)
    plt.legend()
    plt.title('Speech Detection Timeline (Relative to Reference Point)')
    plt.xlabel('Time relative to reference (seconds)')
    plt.ylabel('Speech Detected')
    plt.yticks([0, 1], ['No', 'Yes'])
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"vad_visualization_{base_name}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_filename}")
    plt.close()
    
    return plot_filename

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python app.py <audio_file_path>")
        print("Example: python app.py 41918_song.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found!")
        sys.exit(1)
    
    try:
        # Load VAD model
        model, utils = load_vad_model()
        
        # Process audio
        waveform, sample_rate, speech_timestamps, total_duration, speech_duration, speech_percentage = process_audio(audio_path, model, utils)
        
        # Save results
        audio_filename = os.path.basename(audio_path)
        vad_filename, vad_data = save_results(audio_filename, speech_timestamps, sample_rate, total_duration, speech_duration, speech_percentage)
        
        # Save speech segments
        segments_dir = save_speech_segments(waveform, speech_timestamps, sample_rate, audio_filename)
        
        # Create visualization
        plot_filename = create_visualization(waveform, speech_timestamps, sample_rate, audio_filename)
        
        print(f"\n VAD Analysis Complete!")
        print(f" Files created:")
        print(f"   - VAD data: {vad_filename}")
        print(f"   - Speech segments: {segments_dir}/")
        print(f"   - Visualization: {plot_filename}")
        
        if speech_timestamps:
            print(f" Reference point set at: {speech_timestamps[0]['start'] / sample_rate:.3f} seconds")
        
    except Exception as e:
        print(f" Error processing audio: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
