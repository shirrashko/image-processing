import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from matplotlib import pyplot as plt

# Default window function and parameters for the spectrogram
WINDOW_SIZE_FACTOR = 25
OVERLAP_FACTOR = 2.5


def transform_to_spectrogram(audio, rate, window_size=0, overlap=0):
    """
    Converts audio data into a spectrogram using STFT.

    Parameters:
    - audio: Audio data array.
    - rate: Sample rate of the audio.
    - window_size: Size of each segment for STFT.
    - overlap: Overlap size between segments.

    Returns:
    - frequencies: Frequencies for each segment.
    - times: Time points for each segment.
    - spectrogram: Spectrogram data (magnitude).
    """
    if window_size == 0:
        window_size = rate / WINDOW_SIZE_FACTOR
    window_size = int(min(len(audio), max(1, round(window_size))))

    if overlap == 0:
        overlap = window_size / OVERLAP_FACTOR
    overlap = int(min(window_size, max(0, round(overlap))))

    freqs, times, spec_data = spectrogram(
        audio, fs=rate, nperseg=window_size, noverlap=overlap, detrend=False, mode='magnitude'
    )
    return freqs, times, spec_data


# Load audio data
audio_path = '/Users/srashkovits/PycharmProjects/image/ex2/Exercise Inputs/q2.wav'
sample_rate, audio_data = wavfile.read(audio_path)

# Generate spectrogram
freqs, times, spec_data = transform_to_spectrogram(audio_data, sample_rate)

# Plot the spectrogram
plt.pcolormesh(times, freqs, spec_data, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Audio Spectrogram')
plt.colorbar(label='Intensity')
plt.show()
