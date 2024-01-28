import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import matplotlib.pyplot as plt
import soundfile as sf


def load_audio(file_path):
    """
    Load an audio file.

    Parameters:
    file_path (str): Path to the audio file.

    Returns:
    tuple: Sample rate (int) and audio data (numpy array).
    """
    sample_rate, audio = wav.read(file_path)
    return sample_rate, audio


def generate_spectrogram(audio, sample_rate):
    """
    Generate a spectrogram from an audio signal.

    Parameters:
    audio (numpy array): Audio data.
    sample_rate (int): Sample rate of the audio.

    Returns:
    tuple: Frequencies (numpy array), times (numpy array), and spectrogram (2D numpy array).
    """
    frequencies, times, spectrogram = scipy.signal.spectrogram(audio, fs=sample_rate)
    return frequencies, times, spectrogram


def plot_spectrogram(frequencies, times, spectrogram, title, use_log_scale=True):
    """
    Plot a spectrogram using a dB scale.

    Parameters:
    frequencies (numpy array): Frequencies for the spectrogram.
    times (numpy array): Time bins for the spectrogram.
    spectrogram (2D numpy array): Spectrogram data.
    title (str): Title for the plot.
    """
    # Convert to dB scale to avoid log of zero
    if use_log_scale:
        spectrogram = 10 * np.log10(np.abs(spectrogram) + 1e-10)

    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower',
               extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    if use_log_scale:
        plt.title(f"Audio {title} Spectrogram in Log Scale")
    else:
        plt.title(f"Audio {title} Spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.show()


def display_audio_spectrogram(audio, sample_rate, title, use_log_scale=True):
    """
    Displays a spectrogram of an audio signal and returns its frequency, time, and amplitude data.

    Parameters:
    - audio (numpy array): Audio data array.
    - sample_rate (int): Sample rate of the audio in Hertz.
    - title (str): Title for the spectrogram plot.
    - use_log_scale (bool, optional): Use logarithmic scale for amplitude. Defaults to True.

    Returns:
    - frequencies (numpy array): Frequencies present in the audio (Hz).
    - times (numpy array): Time segments for the spectrogram (seconds).
    - spectrogram (2D numpy array): Amplitude or power of frequencies at each time segment.

    Note:
    This function plots the spectrogram for visualization.
    """
    frequencies, times, spectrogram = generate_spectrogram(audio, sample_rate)
    plot_spectrogram(frequencies, times, spectrogram, title, use_log_scale=use_log_scale)

    return frequencies, times, spectrogram



def plot_centered_fft_magnitude(sample_rate, audio):
    """
    Compute and plot the centered magnitude of the Fourier Transform of an audio signal,
    highlighting the peak frequency and the conjunction frequencies at +/- 1060 Hz.

    Parameters:
    sample_rate (int): Sample rate of the audio.
    audio (numpy array): Audio data.
    """
    fft_result = np.fft.fft(audio)
    frequencies = np.fft.fftfreq(len(audio), d=1 / sample_rate)
    fft_result_shifted = np.fft.fftshift(fft_result)
    frequencies_shifted = np.fft.fftshift(frequencies)

    peak_index = np.argmax(np.abs(fft_result_shifted))
    peak_frequency = frequencies_shifted[peak_index]

    plt.figure(figsize=(12, 6))
    plt.plot(frequencies_shifted, np.abs(fft_result_shifted), color='blue')

    # Highlighting the conjunction frequencies
    conjunction_freq = int(np.abs(peak_frequency))
    plt.axvline(x=conjunction_freq, color='green', linestyle='--')
    plt.text(conjunction_freq, np.max(np.abs(fft_result_shifted)) * 0.9, f'{conjunction_freq} Hz', color='green',
             verticalalignment='bottom', fontsize=10)
    plt.axvline(x=-conjunction_freq, color='green', linestyle='--')
    plt.text(-conjunction_freq, np.max(np.abs(fft_result_shifted)) * 0.9, f'-{conjunction_freq} Hz', color='green',
             verticalalignment='bottom', fontsize=10)

    # Setting frequency labels as integers and enhancing readability
    freq_labels = np.arange(int(frequencies_shifted.min()), int(frequencies_shifted.max()) + 1, 500)
    plt.xticks(freq_labels, rotation=45)
    plt.xlim(frequencies_shifted.min(), frequencies_shifted.max())

    plt.ylabel('Magnitude')
    plt.xlabel('Frequency [Hz]')
    plt.title('Centered FFT Magnitude Spectrum')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_audio(file_path, sample_rate, audio):
    """
    Save an audio file.

    Parameters:
    file_path (str): Path where the audio file will be saved.
    sample_rate (int): Sample rate of the audio.
    audio (numpy array): Audio data to be saved.
    """
    sf.write(file_path, audio, sample_rate)


def find_peak_frequency(audio, sample_rate):
    """
    Find the frequency with the highest magnitude in the audio signal's Fourier Transform.

    Parameters:
    sample_rate (int): Sample rate of the audio.
    audio (numpy array): Audio data.

    Returns:
    float: Frequency of the peak magnitude.
    """
    fft_result = np.fft.fft(audio)
    frequencies = np.fft.fftfreq(len(audio), d=1 / sample_rate)
    fft_result_shifted = np.fft.fftshift(fft_result)
    frequencies_shifted = np.fft.fftshift(frequencies)

    peak_index = np.argmax(np.abs(fft_result_shifted))
    peak_frequency = frequencies_shifted[peak_index]
    return abs(peak_frequency)


def apply_peak_denoising_filter(audio):
    """
    Apply a denoising filter to the audio by zeroing out the peak frequency and its negative counterpart.

    Parameters:
    audio (numpy array): Audio data to be denoised.

    Returns:
    numpy array: Denoised audio data.
    """
    fft_result = np.fft.fft(audio)
    peak_index = np.argmax(np.abs(fft_result))
    neg_peak_index = len(fft_result) - peak_index

    fft_result[peak_index] = 0
    fft_result[neg_peak_index] = 0
    denoised_audio = np.fft.ifft(fft_result).real
    return denoised_audio


def exploration_of_the_audio(audio, sample_rate, title, use_log_scale=True):
    """
    Explore and display the characteristics of an audio signal.

    Parameters:
    audio (numpy array): Audio data.
    sample_rate (int): Sample rate of the audio.
    title_prefix (str): Prefix for the plot title.
    """
    display_audio_spectrogram(audio, sample_rate, title, use_log_scale=use_log_scale)
    plot_centered_fft_magnitude(sample_rate, audio)
    peak_freq = find_peak_frequency(audio, sample_rate)
    print(f"The frequency containing the peak is: {peak_freq:.2f} Hz")


def save_audio_file(audio_path, denoised_audio, sample_rate, file_name):
    """
    Save a denoised audio file.

    Parameters:
    audio_path (str): Original path of the audio file.
    denoised_audio (numpy array): Denoised audio data.
    sample_rate (int): Sample rate of the audio.
    file_name (str): Suffix for the new file name.

    Returns:
    str: Path of the saved denoised audio file.
    """
    output_path = audio_path.replace('.wav', file_name)
    save_audio(output_path, sample_rate, denoised_audio)
    return output_path


def q1(audio_path):
    """
    Process, denoise, and save a denoised version of audio Q1.

    Parameters:
    audio_path (str): Path to the audio file.
    """
    sample_rate, audio = load_audio(audio_path)

    exploration_of_the_audio(audio, sample_rate, 'Q1')

    denoised_audio = apply_peak_denoising_filter(audio)
    save_audio_file(audio_path, denoised_audio, sample_rate, '_denoised.wav')
    exploration_of_the_audio(denoised_audio, sample_rate, 'Denoised Q')


def plot_magnitude_in_problematic_range(audio_path):
    """
    Plot the summed magnitude in the problematic frequency range for each time window of the STFT.

    Parameters:
    audio_path (str): Path to the audio file.
    """
    sample_rate, audio = load_audio(audio_path)

    # Compute the STFT of the audio
    frequencies, times, Zxx = scipy.signal.stft(audio, fs=sample_rate, nperseg=1024)

    # Extract the magnitudes in the problematic frequency range (500-650 Hz)
    noise_band = (frequencies >= 500) & (frequencies <= 650)
    magnitudes_in_noise_band = np.abs(Zxx[noise_band, :])

    # Compute the summed magnitude in the problematic frequency range for each time window
    summed_magnitude_per_window = np.sum(magnitudes_in_noise_band, axis=0)

    # Plot the summed magnitude
    plt.figure(figsize=(10, 5))
    plt.plot(times, summed_magnitude_per_window)
    plt.title("Summed Magnitude in 500-650 Hz Range for Each Time Window")
    plt.xlabel("Time [sec]")
    plt.ylabel("Summed Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# def q2(audio_path):
#     """
#     Process and display the characteristics of audio Q2.
#
#     Parameters:
#     audio_path (str): Path to the audio file.
#     """
#     sample_rate, audio = load_audio(audio_path)
#     exploration_of_the_audio(audio, sample_rate, 'Q2')
def q2(audio_path, output_path):
    """
    Process, denoise, and save the denoised version of audio Q2 based on the STFT algorithm.

    Parameters:
    audio_path (str): Path to the audio file.
    output_path (str): Path to save the denoised audio.

    Returns:
    str: Path of the saved denoised audio file.
    """
    sample_rate, audio = load_audio(audio_path)
    # Display the spectrogram of the denoised audio for verification
    display_audio_spectrogram(audio, sample_rate, 'Denoised Q2')

    # Compute the STFT of the audio
    frequencies, times, Zxx = scipy.signal.stft(audio, fs=sample_rate, nperseg=1024)

    # Suppress noise in the 500-650 Hz range
    noise_band = (frequencies >= 500) & (frequencies <= 650)
    Zxx[noise_band, :] = 0

    # Compute the inverse STFT to get the denoised audio
    _, denoised_audio = scipy.signal.istft(Zxx, fs=sample_rate)

    # Save the denoised audio
    sf.write(output_path, denoised_audio, sample_rate)

    # Display the spectrogram of the denoised audio for verification
    display_audio_spectrogram(denoised_audio, sample_rate, 'Denoised Q2')

    return output_path


# if __name__ == '__main__':
    # q1_audio_path = '/Users/srashkovits/PycharmProjects/image/ex2/Exercise Inputs/q1.wav'
    # q2_audio_path = '/Users/srashkovits/PycharmProjects/image/ex2/Exercise Inputs/q2.wav'
    # q1(q1_audio_path)
    # q2(q2_audio_path, '/Users/srashkovits/PycharmProjects/image/ex2/Exercise Inputs/q2_denoised.wav')
    # plot_magnitude_in_problematic_range(q2_audio_path)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, istft

if __name__ == '__main__':
    # Load the audio file
    file_path = '/Users/srashkovits/PycharmProjects/image/ex2/Exercise Inputs/q2.wav'
    sample_rate, audio_array = wavfile.read(file_path)
    if audio_array.ndim == 2:  # Convert stereo to mono if necessary
        audio_array = audio_array.mean(axis=1)

    # Perform STFT
    n_fft = 1024  # Length of the FFT window
    hop_length = n_fft // 4  # Number of samples between successive frames
    frequencies, times, Zxx = stft(audio_array, fs=sample_rate, nperseg=n_fft, noverlap=hop_length)

    # Define a small constant to avoid log of zero
    epsilon = 1e-10

    # Plot the spectrogram with the constant added
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, frequencies, 20 * np.log10(np.abs(Zxx) + epsilon), shading='gouraud')
    plt.title('STFT Magnitude Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(format='%+2.0f dB')
    plt.ylim(0, sample_rate // 2)
    plt.show()

    # Analyzing the mean and variance of amplitude around 600 Hz for the original audio
    freq_band = (frequencies >= 590) & (frequencies <= 610)
    magnitude_band = np.abs(Zxx[freq_band, :])
    variance_band = np.var(magnitude_band, axis=0)

    # Plotting the variance over time for the original audio
    plt.figure(figsize=(10, 4))
    plt.plot(times, variance_band)
    plt.title("Variance of Amplitude around 600 Hz over Time (Original Audio)")
    plt.xlabel("Time [sec]")
    plt.ylabel("Variance of Amplitude")
    plt.axvline(x=2, color='r', linestyle='--')
    plt.axvline(x=4, color='r', linestyle='--')
    plt.show()

    # Zeroing out the specified frequency range in the STFT for noise reduction
    f_min = 597.66 - 10
    f_max = 597.66 + 10
    f_min_bin = int(np.round(f_min / (sample_rate / n_fft)))
    f_max_bin = int(np.round(f_max / (sample_rate / n_fft)))
    time_indices = (times >= 1.4) & (times <= 4)  # Adjusted time window
    Zxx[f_min_bin:f_max_bin + 1, time_indices] = 0

    # Reconstructing the audio
    _, reconstructed_audio = istft(Zxx, fs=sample_rate, nperseg=n_fft, noverlap=hop_length)
    reconstructed_audio = np.int16(reconstructed_audio / np.max(np.abs(reconstructed_audio)) * 32767)

    # Save the denoised audio
    output_file_path = 'denoised_audio.wav'
    wavfile.write(output_file_path, sample_rate, reconstructed_audio)

    # Perform STFT on the denoised audio
    frequencies_denoised, times_denoised, Zxx_denoised = stft(reconstructed_audio, fs=sample_rate, nperseg=n_fft, noverlap=hop_length)

    # Plot the spectrogram of the denoised audio
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times_denoised, frequencies_denoised, 20 * np.log10(np.abs(Zxx_denoised) + epsilon), shading='gouraud')
    plt.title('STFT Magnitude Spectrogram of Denoised Audio')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(format='%+2.0f dB')
    plt.ylim(0, sample_rate // 2)
    plt.show()

    # Analyzing the mean and variance of amplitude around 600 Hz for the denoised audio
    magnitude_band_denoised = np.abs(Zxx_denoised[freq_band, :])
    variance_band_denoised = np.var(magnitude_band_denoised, axis=0)

    # Plotting the variance over time for the denoised audio
    plt.figure(figsize=(10, 4))
    plt.plot(times_denoised, variance_band_denoised)
    plt.title("Variance of Amplitude around 600 Hz over Time (Denoised Audio)")
    plt.xlabel("Time [sec]")
    plt.ylabel("Variance of Amplitude")
    plt.axvline(x=2, color='r', linestyle='--')
    plt.axvline(x=4, color='r', linestyle='--')
    plt.show()
