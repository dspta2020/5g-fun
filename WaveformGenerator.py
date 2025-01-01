import numpy as np 
import scipy.fft

class WaveformGenerator:

    def __init__(self, fs, dur):
        
        self.fs = fs 
        self.dur = dur 
        self.num_samples = int(self.dur * self.fs)

    def make_lfm_by_startstop(self, f0, f1):

        bw = f1 - f0
        slope = bw / self.dur
        t = np.arange(self.num_samples) / self.fs

        return np.exp(1j * 2*np.pi * (1/2*slope*t**2 + f0*t))
    
    def make_lfm_by_slope(self, f0, slope):

        t = np.arange(self.num_samples) / self.fs
        
        return np.exp(1j * 2*np.pi * (1/2*slope*t**2 + f0*t))


def main():
    
    fs = 1e3 
    dur = 3 
    num_samples = dur * fs
    t = np.arange(num_samples) / fs

    Gen = WaveformGenerator(fs, dur)

    noise = Gen.make_band_limited_noise(75, 8192)
    Lo = np.exp(1j * 2 * np.pi * 233 * t).reshape(-1,1)

    waveform = Lo * noise

    fig = plt.figure()
    window_size = 256
    stft = scipy.signal.ShortTimeFFT(np.hamming(window_size), hop=window_size//2, fs=fs, fft_mode='centered', mfft=8192)
    mag = 20*np.log10(abs(stft.spectrogram(waveform.flatten())))
    freqs = np.arange(-stft.f_pts/2, stft.f_pts/2) * (fs/stft.f_pts)
    plt.pcolormesh(np.linspace(0, len(waveform)/fs*1e3, mag.shape[1]), freqs*1e-6, mag, shading='auto')
    plt.ylabel('Frequency [MHz]')
    plt.xlabel('Time [msec]')
    plt.colorbar(label='Magnitude (dB)')
    plt.show()



if __name__ == "__main__":

    from pathlib import Path
    import time

    import numpy as np 
    import matplotlib.pyplot as plt 
    import scipy.fft

    from scratch.FilterGenerator import FilterGenerator

    print(f"Running File: {Path(__file__).name}")

    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")