class FilterGenerator:

    def __init__(self, cutoff, sampling_rate, order):
        
        self.fc = cutoff
        self.fs = sampling_rate
        self.N = order

        self.fc_normalized = self.fc / self.fs
        self.midpoint = (self.N - 1) / 2

    def make_low_pass_filter(self):
        
        # filter response
        h = np.zeros((self.N, 1))

        for n in np.arange(self.N):
            if n != self.midpoint:
                # sinc function filter formula 
                h[n] = np.sin(2 * np.pi * self.fc_normalized * (n - self.midpoint)) / (np.pi * (n - self.midpoint))
            else:
                h[n] = 2 * self.fc_normalized
        
        return h

def makeWhiteNoise(len, amplitdue):
    return amplitdue * (np.random.randn(len,1) + 1j * np.random.randn(len,1)) / np.sqrt(2)

def main():

    fs = 1e3 
    dur = 3 
    num_samples = dur * fs
    t = np.arange(num_samples) / fs

    sig = np.exp(1j * 2 * np.pi * 72 * t).reshape(-1,1)
    sig2 = np.exp(1j * 2 * np.pi * 27 * t).reshape(-1,1)
    noise = makeWhiteNoise(len(sig), 0.2)
    sig += (noise + sig2)

    # setup fft
    nFFT = 2**16
    freqs = np.arange(-nFFT/2, nFFT/2) * (fs/nFFT)

    # make a filter to apply in the frequency domain
    filter = FilterGenerator(100, fs, nFFT).make_low_pass_filter()
    
    # fft data
    sig_F = scipy.fft.fftshift(scipy.fft.fft(sig, nFFT, 0), 0)
    filter_F = scipy.fft.fftshift(scipy.fft.fft(filter, nFFT, 0), 0)

    # apply filter
    filtered_data_F = sig_F * filter_F
    filtered_data = scipy.fft.ifft(scipy.fft.ifftshift(filtered_data_F, 0), nFFT, 0)
    filtered_data = filtered_data[nFFT//2:nFFT//2 + len(sig)]

    fig = plt.figure()
    window_size = 256
    stft = scipy.signal.ShortTimeFFT(np.hamming(window_size), hop=window_size//2, fs=fs, fft_mode='centered', mfft=8192)
    mag = 20*np.log10(abs(stft.spectrogram(filtered_data.flatten())))
    freqs = np.arange(-stft.f_pts/2, stft.f_pts/2) * (fs/stft.f_pts)

    plt.pcolormesh(np.linspace(0, len(filtered_data)/fs, mag.shape[1]), freqs, mag, shading='auto')
    plt.ylabel('Frequency [MHz]')
    plt.xlabel('Time [msec]')
    plt.colorbar(label='Magnitude (dB)')
    plt.show()


if __name__ == "__main__":

    import time 
    from pathlib import Path

    import numpy as np 
    import matplotlib.pyplot as plt
    import scipy

    print(f"Running File: {Path(__file__).name}")

    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")