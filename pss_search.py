from pathlib import Path

import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal

from DataReader import DataReader
from WaveformGenerator import WaveformGenerator
from FreqGscnMapper import FreqGscnMapper
from ssb_testing import make_pss_waveform


def main():
    
    # get the path to the data and read in the file
    path_to_data = Path(__file__).parent / 'data' / '954_7680KSPS_srsRAN_Project_gnb_short.txt'

    # call the data reader and grab the data
    reader = DataReader(path_to_file=path_to_data)
    data = reader.samples
    fs = reader.fs
    fc = reader.fc
    num_samples = reader.num_samples
    time_vec = reader.time_vec
    dur = reader.dur

    # make the data a bit easier to handle
    # there should be a PSS in here
    start_sample = round(0.008 * fs)
    end_sample = round(0.01 * fs)
    data = data[start_sample:end_sample]

    # so in theory we can check for GSCNs 
    bw = fs // 2
    upper_limit = fc + bw
    lower_limit = fc - bw

    # the PSS is transmitted on these GSCN frequencies so search the frequency space for candidate channels
    mapper = FreqGscnMapper()
    freq_search_space = np.arange(lower_limit, upper_limit, 0.01e6)
    gscn_list = np.round(mapper.freq_to_gscn(freq_list=freq_search_space))
    freq_list = np.unique(mapper.gscn_to_freq(gscn_list=gscn_list))

    if 1:
        fig = plt.figure()
        window_size = 128
        stft = scipy.signal.ShortTimeFFT(np.hamming(window_size), hop=window_size//2, fs=fs, fft_mode='centered', mfft=2048)
        mag = 20*np.log10(abs(stft.spectrogram(data.flatten())))
        freqs = np.arange(-stft.f_pts/2, stft.f_pts/2) * (fs/stft.f_pts)

        plt.pcolormesh(np.linspace(start_sample/fs*1e3, end_sample/fs*1e3, mag.shape[1]), freqs*1e-6, mag, shading='auto')
        for gscn_freq in freq_list:
            plt.axhline((gscn_freq-fc)*1e-6, linewidth=2, color='r')
        plt.ylabel('Frequency [MHz]')
        plt.xlabel('Time [msec]')
        plt.colorbar(label='Magnitude (dB)')
        plt.title('PSS no interference and highlighted GSCN frequencies')
        plt.show()
    
    # now say we want to add some interference to the data with some bandwidth like an LFM or something
    gen = WaveformGenerator(fs, end_sample/fs - start_sample/fs)
    interference_waveform = gen.make_lfm_by_startstop(-3e6, 3e6)

    # add the intereference to the data
    signal_power = np.max(abs(data))
    SINR = 3
    interferer_power = signal_power / 10 ** (SINR / 10) 
    nosisy_data = data + interference_waveform * interferer_power

    if 1:
        fig, [ax1, ax2] = plt.subplots(2,1,tight_layout=True)

        ax1.plot(np.arange(start_sample, end_sample)/fs*1e3, np.real(data),linewidth=2,color='k')
        ax1.set_title(f'Waveform no interference')
        ax2.plot(np.arange(start_sample, end_sample)/fs*1e3, np.real(nosisy_data),linewidth=2,color='k')
        ax2.set_title(f'Waveform with interference ~{SINR} dB SINR')

        for ax in [ax1, ax2]:
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Magnitude')
        plt.show()

    if 1:
        fig = plt.figure()
        window_size = 128
        stft = scipy.signal.ShortTimeFFT(np.hamming(window_size), hop=window_size//2, fs=fs, fft_mode='centered', mfft=2048)
        mag = 20*np.log10(abs(stft.spectrogram(nosisy_data.flatten())))
        freqs = np.arange(-stft.f_pts/2, stft.f_pts/2) * (fs/stft.f_pts)

        plt.pcolormesh(np.linspace(start_sample/fs*1e3, end_sample/fs*1e3, mag.shape[1]), freqs*1e-6, mag, shading='auto')
        for gscn_freq in freq_list:
            plt.axhline((gscn_freq-fc)*1e-6, linewidth=2, color='r')
        plt.ylabel('Frequency [MHz]')
        plt.xlabel('Time [msec]')
        plt.colorbar(label='Magnitude (dB)')
        plt.title('PSS w/ LFM interferer')
        plt.show()

    # set up the FFT to generate a match filter at a compatible rate
    mu = 0
    subcarrier_spacing = 15e3 * 2**mu
    nFFT = int(fs / subcarrier_spacing)

    # check each of the 3 possible PSS sequences
    for nid in [0,1,2]:
        
        # make a match filter
        match_filter = make_pss_waveform(Nid_2=nid, nFFT=nFFT)
        # correlate the filter with the data
        correlation = scipy.signal.correlate(nosisy_data, match_filter.flatten(), 'same', 'fft')

        if 1:
            # plot the correlation results
            plt.figure()
            plt.plot(np.arange(len(correlation)), abs(correlation) / np.linalg.norm(match_filter)**2)
            plt.xlabel('Lag')
            plt.ylabel('Normalized Correlation')
            plt.title(f'Correlation for PSS using Nid_2 = {nid}')
            plt.show()


if __name__ == "__main__":

    import time

    print(f"Running File: {Path(__file__).name}")

    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")