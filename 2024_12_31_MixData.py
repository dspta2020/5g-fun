from pathlib import Path
import time

import numpy as np 
import matplotlib.pyplot as plt
import scipy
import scipy.fft
import scipy.signal 

from DataReader import DataReader
from FreqGscnMapper import FreqGscnMapper
from WaveformGenerator import WaveformGenerator

def main():

    path_to_data = Path('./data/954_7680KSPS_srsRAN_Project_gnb_short.txt')
    reader = DataReader(path_to_file=path_to_data)

    fs, fc, waveform = reader.parse_file()

    # its kinda of an alot of data so just going to truncate it for now
    start_sample = 0.1 * fs
    end_sample = len(waveform)
    dur = (end_sample / fs - start_sample / fs)
    num_samples = end_sample - start_sample
    waveform = waveform[int(start_sample):int(end_sample)]


    # maybe want to increase the bandwidth though
    # i think by 8x is reasonable... its a common rate 
    # this'll give more room to do things
    fs_new = fs * 8
    target_num_samples = dur * fs_new

    # sinc interpolation in the freq domain is easiest right now 
    num_zeros_to_pad = target_num_samples - num_samples
    half_zeros_to_pad = int(num_zeros_to_pad // 2)
    waveform_F = scipy.fft.fftshift(scipy.fft.fft(waveform, len(waveform), 0), 0)
    waveform_F = np.pad(waveform_F, pad_width=((half_zeros_to_pad,half_zeros_to_pad)), mode="constant", constant_values=0)

    # get the interpolated waveform back
    waveform = scipy.fft.ifft(scipy.fft.ifftshift(waveform_F, 0), len(waveform_F), 0)

    # find the bandwidth and the upper and lower edges
    bw = fs_new / 2
    upper_edge = fc + bw
    lower_edge = fc - bw

    # now find the GSCN freqs 
    mapper = FreqGscnMapper()
    search_space = np.arange(lower_edge, upper_edge, 0.01e6)
    gscn_candidates = np.unique(np.round(mapper.freq_to_gscn(search_space)))
    freq_candidates = mapper.gscn_to_freq(gscn_candidates)

    # now we wanna pick a new gscn freq to mix to
    mix_ind = 120
    mixing_freq = (freq_candidates[mix_ind] - fc)
    Lo = np.exp(1j * 2 * np.pi * mixing_freq * np.arange(len(waveform)) / fs_new)
    waveform = waveform * Lo

    #############################################################################################################################
    # from here we'll have the mixed waveform to be at the freq we want 
    # then we can a also have some band limited noise and other waveforms in there
    

    fig = plt.figure()
    window_size = 4096
    stft = scipy.signal.ShortTimeFFT(np.hamming(window_size), hop=window_size//2, fs=fs_new, fft_mode='centered', mfft=8192)
    mag = 20*np.log10(abs(stft.spectrogram(waveform.flatten())))
    freqs = np.arange(-stft.f_pts/2, stft.f_pts/2) * (fs_new/stft.f_pts)

    plt.pcolormesh(np.linspace(0, len(waveform)/fs_new*1e3, mag.shape[1]), freqs*1e-6, mag, shading='auto')
    # for gscn_freq in freq_candidates:
    #     plt.axhline((gscn_freq - fc)*1e-6, color='r', linewidth=2)
    plt.ylabel('Frequency [MHz]')
    plt.xlabel('Time [msec]')
    plt.colorbar(label='Magnitude (dB)')
    plt.show()


if __name__ == "__main__":

    print(f"Running File: {Path(__file__).name}")

    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")