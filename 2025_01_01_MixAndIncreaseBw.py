from pathlib import Path
import time
import struct

import numpy as np 
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft, ifftshift
from scipy.signal import ShortTimeFFT

from classes.FileHandler import FileHandler
from classes.FreqGscnMapper import FreqGscnMapper


def write_data_to_bin(outfile_path, data):
    '''
    Becuase im getting lazy just gonna make a static function here
    '''
    outfile = open(outfile_path, 'wb')
    data = data.flatten()

    for val in data:
        outfile.write(struct.pack('f', val))

    outfile.close()


def main():

    # read in the data with the data reader
    path_to_data = Path('./data/954_7680KSPS_srsRAN_Project_gnb_short.txt')
    reader = FileHandler(path_to_file=path_to_data)

    # get the data
    fs, fc, waveform = reader.parse_file()
    num_samples = len(waveform)

    # derive the new rate
    fs = fs * 4
    num_samples = num_samples * 4

    # sinc interpolate so setup the zero padding
    num_zeros_to_pad = num_samples - len(waveform)
    half_zeros_to_pad = int(num_zeros_to_pad // 2)

    # fft the data and pad
    waveform_F = fftshift(fft(waveform, len(waveform), 0), 0)
    waveform_F = np.pad(waveform_F, pad_width=((half_zeros_to_pad,half_zeros_to_pad)), mode="constant", constant_values=0)

    # get the interpolated waveform back
    waveform = ifft(ifftshift(waveform_F, 0), len(waveform_F), 0).reshape(-1,1)

    # i think it makes sense to shift it to an actual GSCN ind
    mapper = FreqGscnMapper()

    # get the spectrum
    bw = fs / 2
    upper_edge = fc + bw 
    lower_edge = fc - bw

    # define a frequency search space and identify candidate frequencies to shift to
    search_space = np.arange(lower_edge, upper_edge, 0.1e6)
    gscn_candidates = np.unique(np.round(mapper.freq_to_gscn(search_space)))
    freq_candidates = mapper.gscn_to_freq(gscn_candidates)
    
    # convert the frequencies to units of offset from DC
    candidates_offsets = freq_candidates - fc
    candidate_inds = np.arange(len(candidates_offsets))

    # dont want to shift too close to the edge of the band but outside of the current bandwidth
    valid_shift_inds = np.logical_and(candidates_offsets > 10e6, candidates_offsets < 15e6)
    shift_freq = np.random.choice(candidate_inds[valid_shift_inds])

    # make an local oscillator shift the waveform
    Lo = np.exp(1j * 2 * np.pi * candidates_offsets[shift_freq] * np.arange(len(waveform)) / fs).reshape(-1,1)

    # mix 
    waveform = waveform * Lo
    
    # convert to interleved iq
    approx_size = (len(waveform) * 2 * 4) / 1e6 # in MB
    waveform_interleved = np.zeros((len(waveform)*2,1))
    waveform_interleved[::2] = np.real(waveform)
    waveform_interleved[1::2] = np.imag(waveform)

    # write to a bin file in 4 byte floats 
    outfile_name = f'fs{fs*1e-6:0.2f}_fc{(fc)*1e-6:0.2f}_shift{candidates_offsets[shift_freq]*1e-6:0.2f}_data.bin'
    outfile_path = Path(f'{__file__}').parent / 'data' / outfile_name

    # okay now we want to save off the waveform
    write_data_to_bin(outfile_path, waveform_interleved)

    # # plot
    # plt.figure()
    
    # # setup and call STFT object 
    # window_size = 4096
    # stft = ShortTimeFFT(np.hamming(window_size), hop=window_size//2, fs=fs, fft_mode='centered', mfft=8192)
    
    # # return the spectrogram 
    # mag = 20*np.log10(abs(stft.spectrogram(waveform.flatten())))
    # freqs = np.arange(-stft.f_pts/2, stft.f_pts/2) * (fs/stft.f_pts)
    # time_vec = np.linspace(0, len(waveform)/fs*1e3, mag.shape[1])

    # # plot
    # plt.pcolormesh(time_vec, freqs*1e-6, mag, shading='auto')
    # plt.ylabel('Frequency [MHz]')
    # plt.xlabel('Time [msec]')
    # plt.colorbar(label='Magnitude (dB)')
    # plt.show()



if __name__ == "__main__":

    print(f"Running File: {Path(__file__).name}")

    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")