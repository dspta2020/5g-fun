from pathlib import Path
import time

import numpy as np 
from scipy.fft import fft, fftshift, ifft, ifftshift

import numpy as np
import matplotlib.pyplot as plt


class FreqGscnMapper:

    def __init__(self):
        pass

    def _generate_gscn_low(self):

        # store results in a dict[index] = freq
        gscn_dict = {}
        for n in np.arange(1,2500):
            for m in [1,3,5]:

                # calculate the index and freq
                ind = 3*n + (m-3)/2
                freq = n*1200e3 + m*50e3

                # add dict entry 
                gscn_dict[ind] = freq
        
        return gscn_dict

    def _generate_gscn_mid(self):
        
        # store results in a dict[index] = freq
        gscn_dict = {}
        for n in np.arange(0, 14757):

            # calculate the index and freq
            ind = 7499 + n
            freq = 3000e6 + n*1.44e6

            # add dict entry 
            gscn_dict[ind] = freq

        return gscn_dict

    def _generate_gscn_high(self):
        
        # store results in a dict[index] = freq
        gscn_dict = {}
        for n in np.arange(0, 4383):

            # calculate the index and freq
            ind = 22256 + n
            freq = 24250.08e6 + n*17.28e6

            # add dict entry 
            gscn_dict[ind] = freq

        return gscn_dict
    
    def freq_to_gscn(self, freq_list):

        # list to store the results
        results_list = []
        for nth_freq in freq_list:
            
            # determine what range the gscn is in 
            # here is range mid
            if nth_freq >= 3000e6 and nth_freq <= 24250e6:
                # estimate N
                n_estimate = (nth_freq - 3000e6) / 1.44e6

                # solve for freq
                results_list.append(n_estimate + 7499)
            
            # here is range high
            elif nth_freq >= 24250e6:
                # estimate N
                n_estimate = (nth_freq - 24250.08e6) / 17.28e6

                # solve for freq 
                results_list.append(n_estimate + 22256)
            
            # here is range low
            else:   
                # shift to the origin
                offset_freq = nth_freq - 1.25e6

                # calculate the approximate period
                nth_period = offset_freq // 1.2e6

                # wrap around the interval [0,1.2e6)
                relative_phase = offset_freq % 1.2e6

                # make two peice-wise linear function 
                steep_slope = 1e-5
                shallow_slope = 1e-6

                # transition between the two peices is at a relative phase of 0.2e6
                if relative_phase <= 0.2e6:
                    results_list.append(relative_phase * steep_slope + nth_period * 3 + 2)
                else:
                    results_list.append((relative_phase - 0.2e6) * shallow_slope + nth_period * 3 + 4)

        return np.array(results_list)
    
    def gscn_to_freq(self, gscn_list):

        # list to store the results
        results_list = []
        for nth_ind in gscn_list:
            
            # determine what range the gscn is in 
            # here is range mid
            if nth_ind >= 7499 and nth_ind <= 22255:
                # estimate N
                n_estimate = nth_ind - 7499

                # solve for freq
                results_list.append(3000e6 + n_estimate * 1.44e6)
            
            # here is range high
            elif nth_ind >= 22256:
                # estimate N
                n_estimate = nth_ind - 22256

                # solve for freq 
                results_list.append(24250.08e6 + n_estimate * 17.28e6)
            
            # here is range low
            else:   
                # shift to the origin
                offset_ind = nth_ind - 2

                # calculate the approximate period
                nth_period = offset_ind // 3

                # wrap around the interval [0,3)
                relative_phase = offset_ind % 3

                # make two peice-wise linear function 
                shallow_slope = 0.1e6
                steep_slope = 1e6

                # transition between the two peices is at a relative phase of 2
                if relative_phase <= 2:
                    results_list.append(relative_phase * shallow_slope + nth_period * 1.2e6 + 1.25e6)
                else:
                    results_list.append((relative_phase - 2) * steep_slope + nth_period * 1.2e6 + 1.45e6)

        return np.array(results_list)
    
def ReadData(path_to_data):

    # read in the data
    infile = open(path_to_data)
    lines = infile.readlines()
    infile.close()

    fs = int(lines[0].split(',')[1].split('.')[0])
    fc = int(lines[1].split(',')[1].split('.')[0])

    samples = []
    for line_ind in range(2, len(lines)):
        
        sample = float(lines[line_ind].split(',')[1]) + 1j * float(lines[line_ind].split(',')[2])
        samples.append(sample)

    return fs, fc, np.array(samples)

def GetPssSymbols(Nid_2=0):

    if Nid_2 % 1 != 0:
        Nid_2 = round(Nid_2)
        print(f"Decimal Nid_2 not allowed. Rounding to Nid_2 = {round(Nid_2)}.\n")

    # make sure Nid_2 is on the correct interval
    Nid_2 = Nid_2 % 3

    x = ["0","1","1","0","1","1","1"] + ["0"]*120
    x = np.array(x,dtype=int)
    for ind in range(7,127):
        x[ind] = (x[ind+4-7] + x[ind-7]) % 2
    
    m = (np.arange(127) + 43 * Nid_2) % 127
    d_pss = 1 - 2 * x[m]
    
    return d_pss

def MakePssWaveform(Nid_2=0, nFFT=4096):

    # make the symbols for the PSS to generate
    symbols = np.zeros((240,1),dtype=complex)
    symbols[56:183] = GetPssSymbols(Nid_2=Nid_2).reshape(-1,1)

    # setup the ifft
    zeros_to_pad = nFFT - len(symbols)
    half_zeros_to_pad = zeros_to_pad // 2
    symbols_padded = np.pad(symbols, pad_width=((int(half_zeros_to_pad), int(half_zeros_to_pad)),(0,0)), mode='constant', constant_values=0)

    # ifft 
    pss_waveform = ifft(ifftshift(symbols_padded, axes=0), n=len(symbols_padded), axis = 0) * nFFT

    return pss_waveform.reshape(-1,1)

def MakeLowPassFilter(fs, fc, N):

    # Calculate normalized cutoff frequency
    fc_normalized = fc / fs

    # Initialize filter coefficients array
    h = np.zeros((int(N), 1))

    # Calculate mid-point of the filter
    mid = N // 2

    # Calculate the filter coefficients
    for n in np.arange(N):
        if n != mid:
            h[int(n)] = np.sin(2 * np.pi * fc_normalized * (n - mid)) / (np.pi * (n - mid))
        else:
            h[int(n)] = 2 * fc_normalized

    # Normalize the filter coefficients to ensure the gain is one
    h = h / np.sum(h)
    
    return h

def main():
    
    # get the path to the data
    path_to_data = Path('data\954_7680KSPS_srsRAN_Project_gnb_short.txt')
    fs, fc, waveform = ReadData(path_to_data=path_to_data)

    # get the band edges
    upper_edge = fc + (fs/2)
    lower_edge = fc - (fs/2)
    search_space = np.arange(lower_edge, upper_edge, 0.1e6)

    # get potential gscn inds and freqs where the PSS/SSB might be transmitted
    mapper = FreqGscnMapper()
    gscn_candidates = np.unique(np.round(mapper.freq_to_gscn(search_space)))
    freq_candidates = mapper.gscn_to_freq(gscn_candidates) # i think in this case is band 3
    offset_candidates = freq_candidates - fc

    # for now assume the subcarrier spacing is 15 kHz 
    subcarrier_spacing = 15e3
    nIFFT = fs / subcarrier_spacing

    # make a PSS match filter
    Nid_2 = 1
    match_filter = MakePssWaveform(Nid_2, nIFFT) * nIFFT # this will be at baseband

    # setup the FFT of the waveform
    # (freq_candidates - fc) / 4e3 # i think should be able to get away with 4 kHz spacing as minimum FFT size
    # (127 * 15e3) / desired_bin_spacing # also out of curiosity how many bins is the match filter with this bin spacing
    
    desired_bin_spacing = 4e3  # can be optimized later
    nFFT = fs / desired_bin_spacing
    shifts = offset_candidates / desired_bin_spacing
    shifts = ((shifts + nFFT)).astype(int).reshape(-1,1) # these are the freq bin inds we need to grab

    # make the circshift matrix 
    circshift_mat = np.tile(np.arange(nFFT).reshape(-1,1), (1, len(offset_candidates))) # make a matrix for each shift with the current bin inds
    circshift_mat = ((shifts.T + circshift_mat) % nFFT).astype(int) # rotate all the ind positions, then wrap back around the FFT index interval

    # need to double check the circle shifts in python.. i know in matlab it needs a -1 then +1
    freqs = np.arange(-nFFT/2,nFFT/2) * (fs/nFFT)
    # freqs[circshift_mat[int(nFFT // 2)]]
    # offset_candidates

    # now need to set up the loop to grab the data and FFT it
    percent_overlap = 0 # incase the PSS splits the window
    # just gonna set to nFFT for now but basically right now th nFFT is acting as the max window size 
    # but nFFT can be increased.
    window_len = nFFT 
    hop_size = np.floor(window_len * (1 - percent_overlap))
    num_windows = np.ceil((len(waveform) - window_len) / hop_size)

    # also going to make a low pass filter since we are doing the filtering at baseband
    # dont think its necessary since the SNR is so good but it will get rid of some energy
    # in outside our band 
    filter_bw = ((127 * 15e3) / 2) * 1.1
    filter_coeffs = MakeLowPassFilter(fs, filter_bw, nFFT)

    # make the different windows we need
    match_filter_F = fftshift(fft(match_filter, int(nFFT), 0), 0)
    filter_F = fftshift(fft(filter_coeffs, int(nFFT), 0), 0)
    win = np.hamming(nFFT)

    for nth_window in range(int(num_windows)):

        # i think this should be fine for grabbing the right inds
        start_ind = int(nth_window * hop_size)
        end_ind = int(start_ind + window_len)

        # grab the time data
        input_data = waveform[start_ind:end_ind] * win
        # FFT the data
        input_data_F = fftshift(fft(input_data, int(nFFT), 0), 0)
        # rotate the frequencies with the circshift matrix
        shifted_data_F = input_data_F[circshift_mat]

        # can add a filtering step later
        shifted_data_F = shifted_data_F * filter_F

        plt.figure(2)
        plt.plot(20*np.log10(abs(shifted_data_F)))

        # also normalize the 
        input_data_energy = sum(abs(shifted_data_F)**2)
        match_filter_energy = sum(abs(match_filter)**2)

        # do the correlation 
        correlated_data_F = shifted_data_F * np.conj(match_filter_F)

        # ifft the filtered data
        correlated_data = ifft(ifftshift(correlated_data_F, 0), int(nFFT), 0) / np.sqrt(input_data_energy * match_filter_energy).reshape(-1,1).T

        plt.figure(1)
        plt.plot(abs(correlated_data))
        plt.show()



if __name__ == "__main__":

    print(f"Running File: {Path(__file__).name}")

    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")