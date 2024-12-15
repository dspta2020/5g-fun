import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.linalg


def get_pss_symbols(Nid_2=0):

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

    return 1 - 2 * x[m]

def generate_resource_grid(num_slots=1, ofdm_syms_per_slot=14, subcarriers_per_block=12, num_resource_blocks=20):

    num_ofdm_symbols = num_slots * ofdm_syms_per_slot
    num_subcarriers = subcarriers_per_block * num_resource_blocks

    return np.zeros((num_ofdm_symbols, num_subcarriers))

def get_pss_inds(rg):
    
    [_, num_subcarriers] = rg.shape
    pss_len = 127

    lower_ind = num_subcarriers//2 - pss_len//2
    upper_ind = num_subcarriers//2 + pss_len//2

    # ends up being 56 to 182
    return np.arange(lower_ind-1, upper_ind)

def plot_resource_grid(rg):
    
    [num_symbols, _] = rg.shape # would probably be too busy to plot all the subcarriers
    
    plt.figure()
    plt.pcolor(abs(rg.T))
    for nth_sym in range(num_symbols):
        plt.axvline(nth_sym, linewidth=2, color='r')
    plt.xlabel('Symbol Number')
    plt.ylabel('Subcarrier Number')
    plt.title('Symbols of a single slot in subframe')
    plt.show()

def generate_waveform(rg, mu=0, nFFT=4096):

    # derive some params
    subcarrier_spacing = 15e3 * 2**mu # Hz
    sampling_rate = (subcarrier_spacing * nFFT) # samples per sec
    symbol_time = 1 / subcarrier_spacing

    # setup the ifft 
    zeros_to_pad = nFFT - rg.shape[1]
    half_zeros_to_pad = zeros_to_pad // 2
    rg_to_iFFT = np.pad(rg, pad_width=((0, 0), (half_zeros_to_pad, half_zeros_to_pad)), mode='constant', constant_values=0)

    # ifft 
    waveform = np.fft.ifft(rg_to_iFFT, axis=1).reshape(-1,1) * nFFT


    return waveform, sampling_rate, symbol_time

def plot_waveform(waveform, sampling_rate):

    # for the x axis
    time_vec = np.arange(len(waveform)) / sampling_rate

    plt.figure()
    plt.plot(time_vec*1e-6, np.real(waveform),'k-',linewidth=2,label='Real')
    plt.plot(time_vec*1e-6, np.imag(waveform),'r-',linewidth=2,label='Imag')
    plt.xlabel('Time (us)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(f'Waveform; fs = {sampling_rate}')

def main():

    # setting up some of the signals
    pss_phase_shifter = get_pss_symbols(Nid_2=0)
    rg = generate_resource_grid(num_slots=1, ofdm_syms_per_slot=14, subcarriers_per_block=12, num_resource_blocks=20) # using defaults
    pss_inds = get_pss_inds(rg=rg) # works for default grid rn

    rg[1,pss_inds] = pss_phase_shifter

    # plot the resource grid of a single slot
    # plot_resource_grid(rg=rg)

    # # numerology 
    # mu = 0
    # nFFT = 4096

    # generate the waveform 
    waveform, sampling_rate, symbol_time = generate_waveform(rg=rg, mu=0, nFFT=4096)
    # say i want to add some noise as well
    noise = (np.random.randn(waveform.shape[0],1) + 1j * np.random.randn(waveform.shape[0], 1)) / np.sqrt(2)
    waveform += noise

    # if you want to plot spectrogram of the noise injected PSS
    plot_flag = 0
    if plot_flag:

        # plotting it 
        plot_waveform(waveform=waveform, sampling_rate=sampling_rate)

        plt.figure()
        window_size = 128
        # f, t, Sxx = scipy.signal.spectrogram(x=waveform.flatten(), fs=sampling_rate, window=np.hamming(window_size), noverlap=window_size//2, nfft=2048, return_onesided=False, mode='complex')
        stft = scipy.signal.ShortTimeFFT(np.hamming(window_size), hop=window_size//2, fs=sampling_rate, fft_mode='centered', mfft=2048)
        mag = np.log10(stft.spectrogram(waveform.flatten()))
        freqs = np.arange(-stft.f_pts/2, stft.f_pts/2) * (sampling_rate/stft.f_pts)
        # time_vec = np.arange(0, stft.T, stft.delta_t)

        plt.pcolormesh(np.linspace(0, symbol_time*14, mag.shape[1]), freqs*1e-6, scipy.fft.fftshift(mag, axes=0), shading='auto')
        for nth_symbol in range(14):
            plt.axvline(symbol_time*nth_symbol, linewidth=2, color='r')

        plt.ylabel('Frequency [MHz]')
        plt.xlabel('Time [sec]')
        plt.show()

    # pull out a pure pss for the match filter
    match_filt = (waveform - noise)[int(symbol_time*1*sampling_rate):int(symbol_time*2*sampling_rate)]

    # if you want to plot the waveform next to the match filter
    if 1: # plot_flag:
        plt.figure()
        plt.plot(np.arange(len(waveform)) / sampling_rate, np.real(waveform),label='Waveform')
        plt.plot(np.arange(symbol_time*1, symbol_time*2, 1/sampling_rate),np.real(match_filt),linestyle='--',label='Match Filter')
        plt.xlabel('Time')
        plt.ylabel('Amplitude (Real)')
        plt.title('Signal and Match Filter Comparison')
        plt.show()

    # for now we'll match filter in the freq domain
    waveform_F = scipy.fft.fft(waveform, n=len(waveform), axis=0)
    match_filt_F = scipy.fft.fft(match_filt, n=len(waveform), axis=0)


    # if you want to plot the spectra of the waveform and match filter
    if 1: # plot_flag:
        nFFT = len(waveform)
        freqs = np.arange(-nFFT/2, nFFT/2) * (sampling_rate/nFFT)

        plt.figure()
        plt.plot(freqs*1e-6, 20*np.log10(abs(waveform_F)), label="Waveform")
        plt.plot(freqs*1e-6, 20*np.log10(abs(match_filt_F)), linestyle='--', label="Match Filter")
        plt.legend()
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Waveform vs Match Filter Spectra')
        plt.show()

    # do the actual match filtering
    filt_data_F = waveform_F * np.conjugate(match_filt_F)
    filt_data = scipy.fft.ifft(filt_data_F, n=len(waveform), axis=0)

    # not sure if this is the best way to normalize
    normalization_factor = scipy.linalg.norm(match_filt)**2

    # plot the match filtered data
    if 1: # plot_flag

        plt.figure() 
        plt.plot(np.arange(len(filt_data)), abs(filt_data) / normalization_factor)
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.title('Match Filtered Data')
        plt.show()
    

if __name__ == "__main__":

    from pathlib import Path

    print(f"Running File: {Path(__file__).name}")
    main()