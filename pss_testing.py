import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.fft
import scipy.signal


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

def main():

    pss_phase_shifter = get_pss_symbols(Nid_2=0)
    rg = generate_resource_grid(num_slots=1, ofdm_syms_per_slot=14, subcarriers_per_block=12, num_resource_blocks=20) # using defaults
    pss_inds = get_pss_inds(rg=rg) # works for default grid rn

    rg[1,pss_inds] = pss_phase_shifter

    # plot the resource grid of a single slot
    # plot_resource_grid(rg=rg)

    # numerology 
    mu = 0
    nFFT = 4096

    subcarrier_spacing = 15e3 * 2**mu # Hz
    sampling_rate = (subcarrier_spacing * nFFT)
    sample_spacing = 1 / sampling_rate
    symbol_time = 1 / subcarrier_spacing
    
    zeros_to_pad = nFFT - rg.shape[1]
    half_zeros_to_pad = zeros_to_pad // 2

    rg_to_iFFT = np.pad(rg, pad_width=((0, 0), (half_zeros_to_pad, half_zeros_to_pad)), mode='constant', constant_values=0)
    
    # should now be zeros the length of nFFT
    # plot_resource_grid(rg=rg_to_iFFT)

    # ifft the rg
    waveform = np.fft.ifft(rg_to_iFFT.T, axis=0).reshape(-1,1,order="F") * nFFT
    time_vec = np.arange(len(waveform)) / sampling_rate


    # say i want to add some noise as well
    noise = (np.random.randn(waveform.shape[0],1) + 1j * np.random.randn(waveform.shape[0], 1)) / np.sqrt(2)
    waveform += noise

    plt.figure()
    plt.plot(time_vec*1e-6, np.real(waveform),'k-',linewidth=2,label='Real')
    plt.plot(time_vec*1e-6, np.imag(waveform),'r-',linewidth=2,label='Imag')
    plt.xlabel('Time (us)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(f'PSS where; nFFT = {nFFT}; mu = {mu}')

    plt.figure()
    window_size = 128
    # f, t, Sxx = scipy.signal.spectrogram(x=waveform.flatten(), fs=sampling_rate, window=np.hamming(window_size), noverlap=window_size//2, nfft=2048, return_onesided=False, mode='complex')
    stft = scipy.signal.ShortTimeFFT(np.hamming(window_size), hop=window_size//2, fs=sampling_rate, fft_mode='centered', mfft=2048)
    mag = np.log10(stft.spectrogram(waveform.flatten()))
    freqs = np.arange(-stft.f_pts/2, stft.f_pts/2) * (sampling_rate/stft.f_pts)
    # time_vec = np.arange(0, stft.T, stft.delta_t)

    plt.pcolormesh(np.linspace(0, symbol_time*14, mag.shape[1]), freqs*1e-6, scipy.fft.fftshift(mag, axes=0), shading='auto')
    plt.ylabel('Frequency [MHz]')
    plt.xlabel('Time [sec]')
    plt.show()


if __name__ == "__main__":

    from pathlib import Path

    print(f"Running File: {Path(__file__).name}")
    main()