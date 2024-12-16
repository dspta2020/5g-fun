import numpy as np 
import scipy
import matplotlib.pyplot as plt
import scipy.constants


from ssb_testing import get_pss_symbols, make_pss_waveform, make_resource_grid, get_pss_inds, rg_to_waveform
from freq_to_gscn import freq_to_gscn
from gscn_to_freq import gscn_to_freq

def plot_spectrogram(waveform, fs, symbol_time, window_size):

    fig = plt.figure()
    window_size = 128
    stft = scipy.signal.ShortTimeFFT(np.hamming(window_size), hop=window_size//2, fs=fs, fft_mode='centered', mfft=2048)
    mag = np.log10(stft.spectrogram(waveform.flatten()))
    freqs = np.arange(-stft.f_pts/2, stft.f_pts/2) * (fs/stft.f_pts)

    plt.pcolormesh(np.linspace(0, symbol_time*14, mag.shape[1]), freqs*1e-6, scipy.fft.fftshift(mag, axes=0), shading='auto')
    for nth_symbol in range(14):
        plt.axvline(symbol_time*nth_symbol, linewidth=2, color='r')

    plt.ylabel('Frequency [MHz]')
    plt.xlabel('Time [sec]')

    return fig

def main():

    # setting up some of the signals
    pss_phase_shifter = get_pss_symbols(Nid_2=0)
    rg = make_resource_grid(num_slots=1, ofdm_syms_per_slot=14, subcarriers_per_block=12, num_resource_blocks=20) # using defaults
    pss_inds = get_pss_inds(rg=rg) # works for default grid rn

    # add in the pss
    rg[1,pss_inds] = pss_phase_shifter

    # generate the waveform 
    waveform, fs_original, symbol_time = rg_to_waveform(rg=rg, mu=0, nFFT=4096)

    # say i want to add some noise as well
    noise = (np.random.randn(waveform.shape[0],1) + 1j * np.random.randn(waveform.shape[0], 1)) / np.sqrt(2)
    waveform += noise

    # just want to resample to close to 100e6
    fs_desired = 102e6
    num_samples = len(waveform)
    num_samples_desired = (len(waveform) / fs_original) * fs_desired

    # going to sinc interpolate in the frequency domain
    zeros_to_pad = num_samples_desired - num_samples 
    half_zeros_to_pad = int(zeros_to_pad // 2)

    # zero-pad the spectrum and then ifft and calculate the approximate true fs
    waveform_F = scipy.fft.fft(waveform, n=len(waveform), axis=0)
    waveform_F_padded = np.pad(waveform_F, pad_width=((half_zeros_to_pad, half_zeros_to_pad),(0,0)), mode='constant', constant_values=0)
    waveform_resampled = scipy.fft.ifft(waveform_F_padded, n=len(waveform_F_padded), axis=0)
    fs_new = len(waveform_resampled) / (len(waveform) / fs_original)

    # make up a center freq around some nr bands
    fc = 770e6
    bw = fs_new // 2
    upper_limit = fc + bw
    lower_limit = fc - bw

    # the PSS is transmitted on these GSCN frequencies so search the frequency space for candidate channels
    freq_search_space = np.arange(lower_limit, upper_limit, 0.1e6)
    gscn_list = np.round(freq_to_gscn(freq_list=freq_search_space))
    freq_list = np.unique(gscn_to_freq(gscn_list=gscn_list))
    
    # now we calculate the offset from DC 
    gscn_offsets = freq_list - fc

    # lets choose a random gscn frequency to set the PSS
    target_offset = gscn_offsets[np.random.randint(0,len(gscn_offsets) + 1)] # +1 bc randint int is [0,high)

    # make an LO to that offset and mix the data
    Lo = np.exp(1j * 2 * scipy.constants.pi * target_offset * np.arange(waveform_resampled.shape[0]) / fs_new)
    waveform_resampled_mixed = waveform_resampled * Lo.reshape(-1,1)

    # this is a good spot for plotting if you want right here
    if 0:
        plot_spectrogram(waveform=waveform_resampled_mixed, fs=fs_new, symbol_time=symbol_time, window_size=256)
        plt.title(f'PSS; fc = {fc*1e-6} MHz; fs = {fs_new*1e-6} MHz; Offset {target_offset*1e-6} MHz')
        plt.axhline(target_offset*1e-6, linewidth=2, color='r')
        plt.show()

    # now ill make a match filter at the data rate
    mu = 0
    subcarrier_spacing = 15e3 * 2**mu
    nFFT = int(fs_new / subcarrier_spacing)
    match_filter = make_pss_waveform(Nid_2=0, nFFT=nFFT)

    # how many offsets do we need to check based on the GSCNs available in our band
    num_gscn_channels = len(freq_list)
    
    # so now we have our match filter bank and one of them should pop for the PSS we made and shifted
    Lo_base = np.exp(1j * 2 * scipy.constants.pi * (gscn_offsets).reshape(-1,1).T* np.arange(match_filter.shape[0]).reshape(-1,1) / fs_new)
    filter_bank = np.tile(match_filter, (1, num_gscn_channels))
    filter_bank = Lo_base * filter_bank

    # now we need to do the match filter process
    waveform_F = scipy.fft.fft(waveform_resampled_mixed, n=len(waveform_resampled_mixed), axis=0)
    filter_bank_F = scipy.fft.fft(filter_bank, n=len(waveform_resampled_mixed), axis=0)
    waveform_array_F = np.tile(waveform_F, (1, filter_bank.shape[1]))

    # so here is where we do the actual processing
    filtered_data_F = waveform_array_F * np.conjugate(filter_bank_F)

    # ifft
    filtered_data = scipy.fft.ifft(filtered_data_F, n=len(filtered_data_F), axis=0)

    plt.plot(abs(filtered_data))
    plt.show()


# gonna add a second main for messing around
def main2():
    '''
    This main looks at resampling the simulated data and doing a match filter process
    with and without resampling the match filter.
    '''

    # setting up some of the signals
    pss_phase_shifter = get_pss_symbols(Nid_2=0)
    rg = make_resource_grid(num_slots=1, ofdm_syms_per_slot=14, subcarriers_per_block=12, num_resource_blocks=20) # using defaults
    pss_inds = get_pss_inds(rg=rg) # works for default grid rn

    # add in the pss
    rg[1,pss_inds] = pss_phase_shifter

    # generate the waveform 
    waveform, fs_original, symbol_time = rg_to_waveform(rg=rg, mu=0, nFFT=4096)

    # say i want to add some noise as well
    noise = (np.random.randn(waveform.shape[0],1) + 1j * np.random.randn(waveform.shape[0], 1)) / np.sqrt(2)
    waveform += noise

    # just want to resample to close to 100e6
    fs_desired = 102e6
    num_samples = len(waveform)
    num_samples_desired = (len(waveform) / fs_original) * fs_desired

    # going to sinc interpolate in the frequency domain
    zeros_to_pad = num_samples_desired - num_samples 
    half_zeros_to_pad = int(zeros_to_pad // 2)

    # zero-pad the spectrum and then ifft and calculate the approximate true fs
    waveform_F = scipy.fft.fft(waveform, n=len(waveform), axis=0)
    waveform_F_padded = np.pad(waveform_F, pad_width=((half_zeros_to_pad, half_zeros_to_pad),(0,0)), mode='constant', constant_values=0)
    waveform_resampled = scipy.fft.ifft(waveform_F_padded, n=len(waveform_F_padded), axis=0)
    fs_new = len(waveform_resampled) / (len(waveform) / fs_original)

    # make up a center freq around some nr bands
    fc = 770e6
    bw = fs_new // 2

    fig, [ax1, ax2] = plt.subplots(2,1,tight_layout=True)

    # okay so we have this resampled PSS waveform here...
    # now say I wanted to do the match filtering process...
    match_filter = make_pss_waveform(Nid_2=0, nFFT=4096)
    mu = 0
    subcarrier_spacing = 15e3 * 2**mu
    fs_filter = subcarrier_spacing * 4096

    # another option is to generate the match filter at the target rate
    nFFT = int(fs_desired / subcarrier_spacing)
    match_filter = make_pss_waveform(Nid_2=0, nFFT=nFFT)
    
    ### i dont think this will work without having the fs corrected 
    waveform_F = scipy.fft.fft(waveform_resampled, n=len(waveform_resampled), axis=0)
    match_filter_F = scipy.fft.fft(match_filter, n=len(waveform_resampled), axis=0)

    # filter
    filtered_data_F = waveform_F * np.conjugate(match_filter_F)

    # ifft 
    filtered_data = scipy.fft.ifft(filtered_data_F, n=len(waveform_resampled), axis=0)

    # plot with fs mismatch in top subplot
    ax1.plot(np.arange(len(filtered_data)), abs(filtered_data), color='k')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Resampled Data; Not Resampled Match Filter')

    ### repeat this proces with the match filter having been resampled
    fs_desired = 102e6
    num_samples = len(match_filter)
    num_samples_desired = (len(match_filter) / fs_filter) * fs_desired

    # going to sinc interpolate in the frequency domain
    zeros_to_pad = num_samples_desired - num_samples 
    half_zeros_to_pad = int(zeros_to_pad // 2)

    # zero-pad the spectrum and then ifft and calculate the approximate true fs
    match_filter_F = scipy.fft.fft(match_filter, n=len(match_filter), axis=0)
    match_filter_F_padded = np.pad(match_filter_F, pad_width=((half_zeros_to_pad, half_zeros_to_pad),(0,0)), mode='constant', constant_values=0)
    match_filter_resampled = scipy.fft.ifft(match_filter_F_padded, n=len(match_filter_F_padded), axis=0)
    fs_new = len(match_filter_resampled) / (len(match_filter) / fs_filter)

    waveform_F = scipy.fft.fft(waveform_resampled, n=len(waveform_resampled), axis=0)
    match_filter_F = scipy.fft.fft(match_filter_resampled, n=len(waveform_resampled), axis=0)

    # filter
    filtered_data_F = waveform_F * np.conjugate(match_filter_F)

    # ifft 
    filtered_data = scipy.fft.ifft(filtered_data_F, n=len(waveform_resampled), axis=0)

    # plot with fs mismatch in top subplot
    ax2.plot(np.arange(len(filtered_data)), abs(filtered_data), color='r')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Resampled Data; Resampled Match Filter')

    plt.show()


if __name__ == "__main__":

    from pathlib import Path

    print(f"Running File: {Path(__file__).name}")
    main()
    
    # not going to run for now .. just looks at a resampled match filter process
    # main2()