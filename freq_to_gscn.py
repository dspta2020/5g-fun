import numpy as np
import matplotlib.pyplot as plt


def generate_gscn_low():

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


def generate_gscn_mid():
    
    # store results in a dict[index] = freq
    gscn_dict = {}
    for n in np.arange(0, 14757):

        # calculate the index and freq
        ind = 7499 + n
        freq = 3000e6 + n*1.44e6

        # add dict entry 
        gscn_dict[ind] = freq

    return gscn_dict


def generate_gscn_high():
    
    # store results in a dict[index] = freq
    gscn_dict = {}
    for n in np.arange(0, 4383):

        # calculate the index and freq
        ind = 22256 + n
        freq = 24250.08e6 + n*17.28e6

        # add dict entry 
        gscn_dict[ind] = freq

    return gscn_dict


def freq_to_gscn(freq_list):

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


def main():

    # gscn_dict = generate_gscn_low()
    # gscn_dict = generate_gscn_mid()
    gscn_dict = generate_gscn_high()

    y = np.array([key for key in gscn_dict.keys()])
    x = np.array([item for item in gscn_dict.values()])

    xhat = np.linspace(np.min(x),np.max(x),int(100))
    yhat = freq_to_gscn(xhat)

    plt.plot(xhat, yhat, 'o-')
    plt.plot(x, y, 'o-')
    # plt.xlim([0,4e6])
    # plt.ylim([0, 10])
    plt.grid()
    plt.show()

    # test_freqs = [2.68e6]
    # print(freq_to_gscn(test_freqs))

if __name__ == "__main__":

    from pathlib import Path

    print(f"Running File: {Path(__file__).name}")
    main()
    print("Script finished.")