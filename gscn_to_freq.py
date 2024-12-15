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


def gscn_to_freq(gscn_list):

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


def main():

    gscn_list = [4,5,6,7]
    print(gscn_to_freq(gscn_list=gscn_list))

    # # gscn_dict = generate_gscn_low()
    # # gscn_dict = generate_gscn_mid()
    # gscn_dict = generate_gscn_high()

    # x = np.array([key for key in gscn_dict.keys()])
    # y = np.array([item for item in gscn_dict.values()])

    # xhat = np.linspace(22256,26639,500)
    # yhat = gscn_to_freq(xhat)

    # plt.plot(xhat, yhat, 'o-')
    # plt.plot(x, y, '-')
    # # plt.xlim([7498,7498+20])
    # # plt.ylim([0, 4e6])
    # # plt.savefig('temp.png')
    # plt.grid()
    # plt.show()




if __name__ == "__main__":

    from pathlib import Path

    print(f"Running File: {Path(__file__).name}")
    main()
    print("Script finished.")