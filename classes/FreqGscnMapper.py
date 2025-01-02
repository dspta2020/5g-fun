import numpy as np

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
    

if __name__ == '__main__':

    import numpy as np 

    mapper = FreqGscnMapper()

    test_gscn = mapper.freq_to_gscn([645.77e6])
    test_freq = mapper.gscn_to_freq([1614]) * 1e-6

    print(test_freq, test_gscn)