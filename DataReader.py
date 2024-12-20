from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

class DataReader:

    def __init__(self, path_to_file:Path):
        '''
        Reads in the 
        '''
        # read the data 
        self.filename = path_to_file.stem
        self.fs, self.fc, self.samples = self.parse_file(path_to_file)

        # derived perameters
        self.num_samples = len(self.samples)
        self.time_vec = np.arange(self.num_samples) / self.fs
        self.dur = self.time_vec[-1]
    

    def parse_file(self, path:Path):

        # read in the data
        infile = open(path)
        lines = infile.readlines()
        infile.close()

        fs = int(lines[0].split(',')[1].split('.')[0])
        fc = int(lines[1].split(',')[1].split('.')[0])

        samples = []
        for line_ind in range(2, len(lines)):
            
            sample = float(lines[line_ind].split(',')[1]) + 1j * float(lines[line_ind].split(',')[2])
            samples.append(sample)

        return fs, fc, np.array(samples)

def main():

    path_to_data = Path(__file__).parent / '954_7680KSPS_srsRAN_Project_gnb_short.txt'

    reader = DataReader(path_to_data)
    data = reader.samples
    fs = reader.fs 
    fc = reader.fc
    time_vec = reader.time_vec

    # plt.figure()
    # plt.plot(time_vec, abs(data))
    # plt.show()


if __name__ == "__main__":

    import time

    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")