from pathlib import Path
import numpy as np


class DataReader:

    def __init__(self, path_to_file:Path):
        '''
        Reads in the 
        '''
        self.path = path_to_file

        # read the data 
        self.filename = path_to_file.stem
    

    def parse_file(self):

        # read in the data
        infile = open(self.path)
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
    
    from pathlib import Path
    import time

    import numpy as np
    import matplotlib.pyplot as plt

    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")