from pathlib import Path

import numpy as np 
import scipy.signal


class WaveformGenerator:

    def __init__(self, fs, dur):
        
        self.fs = fs 
        self.dur = dur 
        self.num_samples = self.dur * self.fs

    def make_lfm_by_startstop(self, f0, f1):

        bw = f1 - f0
        slope = bw / self.dur
        t = np.arange(self.num_samples) / self.fs

        return np.exp(1j * 2*np.pi * (1/2*slope*t**2 + f0*t))
    
    def make_lfm_by_slope(self, f0, slope):

        t = np.arange(self.num_samples) / self.fs
        
        return np.exp(1j * 2*np.pi * (1/2*slope*t**2 + f0*t))



def main():
    pass

if __name__ == "__main__":

    import time

    print(f"Running File: {Path(__file__).name}")

    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")