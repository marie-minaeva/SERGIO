import os, sys
sys.path.append("/Users/littlequeen/CausalBench/SERGIO")
from run_sergio import *
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

start = time()
sim = SergioSimulator(number_genes=100, number_bins=1, number_sc=300, bio_noise_params=1, decays=0.8, sampling_state=15, noise_type='dpd')
sim.generate_data(input_file_targets = "./steady-state_input_GRN.txt",
                    input_file_regs = "./steady-state_input_MRs.txt",
                    filename="try_mth", 
                    add_noise=False)
end = time()

print(f"Finished in {end - start}")