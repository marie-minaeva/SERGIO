import os, sys
sys.path.append("/Users/littlequeen/CausalBench/SERGIO")
from run_sergio import *
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from tqdm import tqdm

start = time()
for prob in tqdm(np.linspace(0.0, 6.0, 13)):
    sim = SergioSimulator(number_genes=1200, number_bins=1, number_sc=300, bio_noise_params=1, decays=0.8, sampling_state=15, noise_type='dpd')
    sim.generate_data(input_file_targets = "SERGIO/Interaction_cID_6.txt",
                        input_file_regs = "SERGIO/Regs_cID_6.txt",
                        filename="cID_6_outlier_mean_" + str(np.round(prob, 2)), 
                        add_noise=True,
                    noise_params={"outlier_prob": 0.01,
                                  "outlier_mean": prob,
                                  "outlier_scale": 1.0,
                                  "lib_mean": 4.5,
                                  "lib_scale": 0.7,
                                  "shape": 8,
                                  "percentile": 45.0})
end = time()

print(f"Finished in {end - start}")