import numpy as np
import os, sys
import copy
from threading import Thread
import uuid
from SERGIO.sergio import *


class SergioSimulator(sergio):
        def __init__(self, number_genes, number_bins, number_sc, bio_noise_params, noise_type, decays, \
                      sampling_state, dynamics=False, dt=0.01, bifurcation_matrix=None, bio_noise_params_splice=None, noise_type_splice=None, \
                        splice_ratio=4, dt_splice=0.01, exclude_inters=np.array([]).astype(np.float32), var_names=None):
            super().__init__(number_genes=number_genes, number_bins=number_bins, number_sc=number_sc, noise_params=bio_noise_params, noise_type=noise_type, \
                             decays=decays, sampling_state=sampling_state, dynamics=dynamics, dt=dt, bifurcation_matrix=bifurcation_matrix, \
                                noise_params_splice=bio_noise_params_splice, noise_type_splice=noise_type_splice, splice_ratio=splice_ratio, dt_splice=dt_splice)
            if dynamics:
                self.exprU_clean, self.exprS_clean, self.exprU_noisy, self.exprs_noisy = [np.zeros(shape=(number_bins, number_genes, number_sc))] * 4
            else:
                self.expr_clean, self.expr_noisy = [np.zeros(shape=(number_bins, number_genes, number_sc))] * 2
            self.data_obs = None
            self.data_inter = []
            self.exclude_inters = exclude_inters
            self.adj_matrix = np.zeros(shape=(number_genes, number_genes))
            if not var_names:
                self.var_names = np.arange(0, number_genes)
        
        def to_adjacency(self):
            adj_matrix = self.adj_matrix
            for vert, props in self.graph_.items():
                adj_matrix[vert, props["targets"]] = 1.0
                adj_matrix[props["regs"], vert] = 1.0
            return np.array(adj_matrix).astype(np.float32)

        
        def sim_clean(self, input_file_targets, input_file_regs, shared_coop_state=0):
            self.build_graph(input_file_taregts=input_file_targets, input_file_regs=input_file_regs, shared_coop_state=shared_coop_state)
            if self.dyn_:
                self.simulate_dynamics()
                self.exprU_clean, self.exprS_clean = self.getExpressions_dynamics()

            else:
                self.simulate()
                self.expr_clean = self.getExpressions()
        
        def add_noise(self, **kwargs):
            if self.dyn_:
                # Add outlier genes
                exprU_O, exprS_O = outlier_effect_dynamics(U_scData=self.exprU_clean, S_scData=self.exprS_clean, outlier_prob=kwargs['outlier_prob'], \
                                                                  mean=kwargs['outlier_mean'], scale=kwargs['outlier_scale'], nBins_=self.nBins_)
                # Add Library Size Effect
                libFactor, exprU_O_L, exprS_O_L = lib_size_effect_dynamics(U_scData=exprU_O, S_scData=exprS_O, \
                                                                                    mean=kwargs['lib_mean'], scale=kwargs['lib_scale'])
                # Add Dropouts
                binary_indU, binary_indS = dropout_indicator_dynamics(U_scData=exprU_O_L, S_scData=exprS_O_L, shape=kwargs['shape'], \
                                                                             percentile=kwargs['percentile'])
                self.exprU_noisy = np.multiply(binary_indU, exprU_O_L)
                self.exprS_noisy = np.multiply(binary_indS, exprS_O_L)
                return convert_to_UMIcounts_dynamics(U_scData=self.exprU_noisy, S_scData=self.exprS_noisy)

            else:
                # Add outlier genes
                expr_O = outlier_effect(scData=self.expr_clean, outlier_prob=kwargs['outlier_prob'], \
                                                                  mean=kwargs['outlier_mean'], scale=kwargs['outlier_scale'], nBins_=self.nBins_)
                # Add Library Size Effect
                libFactor, expr_O_L = lib_size_effect(scData=expr_O, mean=kwargs['lib_mean'], scale=kwargs['lib_scale'], nBins_=self.nBins_)
                # Add Dropouts
                binary_ind = dropout_indicator(scData=expr_O_L, shape=kwargs['shape'], \
                                                                             percentile=kwargs['percentile'])
                self.expr_noisy = np.multiply(binary_ind, expr_O_L)
                return convert_to_UMIcounts(scData=self.expr_noisy)
            

        # 'data_obs', 'data_int', 'adj_matrix', 'exclude_inters', 'var_names'  
        def simulate_obs(self, input_file_targets, input_file_regs, shared_coop_state=0, add_noise=False, noise_params=None):          
            self.sim_clean(input_file_targets, input_file_regs, shared_coop_state)
            if add_noise:
                return np.concatenate(self.add_noise(**noise_params), axis=1).astype(np.float32)
            else:
                if self.dyn_:
                    return np.concatenate(self.exprU_noisy, axis=1).astype(np.float32), np.concatenate(self.exprS_noisy, axis=1).astype(np.float32)
                else:
                    return np.concatenate(self.expr_clean, axis=1).astype(np.float32)

        @staticmethod    
        def simulate_inter(sim, input_file_targets, input_file_regs, shared_coop_state=0, add_noise=False, noise_params=None, filename=None):
            sim.sim_clean(input_file_targets, input_file_regs, shared_coop_state)
            print("Done")
            if add_noise:
                np.save(input_file_regs.removesuffix("_reg.txt") + ".npy", np.concatenate(sim.add_noise(**noise_params), axis=1).T.astype(np.float32))
            else:
                if sim.dyn_:
                    np.save(input_file_regs.removesuffix("_reg.txt") + "_U.npy", np.concatenate(sim.exprU_noisy, axis=1).T.astype(np.float32))
                    np.save(input_file_regs.removesuffix("_reg.txt") + "_S.npy", np.concatenate(sim.exprS_noisy, axis=1).T.astype(np.float32))
                else:
                    np.save(input_file_regs.removesuffix("_reg.txt") + ".npy", np.concatenate(sim.expr_clean, axis=1).T.astype(np.float32))
            
            


        def generate_data(self, input_file_targets, input_file_regs, shared_coop_state=0, add_noise=False, noise_params=None, filename=None):
            # concat obs and int, add shared noise and store
            self.data_obs = self.simulate_obs(input_file_targets, input_file_regs, shared_coop_state, add_noise, noise_params)
            self.adj_matrix = self.to_adjacency()

            if not filename:
                filename = str(uuid.uuid4())
            os.mkdir(filename)
            args = []
            sims = []
            for vert in range(self.adj_matrix.shape[0]):
                if vert not in self.exclude_inters:
                    # Intervene MRs
                    lines = np.loadtxt(input_file_regs, delimiter=",")
                    lines[np.where(lines[:, 0] == vert), 1] = 0.0
                    np.savetxt(filename + "/" + str(vert) + "_reg.txt", lines, delimiter=',')

                    # Intervene targets
                    with open(input_file_targets, "r") as f:
                        lines = f.readlines()
                    out_lines = lines
                    if len(lines) > len(out_lines):
                        out_lines[-1] = out_lines[-1].removesuffix("\n")
                        mod_line = [x for x in lines if x.startswith(str(float(vert)))][0].removesuffix("\n").split(",")
                        mod_line[2 + int(float(mod_line[1])):] = [str(0.0)] * (len(mod_line) - 2 - int(float(mod_line[1])))
                        out_lines.append("\n" + ",".join(mod_line) + "\n")
                    with open(filename + "/" + str(vert) + "_targets.txt", "w") as f:
                        f.write("".join(out_lines))
                    sims.append(copy.deepcopy(self))
                    args.append(vert)

            threads = []
            for x, vert in zip(sims, args):
                threads.append(Thread(target=x.simulate_inter, args=(x, filename + "/" + str(vert) + "_targets.txt", \
                                                                     filename + "/" + str(vert) + "_reg.txt", \
                                                                        shared_coop_state, add_noise, noise_params, filename)))
            for x in threads:
                x.start()

            for x in threads:
                x.join()
            
            for vert in range(self.adj_matrix.shape[0]):
                if vert not in self.exclude_inters:
                    data = np.load(filename + "/" + str(vert) + ".npy")
                    self.data_inter.append(data)
                else:
                    self.data_inter.append(np.array([0.0]).astype(np.float32))
            np.savez(filename + ".npz", data_obs=self.data_obs.T, data_int=self.data_inter, var_names=self.var_names, 
                     exclude_inters=self.exclude_inters, adj_matrix=self.adj_matrix)