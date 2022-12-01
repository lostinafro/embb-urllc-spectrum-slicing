#!/usr/bin/env python3
# file: embb_urllc_oma.py

# Import packages
import os

import numpy as np
import pandas as pd

from common.method import np_ceil, watt2dbm, dbm2watt
# Local
from slicing.environment import SlicingEnvironment, command_parser, standard_bar, read_outage_csv, output_csv_path


class OMAPreemption(SlicingEnvironment):
    """Class for the OMA simulation environment"""
    def __init__(self,
                 seed: int,
                 embb_mean_rate: float,
                 urllc_mean_rate: float,
                 urllc_target_outage: float,
                 mean_snr: float or np.ndarray,
                 urllc_frequencies: int,
                 urllc_minislots: int,
                 outage_database: pd.DataFrame,
                 **kwargs):
        super(OMAPreemption, self).__init__(seed=seed,
                                            multiple_access='OMA',
                                            embb_mean_rate=embb_mean_rate,
                                            urllc_mean_rate=urllc_mean_rate,
                                            urllc_target_outage=urllc_target_outage,
                                            mean_snr=mean_snr,
                                            urllc_frequencies=urllc_frequencies,
                                            urllc_minislots=urllc_minislots,
                                            **kwargs)
        # urllc attributes
        self.urllc_power_feasible = np.zeros(self.FR)  # power allocation of urllc user WORST algorithm
        # Outages attributes
        self.outage_tab_feasible = np.nan * np.empty(self.FR)
        self.outage_estimated = 1.0
        # Tabulated attributes
        self.urllc_power_possible, self.outage_list = self.interp_outage(outage_database)

    @property
    def alpha_feasible(self):
        return self.urllc_power_feasible / (self.embb_power[0] + self.urllc_power_feasible)

    @property
    def power_spent_feasible(self):
        return np.sum(self.embb_power) + self.urllc_minis * np.sum(self.urllc_power_feasible)

    def interp_outage(self, data) -> tuple:
        # The interpolation is weird because must be done in the dB domain, so the -inf element (index) 0) must be re-added afterward
        urllc_power_tab, _, outage_tab = self.extract_data_from_csv(data)
        x_points = np.linspace(np.min(urllc_power_tab[1:]), np.max(urllc_power_tab[1:]), self.interp_points)
        # Take only outage_list[0] because is the portion of array with 0 interference
        interpolated_outage = np.hstack((1, self.interpolate_outage(urllc_power_tab[1:], x_points, outage_tab[0, 1:])))
        interpolated_power = np.hstack((0.0, dbm2watt(x_points)))
        return interpolated_power, interpolated_outage

    def feasible_urllc_power_allocation(self):
        if self.urllc_freqs == 1:
            epsilon = self.target_outage + self.margin
            self.urllc_power_feasible[self.urllc_allocation] = np_ceil(- (2 ** self.urllc_mean_rate - 1) / self.urllc_mean_snr_linear / np.log(1 - epsilon), 4)
            self.outage_tab_feasible = self.target_outage
        else:
            # Find minimum urllc power needed to reach target outage
            try:
                ido = np.flatnonzero(self.outage_list - self.target_outage < 0)[0]
            except IndexError:
                ido = np.abs(self.outage_list - self.target_outage).argmin()
            self.outage_tab_feasible = self.outage_list[ido]
            self.urllc_power_feasible[self.urllc_allocation] = np.repeat(self.urllc_power_possible[ido], self.urllc_freqs)


if __name__ == "__main__":
    # Command parser
    iterations, mean_snr_list, Fu, Mu, verbose, render, num_tests, batches, suffix = command_parser()
    # Immutable parameters
    from slicing.environment import outage_target, ru, re
    # Read csv
    outage_df = read_outage_csv(ru)
    # Output file
    if render:
        output_csv = output_csv_path('oma_fixed_metrics', 'feasible', Fu, Mu, suffix)

    # Begin simulations
    for snr in mean_snr_list:
        print(f'SNR_u={snr[0]}; SNR_e={snr[1]}')
        for s in standard_bar(iterations):
            # Generate environment
            env = OMAPreemption(s, re, ru, outage_target, snr, Fu, Mu, outage_df, verbose=verbose,
                                montecarlo_tests=num_tests, number_of_batches=batches)
            #  1. eMBB power allocation through water filling
            env.embb_power_allocation(kind='fixed')
            # URLLC power allocation feasible algorithm
            env.feasible_urllc_power_allocation()
            # Control the real outage
            env.outage_estimated = env.estimate_outage(env.urllc_power_feasible)

            # Control on outage
            if verbose:
                print('\n')
                print('__________________________FINAL RESULTS_____________________________')
                print(f'outage_tab_feasible: {env.outage_tab_feasible:.3e}')
                print(f'outage_est_feasible: {env.outage_estimated:.3e}')
                print(f'power_spent: {watt2dbm(env.power_spent_feasible):.3f} dBm')
                print(f'embb power spent: {watt2dbm(np.sum(env.embb_power)):.3f} dBm')
                print(f'urllc power spent: {watt2dbm(Mu * np.sum(env.urllc_power_feasible)):.3f} dBm')
                if env.outage_tab_feasible - env.outage_estimated < 0:
                    print('ERROR! Something wrong here!')
            if render:
                # Print csv
                dic = [{'snr_u': snr[0],
                        'snr_e': snr[1],
                        'seed': s,
                        'power_spent_feasible': env.power_spent_feasible,
                        'outage_estimated_feasible': env.outage_estimated,
                        'outage_tab': env.outage_tab_feasible,
                        'urllc_power_feasible': env.urllc_minis * np.sum(env.urllc_power_feasible),
                        'embb_power': np.sum(env.embb_power),
                        }]
                metrics = {}
                for d in dic:
                    for k, v in d.items():
                        metrics.setdefault(k, []).append(v)

                of = pd.DataFrame(metrics, index=[0])
                of.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
            del env
        print('\r...Done')
