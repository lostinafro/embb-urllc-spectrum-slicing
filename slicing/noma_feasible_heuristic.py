#!/usr/bin/env python3
# file: embb_urllc_noma.py

# Import packages
import os

import numpy as np
import pandas as pd

from common.method import np_ceil, watt2dbm, dbm2watt
# Local
from slicing.environment import SlicingEnvironment, command_parser, standard_bar, read_outage_csv, output_csv_path


class NOMAFeasible(SlicingEnvironment):
    """Class for the NOMA feasible-heuristic simulation environment"""

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
        super(NOMAFeasible, self).__init__(seed=seed,
                                           multiple_access='NOMA',
                                           embb_mean_rate=embb_mean_rate,
                                           urllc_mean_rate=urllc_mean_rate,
                                           urllc_target_outage=urllc_target_outage,
                                           mean_snr=mean_snr,
                                           urllc_frequencies=urllc_frequencies,
                                           urllc_minislots=urllc_minislots,
                                           **kwargs)

        # urllc attributes
        self.urllc_power_feasible = np.zeros(self.FR)  # power allocation of urllc user with the feasible algorithm
        self.urllc_power_heuristic = np.zeros(self.FR)  # power allocation of urllc user with the heuristic algorithm
        # Outages attributes
        self.outage_tab_feasible = np.nan * np.ones(self.FR)
        self.outage_tab_heuristic = np.nan * np.ones(self.FR)
        self.outage_estimated_feasible = 1.0
        self.outage_estimated_heuristic = 1.0
        # Tabulate attributes
        self.urllc_power_possible, self.embb_power_possible, self.outage_list = self.interp_outage(outage_database)

    @property
    def alpha_feasible(self):
        return self.urllc_power_feasible / (self.embb_power[0] + self.urllc_power_feasible)

    @property
    def alpha_heuristic(self):
        return self.urllc_power_heuristic / (self.embb_power + self.urllc_power_heuristic)

    @property
    def power_spent_feasible(self):
        return np.sum(self.embb_power) + self.urllc_minis * np.sum(self.urllc_power_feasible)

    @property
    def power_spent_heuristic(self):
        return np.sum(self.embb_power) + self.urllc_minis * np.sum(self.urllc_power_heuristic)

    def interp_outage(self, data) -> tuple:
        # TODO: complicate way to obtain a -30:step:30 matrix of interpolation IN dB SHITHEAD with first row and column with -np.inf
        urllc_power_tab, embb_power_tab, outage_tab = self.extract_data_from_csv(data)
        x_points = np.linspace(np.min(urllc_power_tab[1:]), np.max(urllc_power_tab), self.interp_points)
        y_points = np.linspace(np.min(embb_power_tab[1:]), np.max(embb_power_tab), self.interp_points)
        return dbm2watt(x_points), dbm2watt(y_points), self.interpolate_outage(urllc_power_tab[1:], x_points, outage_tab[1:, 1:], embb_power_tab[1:], y_points)

    def feasible_urllc_power_allocation(self):
        """Feasible algorithm for URLLC power allocation as described in https://arxiv.org/pdf/2106.08847.pdf."""
        if self.FR == 1:
            epsilon = self.target_outage + self.margin
            self.urllc_power_feasible = np_ceil((2 ** self.urllc_mean_rate - 1) + (self.embb_power - 1 / self.urllc_mean_snr_linear / np.log(1 - epsilon)), 4)
            self.outage_tab_feasible = self.target_outage
        else:
            try:
                # Take the first embb_power equal or greater than the maximum interference power
                ide = np.flatnonzero(np.max(self.embb_power) - self.embb_power_possible <= 0)[0]
            except IndexError:
                ide = np.abs(np.max(self.embb_power) - self.embb_power_possible).argmin()
            outage_tab_feasible = self.outage_list[ide]
            # Find minimum urllc power needed to reach target outage
            try:
                # Take the first outage that is below target outage
                ido = np.flatnonzero(outage_tab_feasible - self.target_outage <= 0)[0]
            except IndexError:
                # In case is non-existent take the nearest one
                ido = np.abs(outage_tab_feasible - self.target_outage).argmin()
            self.outage_tab_feasible = outage_tab_feasible[ido]
            self.urllc_power_feasible[self.urllc_allocation] = np.repeat(self.urllc_power_possible[ido], self.urllc_freqs)

    def heuristic_urllc_power_allocation(self):
        """Heuristic algorithm for URLLC power allocation as described in https://arxiv.org/pdf/2106.08847.pdf."""
        if self.FR == 1:
            self.urllc_power_heuristic = self.urllc_power_feasible
            self.outage_tab_heuristic = self.outage_tab_feasible
            return
        for f in range(self.FR):
            try:
                ide = np.flatnonzero(self.embb_power[0, f] - self.embb_power_possible < 0)[0]
            except IndexError:
                ide = np.abs(self.embb_power[0, f] - self.embb_power_possible).argmin()
            try:
                ido = np.flatnonzero(self.outage_list[ide] - self.target_outage < 0)[0]
            except IndexError:
                ido = np.abs(self.outage_list[ide] - self.target_outage).argmin()
            self.urllc_power_heuristic[f] = self.urllc_power_possible[ido]
            self.outage_tab_heuristic[f] = self.outage_list[ide, ido]


if __name__ == "__main__":
    # Configuration
    iterations, mean_snr_list, Fu, Mu, verbose, render, num_tests, batches, suffix = command_parser()
    # Immutable parameters
    from slicing.environment import outage_target, ru, re, F
    # Read csv
    outage_df = read_outage_csv(ru)
    # Output file
    if render:
        output_csv = output_csv_path('A_noma_metrics', 'feasible', Fu, Mu, suffix)

    # Begin simulations
    for snr in mean_snr_list:
        print(f'SNR_u={snr[0]}; SNR_e={snr[1]}')
        # TODO: transform standard_bar into parallel iteration with cupy ATTENTION TO SAVING TO FILE
        for s in standard_bar(iterations):
            # Generate environment
            env = NOMAFeasible(s, re, ru, outage_target, snr, Fu, Mu, outage_df, verbose=verbose,
                               montecarlo_tests=num_tests, number_of_batches=batches)
            # 1. eMBB power allocation through water filling
            env.embb_power_allocation(kind='adaptive')
            # 2. URLLC power allocation for SIC purpose
            env.urllc_sic_power_allocation()
            # URLLC IL scenario
            env.urllc_il_power_allocation()
            # 3. Meta water-filling
            # 3.a Algorithm using worst mean snr
            env.feasible_urllc_power_allocation()
            # Control the real outage
            env.estimate_outage(env.urllc_power_feasible)
            # # 3.b Algorithm using different mean snr per frequency (not theoretically solid but still...)
            env.heuristic_urllc_power_allocation()
            # # Control the real outage
            env.outage_estimated_heuristic = 0.0 # env.estimate_outage(env.urllc_power_heuristic)

            # Control on outage
            if verbose:
                print('\n')
                print('__________________________FINAL RESULTS_____________________________')
                print(f'outage tab feasible: {env.outage_tab_feasible:.3e}')
                print(f'outage est feasible: {env.outage_estimated_feasible:.3e}')
                print(f'power spent feasible: {watt2dbm(env.power_spent_feasible):.3f} dBm')
                print(f'embb power spent: {watt2dbm(np.sum(env.embb_power)):.3f} dBm')
                print(f'urllc power spent: {watt2dbm(Mu * np.sum(env.urllc_power_feasible)):.3f} dBm')
                if env.outage_tab_feasible - env.outage_estimated_feasible < 0:
                    print('WORST ERROR! Something wrong here!')
                print(f'outage_tab_heuristic: {" ".join(f"{x:.3e}" for x in env.outage_tab_heuristic)}')
                print(f'outage_tab_heuristic_avg: {np.mean(env.outage_tab_heuristic):.3e}')
                print(f'outage_est_heuristic: {env.outage_estimated_heuristic:.3e}')
                print(f'power_spent_heuristic: {watt2dbm(env.power_spent_heuristic):.3f} dBm')
                if np.any(env.outage_tab_heuristic - env.outage_estimated_heuristic < 0):
                    print('ALL ERROR! Something wrong here!')
            if render:
                # Print csv
                dic = [{'snr_u': snr[0],
                        'snr_e': snr[1],
                        'seed': s,
                        'power_spent_sic': env.power_spent_sic,
                        'power_spent_il': env.power_spent_il,
                        'power_spent_feasible': env.power_spent_feasible,
                        'power_spent_heuristic': env.power_spent_heuristic,
                        'outage_estimated_feasible': env.outage_estimated_feasible,
                        'outage_estimated_heuristic': env.outage_estimated_heuristic,
                        'urllc_power_feasible': env.urllc_minis * np.sum(env.urllc_power_feasible),
                        'urllc_power_heuristic': env.urllc_minis * np.sum(env.urllc_power_heuristic),
                        'embb_power': np.sum(env.embb_power)},
                       {f'outage_tab{f}': env.outage_tab_heuristic[f] for f in range(F)}]
                metrics = {}
                for d in dic:
                    for k, v in d.items():
                        metrics.setdefault(k, []).append(v)

                of = pd.DataFrame(metrics, index=[0])
                of.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
            del env
        print('\r...Done')
