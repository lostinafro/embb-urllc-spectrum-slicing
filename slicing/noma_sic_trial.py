#!/usr/bin/env python3
# file: embb_urllc_noma.py

# Import packages
import os

import numpy as np
import pandas as pd

from common.method import watt2dbm
# Local
from slicing.environment import command_parser, standard_bar, read_outage_csv, output_csv_path
from slicing.noma_feasible_heuristic import NOMAFeasible

if __name__ == "__main__":
    # Configuration
    iterations, mean_snr_list, Fu, Mu, verbose, render, num_tests, batches, suffix = command_parser()
    # Immutable parameters
    from slicing.environment import outage_target, ru, re

    # Read csv
    outage_df = read_outage_csv(ru)
    # Output file
    if render:
        output_csv = output_csv_path('sic_metrics', 'feasible', Fu, Mu, suffix)

    # Begin simulations
    for snr in mean_snr_list:
        print(f'SNR_u={snr[0]}; SNR_e={snr[1]}')
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
            # env.feasible_urllc_power_allocation()
            # # Control the real outage
            # env.outage_estimated_feasible = env.estimate_outage(env.urllc_power_feasible)
            # # # 3.b Algorithm using different mean snr per frequency (not theoretically solid but still...)
            # env.heuristic_urllc_power_allocation()
            # # # Control the real outage
            # env.outage_estimated_heuristic = env.estimate_outage(env.urllc_power_heuristic)

            # Control on outage
            if verbose:
                print('\n')
                print('__________________________FINAL RESULTS_____________________________')
                print(f'urllc power spent: {watt2dbm(Mu * np.sum(env.urllc_power_sic)):.3f} dBm')
                print(f'urllc power spent: {watt2dbm(Mu * np.sum(env.urllc_power_il)):.3f} dBm')
            if render:
                # Print csv
                dic = [{'snr_u': snr[0],
                        'snr_e': snr[1],
                        'seed': s,
                        'power_spent_sic': env.power_spent_sic,
                        'power_spent_il': env.power_spent_il,
                        'embb_power': np.sum(env.embb_power)},
                       ]
                metrics = {}
                for d in dic:
                    for k, v in d.items():
                        metrics.setdefault(k, []).append(v)

                of = pd.DataFrame(metrics, index=[0])
                of.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
            del env
        print('\r...Done')
