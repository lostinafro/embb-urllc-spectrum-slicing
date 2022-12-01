#!/usr/bin/env python3
# file: embb_urllc_oma.py

# Import packages
import os

import numpy as np
import pandas as pd

from common.method import watt2dbm
from slicing.environment import command_parser, standard_bar, read_outage_csv, output_csv_path
# Local
from slicing.oma_fixed_preemption import OMAPreemption

if __name__ == "__main__":
    # Command parser
    iterations, mean_snr_list, Fu, Mu, verbose, render, num_tests, batches, suffix = command_parser()
    # Immutable parameters
    from slicing.environment import outage_target, ru, re
    # Read csv
    outage_df = read_outage_csv(ru)
    # Output file
    if render:
        output_csv = output_csv_path('A_oma_adaptive_metrics', 'feasible', Fu, Mu, suffix)

    # Begin simulations
    for snr in mean_snr_list:
        print(f'SNR_u={snr[0]}; SNR_e={snr[1]}')
        for s in standard_bar(iterations):
            # Generate environment
            env = OMAPreemption(s, re, ru, outage_target, snr, Fu, Mu, outage_df, verbose=verbose,
                                montecarlo_tests=num_tests, number_of_batches=batches)
            #  1. eMBB power allocation through water filling
            env.embb_power_allocation(kind='adaptive')
            # URLLC power allocation feasible algorithm
            env.feasible_urllc_power_allocation()
            # Control the real outage
            env.outage_estimated = 0.0 # env.estimate_outage(env.urllc_power_feasible)

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
