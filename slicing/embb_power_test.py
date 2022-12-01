#!/usr/bin/env python3
# file: embb_urllc_oma.py

# Import packages
import os

import numpy as np
import pandas as pd

from common.method import watt2dbm
from slicing.environment import command_parser, standard_bar, read_outage_csv, output_csv_path
# Local
from slicing.noma_feasible_heuristic import NOMAFeasible
from slicing.oma_fixed_preemption import OMAPreemption

if __name__ == "__main__":
    # Command parser
    iterations, mean_snr_list, Fu, Mu, verbose, render, num_tests, batches, suffix = command_parser()
    mean_snr_list = np.array(0, mean_snr_list[1])
    # Immutable parameters
    from slicing.environment import outage_target, ru, F
    # Read csv
    outage_df = read_outage_csv(ru)

    for re in [1, 2, 3, 4, 5, 6]:
        print(f're: {re}: --------------------------------------------------------')
        for Fu_OMA in [3, 6, 9, 12]:
            print(f'Fu: {Fu_OMA} --------------------------------------------------------')
            # Output file
            if render:
                csv_name = 'embb_test'
                if Fu_OMA == F:
                    output_csv = output_csv_path('noma_embb_test', 'embb', Fu_OMA, Mu, suffix)
                else:
                    output_csv = output_csv_path('oma_embb_test', 'embb', Fu_OMA, Mu, suffix)

            # Begin simulations
            for snr in mean_snr_list:
                print(f'SNR: {snr[1]}:')
                for s in standard_bar(iterations):
                    if Fu_OMA == F:
                        env = NOMAFeasible(s, re, ru, outage_target, snr, Fu, Mu, outage_df, verbose=verbose,
                                           montecarlo_tests=num_tests, number_of_batches=batches)
                    else:
                        # Generate environment
                        env = OMAPreemption(s, re, ru, outage_target, snr, Fu_OMA, Mu, outage_df, verbose=verbose,
                                            montecarlo_tests=num_tests, number_of_batches=batches)
                    #  eMBB power allocation through water filling
                    env.embb_power_allocation(kind='adaptive')

                    # Control on outage
                    if verbose:
                        print('\n')
                        print('__________________________FINAL RESULTS_____________________________')
                        print(f'embb_power_spent: {watt2dbm(env.power_spent_feasible):.3f} dBm')
                    if render:
                        # Print csv
                        dic = [{'snr_e': snr[1],
                                'seed': s,
                                'power_spent_feasible': env.power_spent_feasible}]
                        metrics = {}
                        for d in dic:
                            for k, v in d.items():
                                metrics.setdefault(k, []).append(v)

                        of = pd.DataFrame(metrics, index=[0])
                        of.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
                    del env
                print('\r...Done')
