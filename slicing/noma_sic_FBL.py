#!/usr/bin/env python3
# file: embb_urllc_noma.py

# Import packages
import os
from datetime import date

import numpy as np
import tikzplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.special import erfinv

# Local
from slicing.environment import command_parser, standard_bar, read_outage_csv, output_csv_path
from slicing.noma_feasible_heuristic import NOMAFeasible

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('lines', **{'markerfacecolor': "None", 'markersize': 5})
rc('axes.grid', which='minor')

def qfuncinv(x):
    return np.sqrt(2) * erfinv(1 - 2*x)

def rate_fbl(capacity, penalty, blocklength, error_e):
    return capacity - np.sqrt(penalty / blocklength) * qfuncinv(error_e)

def penalty_parallel(sinr, power):
    x = power[power > 0] * sinr[power > 0]
    return len(x) - np.sum((1 + x)**(-2))


if __name__ == "__main__":
    # Configuration
    iterations, mean_snr_list, Fu, Mu, verbose, render, num_tests, batches, suffix = command_parser()
    # Immutable parameters
    from slicing.environment import outage_target, ru, re

    # Read csv
    outage_df = read_outage_csv(ru)
    # Output file
    if render:
        output_csv = output_csv_path('sic_metrics_fbl', 'feasible', Fu, Mu, suffix)

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
            # FBL rate
            outage_e = 1e-2
            n = np.arange(1, 1001)
            chan = env.embb_chan_snr[0] / (1 + env.embb_chan_snr[0] * env.embb_power[0])
            C = np.sum(np.log2(1+ chan * env.urllc_power_sic))
            V = penalty_parallel(chan, env.urllc_power_sic)
            R = rate_fbl(C, V, n, outage_e)

            _, ax = plt.subplots()
            plt.plot(n, R/C)

            # Decorating
            ax.set_ylabel(r'$R_{u,e}(\epsilon_e) / I_{u,e}$')
            ax.set_xlabel(r'blocklength $m$')
            ax.grid()

            # Show
            output_dir = os.path.join(os.path.expanduser('~'), 'OneDrive/Slicing/plots', str(date.today()))
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            filename = os.path.join(output_dir, f'FBL_ru{ru:.1f}_re{re:.1f}_snre{snr[1]:02d}')
            tikzplotlib.save(filename + '.tex')
            plt.title(r'FBL Rate vs blocklength using $P^{SIC}_u$')
            plt.savefig(filename + '.png', dpi=300)
            plt.close()



            # # Control on outage
            # if verbose:
            #     print('\n')
            #     print('__________________________FINAL RESULTS_____________________________')
            #     print(f'urllc power spent: {watt2dbm(Mu * np.sum(env.urllc_power_sic)):.3f} dBm')
            #     print(f'urllc power spent: {watt2dbm(Mu * np.sum(env.urllc_power_il)):.3f} dBm')
            # if render:
            #     # Print csv
            #     dic = [{'snr_u': snr[0],
            #             'snr_e': snr[1],
            #             'seed': s,
            #             'power_spent_sic': env.power_spent_sic,
            #             'power_spent_il': env.power_spent_il,
            #             'embb_power': np.sum(env.embb_power)},
            #            ]
            #     metrics = {}
            #     for d in dic:
            #         for k, v in d.items():
            #             metrics.setdefault(k, []).append(v)
            #
            #     of = pd.DataFrame(metrics, index=[0])
            #     of.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
            del env
        print('\r...Done')
