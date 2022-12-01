#!/usr/bin/env python3
# file: slicing_plots.py

import argparse
# Import packages
import os
from datetime import date

import numpy as np
import pandas as pd
import tikzplotlib
from matplotlib import pyplot as plt
from matplotlib import rc

from common.method import watt2dbm

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

markers = ['*', 'v', 's']
colors = ['orangered', 'darkorange', 'gold']


def command_parser():
    """Parse command line using arg-parse and get user data to run the render.

        :return: the parsed arguments
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("-verbose", action="store_true", default=False)
    parser.add_argument("-render", action="store_true", default=False)
    parser.add_argument("-algorithm", type=str, default='')
    args: dict = vars(parser.parse_args())
    return list(args.values())


if __name__ == '__main__':
    # General parameter
    F = 12
    oma_urllc_freqs = [3, 6, 9]
    re_vec = [336]
    target = 1e-5
    rate = 6
    V, R, _ = command_parser()
    suffix = ''

    output_dir = os.path.join(os.path.expanduser('~'), str(date.today()))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    mean_snr = np.arange(0, 32, 2)

    # Plotting
    _, ax = plt.subplots()
    noma_re = np.zeros(len(re_vec))
    oma_re = np.zeros((len(oma_urllc_freqs), len(re_vec)))
    for j, re in enumerate(re_vec):
        # NOMA
        noma_results = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results', 'embb', f'noma_embb_test_F12_re{re:02d}{suffix}.csv'))
        mean_snr = np.unique(noma_results.mean_snr.to_numpy())
        avg_noma_embb_power = np.zeros(mean_snr.shape)
        for i, snr in enumerate(mean_snr):
            avg_noma_embb_power[i] = np.mean(noma_results.loc[noma_results['mean_snr'] == snr].power_spent_feasible.to_numpy())
        noma_re[j] = np.mean(avg_noma_embb_power)
        # OMA
        oma_results = list()
        avg_oma_embb_power = np.zeros((len(oma_urllc_freqs), len(mean_snr)))
        for fo, Fu in enumerate(oma_urllc_freqs):
            oma_results.append(pd.read_csv(os.path.join(os.path.dirname(__file__), 'results', 'embb', f'oma_embb_test_F12_Fu{Fu:02d}_re{re:02d}{suffix}.csv')))
            for i, snr in enumerate(mean_snr):
                avg_oma_embb_power[fo, i] = np.mean(oma_results[fo].loc[oma_results[fo]['mean_snr'] == snr].power_spent_feasible.to_numpy())
            oma_re[fo, j] = np.mean(avg_oma_embb_power[fo])

        plt.plot(mean_snr, watt2dbm(avg_noma_embb_power), label=f'NOMA-re{re:02d}', marker='x', markerfacecolor="None")
        for fo, freq in enumerate(oma_urllc_freqs):
            plt.plot(mean_snr, watt2dbm(avg_oma_embb_power[fo]), label=f'O-{freq}-re{re:02d}', marker=markers[fo], markerfacecolor="None")
    # Decorating
    x_label = r'$\Gamma_e$ [dB]'  # r'$d_u$ [m]'    # $\rho_u$ [dB]
    ax.set_ylabel(r'$P_e^{tot}$ [dBm]')
    ax.set_xlabel(x_label)
    ax.legend()
    ax.grid()
    if not R:
        plt.show(block=V)
    else:
        filename = os.path.join(output_dir, f'e_powersVSsnr')
        plt.savefig(filename + '.png', dpi=300)
        tikzplotlib.save(filename + '.tex')
    # Plotting THE MEAN
    _, ax = plt.subplots()
    plt.plot(re_vec, watt2dbm(noma_re), label=r'NOMA', color='b', marker='x')
    for fo, freq in enumerate(oma_urllc_freqs):
        plt.plot(re_vec, watt2dbm(oma_re[fo]), label=f'O-{freq}', color=colors[fo], marker=markers[fo])
    # Decorating
    x_label = r'$r_e$ [dB]'  # r'$d_u$ [m]'    # $\rho_u$ [dB]
    ax.set_ylabel(r'$P_e^{tot}$ [dBm]')
    ax.set_xlabel(x_label)
    ax.legend()
    ax.grid()
    if not R:
        plt.show(block=V)
    else:
        filename = os.path.join(output_dir, f'e_powers')
        plt.savefig(filename + '.png', dpi=300)
        tikzplotlib.save(filename + '.tex')

