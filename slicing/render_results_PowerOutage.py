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
from scipy.signal import savgol_filter

from common.method import dbm2watt, compute_distance
from slicing.environment import F, M, ru, re, noise

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('lines', **{'markerfacecolor': "None", 'markersize': 5})
rc('axes.grid', which='minor')

markers = ['*', 'v', 's', '^']
colors = ['darkred', 'orangered', 'darkorange', 'gold']

def smoothdata(x):
    return savgol_filter(x, window_length = 51, polyorder = 1)


def power_outage(urllc_freqs: list, urllc_minis: int, urllc_rate: float, embb_rate: float, embb_snr: list, urllc_snr: list, name='', render=False, verbose=True):
    # Read results
    noma_results, oma_results = read_data(urllc_freqs, urllc_minis, urllc_rate, embb_rate, name)

    max_power_budget = 46   # dBm
    power_budget_dBm = np.linspace(10, max_power_budget, 1000)
    power_budget = dbm2watt(power_budget_dBm)
    x_interp = np.linspace(10, max_power_budget, 100000)

    # Results
    for e, snr_e in enumerate(embb_snr):
        for u, snr_u in enumerate(urllc_snr):
            du = compute_distance(- (snr_u + noise - 30), 4, 17.15)
            de = compute_distance(- (snr_e + noise - 30), 4, 17.15)
            # NOMA
            index = (noma_results['snr_u'] == snr_u) & (noma_results['snr_e'] == snr_e)
            power_noma_fea = noma_results.loc[index].power_spent_feasible.to_numpy()
            samples = len(power_noma_fea)
            noma_prob = np.sum(np.tile(power_noma_fea, (len(power_budget), 1)) > np.tile(power_budget[np.newaxis].T, (1, samples)), 1) / samples

            # noma_interp = interpolate.interp1d(power_budget, noma_prob, kind='quadratic', fill_value='extrapolate')
            # OMA
            oma_prob = []
            oma_interp = []
            for fo, _ in enumerate(urllc_freqs):
                index = (oma_results[fo]['snr_u'] == snr_u) & (oma_results[fo]['snr_e'] == snr_e)
                power_oma = oma_results[fo].loc[index].power_spent_feasible.to_numpy()
                samples = len(power_oma)
                oma_prob.append(np.sum(np.tile(power_oma, (len(power_budget), 1)) > np.tile(power_budget[np.newaxis].T, (1, samples)), 1) / samples)
                # oma_interp.append(interpolate.interp1d(power_budget, oma_prob[fo], kind='quadratic', fill_value='extrapolate'))
            oma_prob = np.array(oma_prob)
            # smooth the data!

            _, ax = plt.subplots()
            plt.semilogy(power_budget_dBm, smoothdata(noma_prob), label=r'N-fea', color='b', marker='x', markevery=50)
            # plt.semilogy(watt2dbm(x_interp), noma_interp(x_interp), label=r'N-fea', color='b', marker='x', markevery=1000)
            for fo, freq in enumerate(urllc_freqs):
                plt.semilogy(power_budget_dBm, smoothdata(oma_prob[fo]), label=f'O-{freq}', color=colors[fo], marker=markers[fo], markevery=50)
                # plt.semilogy(watt2dbm(x_interp), oma_interp[fo](x_interp), label=f'O-{freq}', color=colors[fo], marker=markers[fo], markevery=50)
            # Decorating
            ax.set_ylabel(r'Power outage probability')
            ax.set_xlabel('Power budget [dBm]')
            title = f'$d_e = {de}$ {snr_e}, $d_u = {du}$ {snr_u}'
            ax.legend()
            ax.grid()
            if not render:
                plt.title(title)
                plt.show(block=verbose)
            else:
                filename = os.path.join(output_dir, f'PowerOutage_snru{snr_u:02d}_snre{snr_e:02d}')
                tikzplotlib.save(filename + '.tex')
                plt.title(title)
                plt.savefig(filename + '.png', dpi=300)
                plt.close()


def read_data(urllc_freqs: list, urllc_minis: int, urllc_rate: float, embb_rate: float, folder: str = '') -> tuple:
    # Select results folder
    if folder == 'adaptive':
        input_dir = os.path.join(os.path.dirname(__file__), 'results', 'feasible')
        grid_name = f'_F{F}_M{M}_Fu{12:02d}_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}'
        noma_results = pd.read_csv(os.path.join(input_dir, 'A_noma_metrics' + grid_name + '.csv'))
        oma_results = list()
        for f in urllc_freqs:
            grid_name = f'_F{F}_M{M}_Fu{f:02d}_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}'
            oma_results.append(pd.read_csv(os.path.join(input_dir, 'A_oma_adaptive_metrics' + grid_name + '.csv')))
    elif folder == 'fixed':
        input_dir = os.path.join(os.path.dirname(__file__), 'results', 'feasible')
        grid_name = f'_F{F}_M{M}_Fu{12:02d}_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}'
        noma_results = pd.read_csv(os.path.join(input_dir, 'noma_metrics' + grid_name + '.csv'))
        oma_results = list()
        for f in urllc_freqs:
            grid_name = f'_F{F}_M{M}_Fu{f:02d}_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}'
            oma_results.append(pd.read_csv(os.path.join(input_dir, 'oma_fixed_metrics' + grid_name + '.csv')))
    elif folder == 'optimization':
        input_noma = os.path.join(os.path.dirname(__file__), 'results', 'optimization')
        input_oma = os.path.join(os.path.dirname(__file__), 'results', 'feasible')
        grid_name = f'_F{F}_M{M}_Fu{12:02d}_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}'
        try:
            noma_results = pd.read_csv(os.path.join(input_noma, 'noma_exhaustive_metrics' + grid_name + '.csv'))
        except FileNotFoundError:
            noma_results = pd.read_csv(os.path.join(input_oma, 'noma_metrics' + grid_name + '.csv'))
        oma_results = list()
        for f in urllc_freqs:
            grid_name = f'_F{F}_M{M}_Fu{f:02d}_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}'
            oma_results.append(pd.read_csv(os.path.join(input_oma, f'oma_adaptive_metrics' + grid_name + '.csv')))
    elif folder == 'rand_urllc':    # deprecated
        input_noma = os.path.join(os.path.dirname(__file__), 'results', 'optimization')
        input_oma = os.path.join(os.path.dirname(__file__), 'results', 'feasible')
        grid_name = f'_F{F}_M{M}_Fu{12:02d}_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}'
        try:
            noma_results = pd.read_csv(os.path.join(input_noma, 'noma_exhaustive_metrics' + grid_name + '_rand_urllc.csv'))
        except FileNotFoundError:
            noma_results = pd.read_csv(os.path.join(input_oma, 'noma_metrics' + grid_name + '_rand_urllc.csv'))
        oma_results = list()
        for f in urllc_freqs:
            grid_name = f'_F{F}_M{M}_Fu{f:02d}_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}'
            oma_results.append(pd.read_csv(os.path.join(input_oma, 'oma_adaptive_metrics' + grid_name + '_rand_urllc.csv')))
    else:
        return None, None
    return noma_results, oma_results


def command_parser():
    """Parse command line using arg-parse and get user data to run the render.

        :return: the parsed arguments
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("-snr_e", type=int, nargs='+', default=[30, 40, 50])
    parser.add_argument("-snr_u", type=int, nargs='+', default=[30, 40, 50])
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-r", "--render", action="store_true", default=False)
    args: dict = vars(parser.parse_args())
    return list(args.values())


if __name__ == '__main__':
    # General parameter
    Fu = [3]
    snr_e, snr_u, V, R = command_parser()

    algorithm = 'adaptive'
    Mu = 1

    if R:
        output_dir = os.path.join(os.path.expanduser('~'), str(date.today()))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    power_outage(Fu, Mu, ru, re, snr_e, snr_u, name=algorithm, render=R, verbose=V)

