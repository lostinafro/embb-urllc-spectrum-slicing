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
from scipy.interpolate import interpolate
from tabulate import tabulate

from common.method import watt2dbm, compute_distance
from slicing.environment import F, M, outage_target, ru, re, noise, cell

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('lines', **{'markerfacecolor': "None", 'markersize': 5})
rc('axes.grid', which='minor')

markers = ['*', 'v', 's', '^']
colors = ['darkred', 'orangered', 'darkorange', 'gold']


def power_outage(urllc_freqs: list, urllc_minis: int, urllc_rate: float, embb_rate: float, embb_snr: list, name='', render=False, verbose=True):
    # Read results
    noma_results, oma_results = read_data(urllc_freqs, urllc_minis, urllc_rate, embb_rate, name)

    de = compute_distance(- (embb_snr + noise - 30), 4, 17.15)
    embb_power_tot = np.zeros((len(embb_snr), 4))
    for e, snr_e in enumerate(embb_snr):

        param = f'$M_u = {urllc_minis}$, $r_u ={urllc_rate:.2f}$, $r_e = {embb_rate:.1f}$, $\Gamma_e = {snr_e:d}$, $d_e = {de[e]:.1f}$'
        # NOMA
        snr_u_noma = np.unique(noma_results.snr_u.to_numpy())
        avg_noma_power_init = np.zeros(snr_u_noma.shape)
        avg_noma_power_final = np.zeros(snr_u_noma.shape)
        avg_noma_outage_init = np.zeros(snr_u_noma.shape)
        avg_noma_outage_final = np.zeros(snr_u_noma.shape)
        avg_noma_power_sic = np.zeros(snr_u_noma.shape)
        avg_noma_embb_power = np.zeros(snr_u_noma.shape)
        avg_noma_urllc_power_sic = np.zeros(snr_u_noma.shape)
        avg_noma_urllc_power_init = np.zeros(snr_u_noma.shape)
        avg_noma_urllc_power_final = np.zeros(snr_u_noma.shape)
        avg_noma_urllc_power_il = np.zeros(snr_u_noma.shape)
        for i, snr_u in enumerate(snr_u_noma):
            index = (noma_results['snr_u'] == snr_u) & (noma_results['snr_e'] == snr_e)
            try:
                avg_noma_power_init[i] = np.mean(noma_results.loc[index].power_spent_init.to_numpy())
                avg_noma_power_final[i] = np.mean(noma_results.loc[index].power_spent_final.to_numpy())
                avg_noma_outage_init[i] = np.mean(noma_results.loc[index].outage_init.to_numpy())
                avg_noma_outage_final[i] = np.mean(noma_results.loc[index].outage_final.to_numpy())
                avg_noma_urllc_power_init[i] = np.mean(noma_results.loc[index].urllc_power_init.to_numpy())
                avg_noma_urllc_power_final[i] = np.mean(noma_results.loc[index].urllc_power_final.to_numpy())
            except AttributeError:
                avg_noma_power_init[i] = np.mean(noma_results.loc[index].power_spent_feasible.to_numpy())
                avg_noma_power_final[i] = 0.0
                avg_noma_outage_init[i] = np.mean(noma_results.loc[index].outage_estimated_feasible.to_numpy())
                avg_noma_outage_final[i] = 0.0
                avg_noma_urllc_power_init[i] = np.mean(noma_results.loc[index].urllc_power_feasible.to_numpy())
                avg_noma_urllc_power_final[i] = 0.0
            avg_noma_power_sic[i] = np.mean(noma_results.loc[index].power_spent_sic.to_numpy())
            avg_noma_embb_power[i] = np.mean(noma_results.loc[index].embb_power.to_numpy())
            # Subtracting embb_power for SIC and IL
            embb_power_vec = noma_results.loc[index].embb_power.to_numpy()
            avg_noma_urllc_power_sic[i] = np.mean(noma_results.loc[index].power_spent_sic.to_numpy() - embb_power_vec)
            avg_noma_urllc_power_il[i] = np.mean(noma_results.loc[index].power_spent_il.to_numpy() - embb_power_vec)

        # OMA
        snr_u_oma = snr_u_noma
        avg_oma_power_fea = np.zeros((len(urllc_freqs), len(snr_u_oma)))
        avg_oma_outage_fea = np.zeros((len(urllc_freqs), len(snr_u_oma)))
        avg_oma_embb_power = np.zeros((len(urllc_freqs), len(snr_u_oma)))
        avg_oma_urllc_power_fea = np.zeros((len(urllc_freqs), len(snr_u_oma)))
        for fo, _ in enumerate(urllc_freqs):
            for i, snr_u in enumerate(snr_u_oma):
                index = (oma_results[fo]['snr_u'] == snr_u) & (oma_results[fo]['snr_e'] == snr_e)
                avg_oma_power_fea[fo, i] = np.mean(oma_results[fo].loc[index].power_spent_feasible.to_numpy())
                avg_oma_outage_fea[fo, i] = np.mean(oma_results[fo].loc[index].outage_estimated_feasible.to_numpy())
                avg_oma_embb_power[fo, i] = np.mean(oma_results[fo].loc[index].embb_power.to_numpy())
                avg_oma_urllc_power_fea[fo, i] = np.mean(oma_results[fo].loc[index].urllc_power_feasible.to_numpy())

        # x-axis
        pl_noma = - (snr_u_noma + noise - 30)
        x_noma = compute_distance(pl_noma, 4, 17.15)
        pl_oma = - (snr_u_oma + noise - 30)
        x_oma = compute_distance(pl_oma, 4, 17.15)
        x_label = r'$d_u$ [m]'
        d_u = np.linspace(cell[1], cell[0], 20)

        # Interpolating Data to obtain distance in a linear plot
        init_index = avg_noma_outage_init <= outage_target + 2e-6
        noma_interp_init = interpolate.interp1d(x_noma[init_index], avg_noma_power_init[init_index], kind='quadratic', fill_value='extrapolate')
        noma_interp_final = interpolate.interp1d(x_noma, avg_noma_power_final, kind='quadratic', fill_value='extrapolate')
        noma_out_interp_init = interpolate.interp1d(x_noma, avg_noma_outage_init, kind='slinear', fill_value='extrapolate')
        noma_out_interp_final = interpolate.interp1d(x_noma, avg_noma_outage_final, kind='slinear', fill_value='extrapolate')
        noma_u_interp_init = interpolate.interp1d(x_noma[init_index], avg_noma_urllc_power_init[init_index], kind='quadratic', fill_value='extrapolate')
        noma_u_interp_final = interpolate.interp1d(x_noma, avg_noma_urllc_power_final, kind='quadratic', fill_value='extrapolate')
        oma_interp = []
        oma_out_interp = []
        oma_u_interp = []
        for f, _ in enumerate(urllc_freqs):
            oma_index = avg_oma_outage_fea[f] <= outage_target + 2e-6  # if f != 0 else 'cubic'
            oma_interp.append(interpolate.interp1d(x_oma[oma_index], avg_oma_power_fea[f][oma_index], kind='quadratic', fill_value='extrapolate'))
            oma_out_interp.append(interpolate.interp1d(x_oma, avg_oma_outage_fea[f], kind='slinear', fill_value='extrapolate'))
            oma_u_interp.append(interpolate.interp1d(x_oma, avg_oma_urllc_power_fea[f], kind='quadratic', fill_value='extrapolate'))

        # Plotting Total Power
        noma_power_init = watt2dbm(np.maximum(noma_interp_init(d_u), np.repeat(avg_noma_power_sic[0], len(d_u))))
        noma_power_final = watt2dbm(noma_interp_final(d_u))
        oma_power_fea = np.zeros((len(urllc_freqs), len(d_u)))
        for f, _ in enumerate(urllc_freqs):
            oma_power_fea[f] = watt2dbm(oma_interp[f](d_u))
        _, ax = plt.subplots()
        plt.plot(d_u, noma_power_init, label=r'N-fea', color='b', marker='x')
        plt.plot(d_u, noma_power_final, label=r'N-BCD', color='g', marker='o')
        for fo, freq in enumerate(urllc_freqs):
            plt.plot(d_u, oma_power_fea[fo], label=f'O-{freq}', color=colors[fo], marker=markers[fo])
        # Decorating
        ax.set_ylabel(r'$P^\mathrm{tot}$ [dBm]')
        ax.set_xlabel(x_label)
        ax.set_xlim([0, cell[0]])
        ax.legend()
        ax.grid()

        # Show
        if not render:
            plt.title(param)
            plt.show(block=verbose)
        else:
            filename = os.path.join(output_dir, f'{name}_power_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}_snre{snr_e:02d}')
            tikzplotlib.save(filename + '.tex')
            plt.title(param)
            plt.savefig(filename + '.png', dpi=300)
            plt.close()
        # Plot OUTAGE
        noma_outage_init = np.minimum(noma_out_interp_init(d_u), np.repeat(outage_target, len(d_u)))
        noma_outage_final = np.minimum(noma_out_interp_final(d_u), np.repeat(outage_target, len(d_u)))
        oma_outage_fea = np.zeros((len(urllc_freqs), len(d_u)))
        for f, _ in enumerate(urllc_freqs):
            oma_outage_fea[f] = np.minimum(oma_out_interp[f](d_u), np.repeat(outage_target, len(d_u)))
        _, ax = plt.subplots()
        plt.axhline(outage_target, linewidth=0.5, label=r'$\epsilon_u$', color='black')
        plt.semilogy(d_u, noma_outage_init, label=r'N-fea', color='b', marker='x')
        plt.semilogy(d_u, noma_outage_final, label=r'N-BCD', color='g', marker='o')
        for fo, freq in enumerate(urllc_freqs):
            plt.semilogy(d_u, oma_outage_fea[fo], label=f'O-{freq}', color=colors[fo], marker=markers[fo])
        # Decorating
        ax.set_ylabel(r'$p_u$')
        ax.set_xlabel(x_label)
        ax.legend()
        ax.grid()
        ax.set_xlim([0, cell[0]])
        plt.minorticks_on()
        # Show
        if not render:
            plt.title(param)
            plt.show(block=verbose)
        else:
            filename = os.path.join(output_dir, f'{name}_outage_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}_snre{snr_e:02d}')
            tikzplotlib.save(filename + '.tex')
            plt.title(param)
            plt.savefig(filename + '.png', dpi=300)
            plt.close()
        # Gain wrt NOMA
        _, ax = plt.subplots()
        for fo, f in enumerate(urllc_freqs):
            plt.plot(d_u, oma_power_fea[fo] - noma_power_final, label=f'O-{f}', color=colors[fo],  marker=markers[fo])
        plt.legend()
        plt.grid()
        plt.minorticks_on()
        plt.xlabel(x_label)
        plt.ylabel('Gain between NOMA and OMA [dB]')
        ax.set_xlim([0, cell[0]])
        if not render:
            plt.title(param)
            plt.show(block=verbose)
        else:
            filename = os.path.join(output_dir, f'{name}_gain_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}_snre{snr_e:02d}')
            tikzplotlib.save(filename + '.tex')
            plt.title(param)
            plt.savefig(filename + '.png', dpi=300)
            plt.close()
        # Plot URLLC power
        noma_urllc_power_fea = watt2dbm(np.maximum(noma_u_interp_init(d_u), np.repeat(avg_noma_urllc_power_sic[0], len(d_u))))
        noma_urllc_power_final = watt2dbm(noma_u_interp_final(d_u))
        oma_urllc_power_fea = np.zeros((len(urllc_freqs), len(d_u)))
        for f, _ in enumerate(urllc_freqs):
            oma_urllc_power_fea[f] = watt2dbm(oma_u_interp[f](d_u))
        _, ax = plt.subplots()
        plt.plot(d_u, noma_urllc_power_fea, label=r'N-fea', color='b', marker='x')
        plt.plot(d_u, noma_urllc_power_final, label=r'N-BCD', color='g', marker='o')
        plt.plot(x_noma, watt2dbm(avg_noma_urllc_power_sic), label=r'N-SIC', color='dimgrey', linestyle='dashed')
        plt.plot(x_noma, watt2dbm(avg_noma_urllc_power_il), label=r'N-IL', color='black', linestyle='dotted')
        for fo, freq in enumerate(urllc_freqs):
            plt.plot(d_u, oma_urllc_power_fea[fo], label=f'O-{freq}', color=colors[fo], marker=markers[fo])
        # Decorating
        ax.set_ylabel(r'$P_{u}^\mathrm{tot}$ [dBm]')
        ax.set_xlabel(x_label)
        ax.set_xlim([0, cell[0]])
        # Legend
        ax.legend()
        ax.grid()
        plt.minorticks_on()
        if not render:
            plt.title(param)
            plt.show(block=verbose)
        else:
            filename = os.path.join(output_dir, f'{name}_u_powers_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}_snre{snr_e:02d}')
            tikzplotlib.save(filename + '.tex')
            plt.title(param)
            plt.savefig(filename + '.png', dpi=300)
            plt.close()
        # Table e power
        elements = watt2dbm(np.array([np.mean(avg_noma_embb_power)] + [np.mean(avg_oma_embb_power[f]) for f, _ in enumerate(urllc_freqs)]))[np.newaxis]
        if not render:
            print(tabulate(elements, headers=['NOMA'] + [f'O-{f}' for f in urllc_freqs], floatfmt='.2f'))
        else:
            with open(os.path.join(output_dir, name + f'_embb_power_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}_snre{snr_e:2d}.tex'), 'w') as f:
                print(tabulate(elements, headers=['NOMA'] + [f'O-{f}' for f in urllc_freqs], tablefmt='latex_raw', floatfmt='.2f'), file=f)
        embb_power_tot[e] = elements[0]
    # Plot all eMBB powers
    _, ax = plt.subplots()
    plt.plot(de, embb_power_tot[:, 0], label=r'NOMA', color='b', marker='x')
    for fo, freq in enumerate(urllc_freqs):
        plt.plot(de, embb_power_tot[:, fo + 1], label=f'O-{freq}', color=colors[fo], marker=markers[fo])
    # Decorating
    ax.set_ylabel(r'$P_{e}^\mathrm{tot}$ [dBm]')
    ax.set_xlabel(r'$d_e$ [m]')
    ax.set_xlim([0, 300])
    # Legend
    ax.legend()
    ax.grid()
    plt.minorticks_on()
    if not render:
        plt.title(f'$M_u = {urllc_minis}$, $r_u ={urllc_rate:.2f}$, $r_e = {embb_rate:.1f}$')
        plt.show(block=verbose)
    else:
        filename = os.path.join(output_dir, f'{name}_e_powers_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}')
        tikzplotlib.save(filename + '.tex')
        plt.title(f'$M_u = {urllc_minis}$, $r_u ={urllc_rate:.2f}$, $r_e = {embb_rate:.1f}$')
        plt.savefig(filename + '.png', dpi=300)
        plt.close()


def embb_direction(urllc_freqs: list, urllc_minis: int, urllc_rate: float, embb_rate: float, urllc_snr: list, name='', render=False, verbose=True):
    # Read results
    noma_results, oma_results = read_data(urllc_freqs, urllc_minis, urllc_rate, embb_rate, name)

    du = compute_distance(- (urllc_snr + noise - 30), 4, 17.15)
    urllc_power_tot = np.zeros((len(urllc_snr), 4))
    for u, snr_u in enumerate(urllc_snr):

        param = f'$M_u = {urllc_minis}$, $r_u ={urllc_rate:.2f}$, $r_e = {embb_rate:.1f}$, $\Gamma_u = {snr_u:d}$, $d_e = {du[u]:.1f}$'
        # NOMA
        snr_e_noma = np.unique(noma_results.snr_e.to_numpy())
        avg_noma_power_init = np.zeros(snr_e_noma.shape)
        avg_noma_power_final = np.zeros(snr_e_noma.shape)
        avg_noma_outage_init = np.zeros(snr_e_noma.shape)
        avg_noma_outage_final = np.zeros(snr_e_noma.shape)
        avg_noma_power_sic = np.zeros(snr_e_noma.shape)
        avg_noma_embb_power = np.zeros(snr_e_noma.shape)
        avg_noma_urllc_power_sic = np.zeros(snr_e_noma.shape)
        avg_noma_urllc_power_init = np.zeros(snr_e_noma.shape)
        avg_noma_urllc_power_final = np.zeros(snr_e_noma.shape)
        avg_noma_urllc_power_il = np.zeros(snr_e_noma.shape)
        for i, snr_e in enumerate(snr_e_noma):
            index = (noma_results['snr_u'] == snr_u) & (noma_results['snr_e'] == snr_e)
            try:
                avg_noma_power_init[i] = np.mean(noma_results.loc[index].power_spent_init.to_numpy())
                avg_noma_power_final[i] = np.mean(noma_results.loc[index].power_spent_final.to_numpy())
                avg_noma_outage_init[i] = np.mean(noma_results.loc[index].outage_init.to_numpy())
                avg_noma_outage_final[i] = np.mean(noma_results.loc[index].outage_final.to_numpy())
                avg_noma_urllc_power_init[i] = np.mean(noma_results.loc[index].urllc_power_init.to_numpy())
                avg_noma_urllc_power_final[i] = np.mean(noma_results.loc[index].urllc_power_final.to_numpy())
            except AttributeError:
                avg_noma_power_init[i] = np.mean(noma_results.loc[index].power_spent_feasible.to_numpy())
                avg_noma_power_final[i] = 0.0
                avg_noma_outage_init[i] = np.mean(noma_results.loc[index].outage_estimated_feasible.to_numpy())
                avg_noma_outage_final[i] = 0.0
                avg_noma_urllc_power_init[i] = np.mean(noma_results.loc[index].urllc_power_feasible.to_numpy())
                avg_noma_urllc_power_final[i] = 0.0
            avg_noma_power_sic[i] = np.mean(noma_results.loc[index].power_spent_sic.to_numpy())
            avg_noma_embb_power[i] = np.mean(noma_results.loc[index].embb_power.to_numpy())
            # Subtracting embb_power for SIC and IL
            embb_power_vec = noma_results.loc[index].embb_power.to_numpy()
            avg_noma_urllc_power_sic[i] = np.mean(noma_results.loc[index].power_spent_sic.to_numpy() - embb_power_vec)
            avg_noma_urllc_power_il[i] = np.mean(noma_results.loc[index].power_spent_il.to_numpy() - embb_power_vec)

        # OMA
        snr_e_oma = snr_e_noma
        avg_oma_power_fea = np.zeros((len(urllc_freqs), len(snr_e_oma)))
        avg_oma_outage_fea = np.zeros((len(urllc_freqs), len(snr_e_oma)))
        avg_oma_embb_power = np.zeros((len(urllc_freqs), len(snr_e_oma)))
        avg_oma_urllc_power_fea = np.zeros((len(urllc_freqs), len(snr_e_oma)))
        for fo, _ in enumerate(urllc_freqs):
            for i, snr_e in enumerate(snr_e_oma):
                index = (oma_results[fo]['snr_u'] == snr_u) & (oma_results[fo]['snr_e'] == snr_e)
                avg_oma_power_fea[fo, i] = np.mean(oma_results[fo].loc[index].power_spent_feasible.to_numpy())
                avg_oma_outage_fea[fo, i] = np.mean(oma_results[fo].loc[index].outage_estimated_feasible.to_numpy())
                avg_oma_embb_power[fo, i] = np.mean(oma_results[fo].loc[index].embb_power.to_numpy())
                avg_oma_urllc_power_fea[fo, i] = np.mean(oma_results[fo].loc[index].urllc_power_feasible.to_numpy())

        # x-axis
        pl_noma = - (snr_e_noma + noise - 30)
        x_noma = compute_distance(pl_noma, 4, 17.15)
        pl_oma = - (snr_e_oma + noise - 30)
        x_oma = compute_distance(pl_oma, 4, 17.15)
        x_label = r'$d_e$ [m]'
        d_e = np.linspace(cell[1], cell[0], 20)

        # Interpolating Data to obtain distance in a linear plot
        init_index = avg_noma_outage_init <= outage_target + 2e-6
        noma_interp_init = interpolate.interp1d(x_noma[init_index], avg_noma_power_init[init_index], kind='quadratic', fill_value='extrapolate')
        noma_interp_final = interpolate.interp1d(x_noma, avg_noma_power_final, kind='quadratic', fill_value='extrapolate')
        noma_out_interp_init = interpolate.interp1d(x_noma, avg_noma_outage_init, kind='slinear', fill_value='extrapolate')
        noma_out_interp_final = interpolate.interp1d(x_noma, avg_noma_outage_final, kind='slinear', fill_value='extrapolate')
        noma_u_interp_init = interpolate.interp1d(x_noma[init_index], avg_noma_urllc_power_init[init_index], kind='quadratic', fill_value='extrapolate')
        noma_u_interp_final = interpolate.interp1d(x_noma, avg_noma_urllc_power_final, kind='quadratic', fill_value='extrapolate')
        sic_interp = interpolate.interp1d(x_noma, avg_noma_power_sic, kind='quadratic', fill_value='extrapolate')
        sic_u_interp = interpolate.interp1d(x_noma, avg_noma_urllc_power_sic, kind='quadratic', fill_value='extrapolate')
        il_interp = interpolate.interp1d(x_noma, avg_noma_urllc_power_il, kind='quadratic', fill_value='extrapolate')
        oma_interp = []
        oma_out_interp = []
        oma_u_interp = []
        for f, _ in enumerate(urllc_freqs):
            oma_index = avg_oma_outage_fea[f] <= outage_target + 2e-6  # if f != 0 else 'cubic'
            oma_interp.append(interpolate.interp1d(x_oma[oma_index], avg_oma_power_fea[f][oma_index], kind='quadratic', fill_value='extrapolate'))
            oma_out_interp.append(interpolate.interp1d(x_oma, avg_oma_outage_fea[f], kind='slinear', fill_value='extrapolate'))
            oma_u_interp.append(interpolate.interp1d(x_oma, avg_oma_urllc_power_fea[f], kind='quadratic', fill_value='extrapolate'))

        # Plotting Total Power
        noma_power_init = watt2dbm(np.maximum(noma_interp_init(d_e), sic_interp(d_e)))
        noma_power_final = watt2dbm(noma_interp_final(d_e))
        oma_power_fea = np.zeros((len(urllc_freqs), len(d_e)))
        for f, _ in enumerate(urllc_freqs):
            oma_power_fea[f] = watt2dbm(oma_interp[f](d_e))
        _, ax = plt.subplots()
        plt.plot(d_e, noma_power_init, label=r'N-fea', color='b', marker='x')
        plt.plot(d_e, noma_power_final, label=r'N-BCD', color='g', marker='o')
        for fo, freq in enumerate(urllc_freqs):
            plt.plot(d_e, oma_power_fea[fo], label=f'O-{freq}', color=colors[fo], marker=markers[fo])
        # Decorating
        ax.set_ylabel(r'$P^\mathrm{tot}$ [dBm]')
        ax.set_xlabel(x_label)
        ax.set_xlim([0, cell[0]])
        ax.legend()
        ax.grid()

        # Show
        if not render:
            plt.title(param)
            plt.show(block=verbose)
        else:
            filename = os.path.join(output_dir, f'{name}_power_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}_snru{snr_u:02d}')
            tikzplotlib.save(filename + '.tex')
            plt.title(param)
            plt.savefig(filename + '.png', dpi=300)
            plt.close()
        # Plot OUTAGE
        noma_outage_init = np.minimum(noma_out_interp_init(d_e), np.repeat(outage_target, len(d_e)))
        noma_outage_final = np.minimum(noma_out_interp_final(d_e), np.repeat(outage_target, len(d_e)))
        oma_outage_fea = np.zeros((len(urllc_freqs), len(d_e)))
        for f, _ in enumerate(urllc_freqs):
            oma_outage_fea[f] = np.minimum(oma_out_interp[f](d_e), np.repeat(outage_target, len(d_e)))
        _, ax = plt.subplots()
        plt.axhline(outage_target, linewidth=0.5, label=r'$\epsilon_u$', color='black')
        plt.semilogy(d_e, noma_outage_init, label=r'N-fea', color='b', marker='x')
        plt.semilogy(d_e, noma_outage_final, label=r'N-BCD', color='g', marker='o')
        for fo, freq in enumerate(urllc_freqs):
            plt.semilogy(d_e, oma_outage_fea[fo], label=f'O-{freq}', color=colors[fo], marker=markers[fo])
        # Decorating
        ax.set_ylabel(r'$p_u$')
        ax.set_xlabel(x_label)
        ax.legend()
        ax.grid()
        ax.set_xlim([0, cell[0]])
        plt.minorticks_on()
        # Show
        if not render:
            plt.title(param)
            plt.show(block=verbose)
        else:
            filename = os.path.join(output_dir, f'{name}_outage_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}_snru{snr_u:02d}')
            tikzplotlib.save(filename + '.tex')
            plt.title(param)
            plt.savefig(filename + '.png', dpi=300)
            plt.close()
        # Plot URLLC power
        noma_urllc_power_fea = watt2dbm(np.maximum(noma_u_interp_init(d_e), sic_u_interp(d_e)))
        noma_urllc_power_final = watt2dbm(noma_u_interp_final(d_e))
        oma_urllc_power_fea = np.zeros((len(urllc_freqs), len(d_e)))
        for f, _ in enumerate(urllc_freqs):
            oma_urllc_power_fea[f] = watt2dbm(oma_u_interp[f](d_e))
        _, ax = plt.subplots()
        plt.plot(d_e, noma_urllc_power_fea, label=r'N-fea', color='b', marker='x')
        plt.plot(d_e, noma_urllc_power_final, label=r'N-BCD', color='g', marker='o')
        plt.plot(d_e, watt2dbm(sic_u_interp(d_e)), label=r'N-SIC', color='dimgrey', linestyle='dashed')
        plt.plot(d_e, watt2dbm(il_interp(d_e)), label=r'N-IL', color='black', linestyle='dotted')
        for fo, freq in enumerate(urllc_freqs):
            plt.plot(d_e, oma_urllc_power_fea[fo], label=f'O-{freq}', color=colors[fo], marker=markers[fo])
        # Decorating
        ax.set_ylabel(r'$P_{u}^\mathrm{tot}$ [dBm]')
        ax.set_xlabel(x_label)
        ax.set_xlim([0, cell[0]])
        # Legend
        ax.legend()
        ax.grid()
        plt.minorticks_on()
        if not render:
            plt.title(param)
            plt.show(block=verbose)
        else:
            filename = os.path.join(output_dir, f'{name}_u_powers_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}_snru{snr_u:02d}')
            tikzplotlib.save(filename + '.tex')
            plt.title(param)
            plt.savefig(filename + '.png', dpi=300)
            plt.close()


def read_data(urllc_freqs: list, urllc_minis: int, urllc_rate: float, embb_rate: float, folder: str = '') -> tuple:
    # Select results folder
    if folder == 'adaptive':
        input_dir = os.path.join(os.path.dirname(__file__), 'results', 'feasible')
        grid_name = f'_F{F}_M{M}_Fu{12:02d}_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}'
        noma_results = pd.read_csv(os.path.join(input_dir, 'noma_metrics' + grid_name + '.csv'))
        oma_results = list()
        for f in urllc_freqs:
            grid_name = f'_F{F}_M{M}_Fu{f:02d}_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{embb_rate:.1f}'
            oma_results.append(pd.read_csv(os.path.join(input_dir, 'oma_adaptive_metrics' + grid_name + '.csv')))
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


def plot_sic_experiment(urllc_freqs: list, urllc_minis: int, urllc_rate: float, render: bool = False, verbose: bool = False):
    re_vec = [2, 4, 6]
    input_dir = os.path.join(os.path.dirname(__file__), 'results', 'feasible')
    _, ax = plt.subplots()
    styles = ['-', '--']
    for j, re in enumerate(re_vec):
        for k, Fu in enumerate(urllc_freqs):
            grid_name = f'_F{F}_M{M}_Fu{Fu:02d}_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{re:.1f}'
            results = pd.read_csv(os.path.join(input_dir, 'sic_metrics' + grid_name + '.csv'))
            # Compute sic
            snr_e = np.unique(results.snr_e)
            power_sic = np.zeros(len(snr_e))
            for i, s in enumerate(snr_e):
                index = results.snr_e == s
                power_sic[i] = np.mean(results.loc[index].power_spent_sic.to_numpy() - results.loc[index].embb_power.to_numpy())
            # Add the plot
            d_e = compute_distance(- (snr_e + noise - 30), 4, 17.15)
            plt.plot(d_e, watt2dbm(power_sic), label=f'$r_e = {re:1.0f}$, $F_u = {Fu}$', color=colors[j], marker=markers[j], linestyle=styles[k])

    # Decorating
    ax.set_ylabel(r'$\sum_{f\in\mathcal{F}_u} P_{u}^\mathrm{SIC}$ [dBm]')
    ax.set_xlabel(r'$d_e$ [m]')
    # Legend
    ax.legend()
    ax.grid()
    plt.minorticks_on()
    if not render:
        plt.title(f'$M_u = {urllc_minis}$, $r_u ={urllc_rate:.2f}$')
        plt.show(block=verbose)
    else:
        filename = os.path.join(output_dir, f'sic_plot_Mu{urllc_minis}_ru{urllc_rate:.1f}')
        tikzplotlib.save(filename + '.tex')
        plt.title(f'$M_u = {urllc_minis}$, $r_u ={urllc_rate:.2f}$')
        plt.savefig(filename + '.png', dpi=300)
        plt.close()


def plot_il_sic_comparison(urllc_minis: int, urllc_rate: float, render: bool = False, verbose: bool = False):
    re_vec = [2, 4, 6]
    input_dir = os.path.join(os.path.dirname(__file__), 'results', 'feasible')
    _, ax = plt.subplots()
    styles = ['--', ':']
    for j, re in enumerate(re_vec):
        grid_name = f'_F{F}_M{M}_Fu{12:02d}_Mu{urllc_minis}_ru{urllc_rate:.1f}_re{re:.1f}'
        results = pd.read_csv(os.path.join(input_dir, 'sic_metrics' + grid_name + '.csv'))
        # Compute sic
        snr_e = np.unique(results.snr_e)
        power_sic = np.zeros(len(snr_e))
        power_il = np.zeros(len(snr_e))
        for i, s in enumerate(snr_e):
            index = results.snr_e == s
            power_sic[i] = np.mean(results.loc[index].power_spent_sic.to_numpy() - results.loc[index].embb_power.to_numpy())
            power_il[i] = np.mean(results.loc[index].power_spent_il.to_numpy() - results.loc[index].embb_power.to_numpy())
        # Add the plot
        d_e = compute_distance(- (snr_e + noise - 30), 4, 17.15)
        plt.plot(d_e, watt2dbm(power_sic), label=f'$r_e = {re:1.0f}$', color=colors[j], marker=markers[j], linestyle=styles[0])
        plt.plot(d_e, watt2dbm(power_il), label=f'$r_e = {re:1.0f}$', color=colors[j], marker=markers[j], linestyle=styles[1])

    # Decorating
    ax.set_ylabel(r'$\sum_{f\in\mathcal{F}_u} P_{u}^\mathrm{SIC}$ [dBm]')
    ax.set_xlabel(r'$d_e$ [m]')
    # Legend
    ax.legend()
    ax.grid()
    plt.minorticks_on()
    if not render:
        plt.title(f'$M_u = {urllc_minis}$, $r_u ={urllc_rate:.2f}$')
        plt.show(block=verbose)
    else:
        filename = os.path.join(output_dir, f'sic_il_plot_Mu{urllc_minis}_ru{urllc_rate:.1f}')
        tikzplotlib.save(filename + '.tex')
        plt.title(f'$M_u = {urllc_minis}$, $r_u ={urllc_rate:.2f}$')
        plt.savefig(filename + '.png', dpi=300)
        plt.close()


def plot_outage_experiments_urllc(freqs, mean_snr, render, verbose):
    input_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    ru_list = [1, 0.5, 0.25, 1/7]
    mu_list = [1, 2, 4, 7]
    e_plots = [-np.inf, 0]
    _, ax = plt.subplots()
    for e in e_plots:
        linesty = '-' if e != -np.inf else '--'
        a = e if e != -np.inf else '-\infty'
        for i, ru in enumerate(ru_list):
            outage_csv = os.path.join(input_dir, f'mean_outage_overall_rate{ru * 12:.2f}.csv')
            # Read csv
            data = pd.read_csv(outage_csv)

            # Extract data
            for f in freqs:
                outage_df = data.loc[(data['mean_snr'] == mean_snr) & (data['FR'] == f)]
                outage = outage_df.loc[outage_df['embb_power'] == e].outage.cummin().to_numpy()
                u_power = np.unique(outage_df.urllc_power)
                # Plot noma outage
                plt.semilogy(u_power, outage, label=f'$M_u={mu_list[i]}$', linestyle=linesty, color=colors[i], marker=markers[i])
    # Axis
    ax.set_ylabel(r'$p_u$')
    ax.set_xlabel(r'$P_u(f)$ [dBm]')
    plt.ylim((1e-6, 0.5e1))
    plt.xlim((-15, 15))
    # Legend
    ax.legend()
    ax.grid()
    if not render:
        # plt.title(f'$P_e(f) = {e:.2f}$ [dBm]')
        plt.show(block=verbose)
    else:
        filename = os.path.join(output_dir, f'outage_urllc_interference')
        plt.savefig(filename + '.png', dpi=300)
        tikzplotlib.save(filename + '.tex')


def command_parser():
    """Parse command line using arg-parse and get user data to run the render.

        :return: the parsed arguments
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", type=str, default='feasible')
    parser.add_argument("-Mu", type=int, default=0)
    parser.add_argument("-snr_e", type=int, nargs='+', default=[30, 40, 50, 60, 70, 80])
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-r", "--render", action="store_true", default=False)
    args: dict = vars(parser.parse_args())
    return list(args.values())


if __name__ == '__main__':
    # General parameter
    Fu = [3, 6, 9]
    algorithm, Mu, snr_e, V, R = command_parser()

    if R:
        output_dir = os.path.join(os.path.expanduser('~'), str(date.today()))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    if algorithm == 'outage':
        plot_outage_experiments_urllc([12], 30, R, V)
    elif algorithm == 'sic':
        plot_sic_experiment([6, 12], 1, ru, R, V)
    elif algorithm == 'il':
        plot_il_sic_comparison(1, ru, R, V)
    else:
        Mu = [1, 2, 4] if Mu == 0 else [Mu]
        for m in Mu:
            power_outage(Fu, m, ru, re, snr_e, name=algorithm, render=R, verbose=V)
            # embb_direction(Fu, m, ru, re, [50], name=algorithm, render=R, verbose=V)
