#!/usr/bin/env python3
# file:

# Import packages

import argparse

import numpy as np

try:
    import cupy as cp
except ImportError:
    import numpy as cp
import pandas as pd
import os
from tqdm import tqdm
import sys
from common.method import dbm2watt, touch_csv


def command_outage():
    """Parse command line using arg-parse and get user data to run the experiment in inference mode.

            :return: the parsed arguments
        """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--overall_rate", type=float, default=1)
    parser.add_argument("-F", "--frequencies", nargs='+', type=int, default=[12])
    parser.add_argument("-mean_snr", nargs='+', type=int, default=[40, 96, 5])
    parser.add_argument("-montecarlo_tests", type=int, default=int(1e7))
    parser.add_argument("-batches", type=int, default=1)
    args: dict = vars(parser.parse_args())
    # SNR list processing
    if len(args["mean_snr"]) > 1:
        args["mean_snr"] = np.arange(*args["mean_snr"])
    else:
        args["mean_snr"] = np.array(args["mean_snr"])
    return [val for val in args.values()]


def batch_montecarlo_outage(tot_num_test: int,
                            division: int,
                            target_rate: float,
                            mean_snr: np.ndarray,
                            useful: np.ndarray,
                            interference: np.ndarray,
                            rng=cp.random.RandomState()):
    # cupy explicit conversion
    mean_snr = cp.asarray(mean_snr)
    useful = cp.asarray(useful)
    interference = cp.asarray(interference)
    # Simulation
    test_per_batch = int(tot_num_test / division)
    outage_per_division = cp.zeros(division)
    for i in range(division):
        gain = rng.exponential(scale=cp.repeat(mean_snr[np.newaxis], test_per_batch, axis=0))
        information = cp.sum(cp.log2(1 + useful * gain / (1 + gain * interference)), axis=1)
        outage_per_division[i] = cp.mean(information < target_rate)
    try:
        output = cp.asnumpy(cp.mean(outage_per_division))
        cp.cuda.Stream.null.synchronize()
    except AttributeError:
        output = np.mean(outage_per_division)
    return float(output)


if __name__ == '__main__':
    # Physical values
    overall_rate, F, mean_snr_dB, num_test, batches = command_outage()

    # Csv touching
    file_name = f'mean_outage_overall_rate{overall_rate:.2f}'
    column_names = ['FR', 'mean_snr', 'urllc_power', 'embb_power', 'outage']
    csv, output_path = touch_csv(file_name, 'results', column_names, path=__file__,)

    # Simulation
    urllc_power_dBm = np.round(np.arange(-30, 31, 1), 2)
    embb_power_dBm = np.hstack((-np.inf, np.arange(-30, 31, 1), 2))
    urllc_power = dbm2watt(urllc_power_dBm)
    embb_power = dbm2watt(embb_power_dBm)
    mean_outage = np.zeros((len(F), len(embb_power), len(urllc_power)))
    for ind, f in enumerate(F):
        for snr in mean_snr_dB:
            print(f'F = {f:d}, SNR = {snr}')
            snr_vec = np.repeat(10 ** (snr / 10), f)
            with tqdm(total=len(urllc_power) * len(embb_power), leave=False, file=sys.stdout, ascii=True) as bar:
                for j in range(len(embb_power)):
                    check_if_zero = 0
                    for i in range(len(urllc_power)):
                        # control if the result is not present into the data frame
                        if csv.loc[(csv['FR'] == f) &
                                   (csv['mean_snr'] == snr) &
                                   (csv['urllc_power'] == urllc_power_dBm[i]) &
                                   (csv['embb_power'] == embb_power_dBm[j])].empty:
                            # Control if outage has been zero twice in the past
                            if check_if_zero >= 2:
                                mean_outage[ind, j, i] = 0
                            else:
                                # run simulation
                                mean_outage[ind, j, i] = batch_montecarlo_outage(num_test, batches, overall_rate, snr_vec, urllc_power[i], embb_power[j])
                                if mean_outage[ind, j, i] == 0:
                                    check_if_zero += 1
                        else:
                            mean_outage[ind, j, i] = np.nan
                        bar.update(1)

                    output_df = pd.DataFrame({'FR': np.repeat(f, len(mean_outage[ind, j])),
                                              'mean_snr': np.repeat(snr, len(mean_outage[ind, j])),
                                              'urllc_power': urllc_power_dBm,
                                              'embb_power': np.repeat(embb_power_dBm[j], len(mean_outage[ind, j])),
                                              'outage': mean_outage[ind, j]})
                    output_df.dropna().to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    # POSTPROCESSING
    written_df = pd.read_csv(output_path)
    # Delete duplicates
    written_df.drop_duplicates(subset=['FR', 'mean_snr', 'embb_power', 'urllc_power'], inplace=True, ignore_index=True)
    # Sort the csv and re-save it
    written_df.sort_values(by=column_names).to_csv(output_path, index=False)
