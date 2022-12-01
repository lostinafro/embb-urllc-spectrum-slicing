# file: environment.py

import argparse
# Import packages
import os
import sys

import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm import tqdm

from common.method import circ_uniform, thermal_noise, compute_distance
# Local
from optimization.waterfilling import perform_wf
from scenario.cellcluster import CellCluster
from scenario.resources import Resources
from slicing.outage_probability_experiment import batch_montecarlo_outage

# Users parameters
outage_target = 1e-5        # outage probability desired
tolerable_latency = 1e-3    # latency tolerable
# Resource block
time = 1e-3                 # slot time
F = 12                      # RB available
Fu = 9                      # RB reserved for URLLC
M = 7                       # minislots in a slot
Mu = 1                      # minislot reserved for URLLC
f0 = 2e9                    # central frequency
BW = 180e3                  # Bandwidth per RB
# Cell physics
cell = (500, 10)            # cell radii
pl_exponent = 4.0           # pathloss exponent
default_snr_e = 60                  # mean snr of the eMBB user [dB] ~ 82.6 [m]
noise = np.around(thermal_noise(F * 180e3))     # [dBm] noise power
# Numerical variables
int_points = 1000                  # number of interpolation point for the outage
default_num_test = int(1e7)         # number of montecarlo evaluation for outage estimation
default_batches = 1                 # number of batches of montecarlo tests
# Spectral efficiency
Nu_bit = 256
re = 4                               # embb spectral efficiency averaged on the resource grid assuming 12 freqs and 7 minis transmission
ru = 1/7    # Nu_bit / F / M / BW / (time/M)  # urllc spectral efficiency averaged on the resource grid assuming 12 freqs and 7 minis transmission

# Available MA schemes
available_multiple_access = {'NOMA', 'OMA'}
available_embb_allocation = {'fixed', 'adaptive'}
available_simulation_model = {'embb_fixed', 'urllc_fixed', '2d'}


# Argparse functions
def command_parser():
    """Parse command line using arg-parse and get user data to run the experiment in inference mode.

        :return: the parsed arguments
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("iterations", type=int)
    parser.add_argument("snr_u", type=int, nargs='+')
    parser.add_argument("-snr_e", "-mean_snr_embb", type=int, nargs='+', default=[default_snr_e])
    parser.add_argument("-Fu", "--urllc_frequencies", type=int, default=Fu)
    parser.add_argument("-Mu", "--urllc_minislots", type=int, default=Mu)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-r", "--render", action="store_true", default=False)
    parser.add_argument("-m", "--montecarlo_tests", type=int, default=None)
    parser.add_argument("-b", "--batches", type=int, default=None)
    parser.add_argument("-s", "--suffix", type=str, default='')
    args: dict = vars(parser.parse_args())
    # Iteration processing
    if args["iterations"] == 0:
        args["iterations"] = np.random.randint(34567, size=1)
    else:
        args["iterations"] = np.arange(args["iterations"])
    # SNR list processing
    # Transform the list in array in np.arange if needed
    snr_u = args["snr_u"] if len(args["snr_u"]) == 1 else np.arange(*args["snr_u"])
    snr_e = args["snr_e"] if len(args["snr_e"]) == 1 else np.arange(*args["snr_e"])
    # Create the matrix of all possible combination using repeat and tile
    args["mean_snr_list"] = np.stack((np.tile(snr_u, len(snr_e)), np.repeat(snr_e, len(snr_u)))).T
    del args["snr_u"], args["snr_e"]
    return args["iterations"], args["mean_snr_list"], args["urllc_frequencies"], args["urllc_minislots"], \
           args["verbose"], args["render"], args["montecarlo_tests"], args["batches"], args['suffix']


# Utilities
def standard_bar(total_iteration):
    return tqdm(total_iteration, file=sys.stdout, leave=False, ncols=60, ascii=True)


def read_outage_csv(rate, urllc_freqs: int = 12, urllc_minislot: int = 1):
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'results', f'mean_outage_overall_rate{rate * F * M / urllc_minislot:.2f}.csv'))


def output_csv_path(csv_name, subfolder, urllc_freqs, urllc_minis, suffix =''):
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', subfolder)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return os.path.join(output_dir, csv_name + f'_F{F:02d}_M{M:01d}_Fu{urllc_freqs:02d}_Mu{urllc_minis:01d}_ru{ru:.1f}_re{re:.1f}{suffix}' + '.csv')


# Classes
class SlicingEnvironment(CellCluster):
    """Class for the simulation environment regardless of type of Multiple Access"""

    def __init__(self,
                 seed: int,
                 multiple_access: str,
                 embb_mean_rate: float,
                 urllc_mean_rate: float,
                 urllc_target_outage: float,
                 mean_snr: float or np.ndarray,
                 verbose: bool = False,
                 frequencies: int = F,
                 urllc_frequencies: int = Fu,
                 central_frequency: float = f0,
                 minislot: int = M,
                 urllc_minislots: int = Mu,
                 slot_time: float = time,
                 urllc_tolerable_latency: float = tolerable_latency,
                 cell_radii: tuple = cell,
                 pathloss_exponent: float = pl_exponent,
                 noise_power: float = noise,
                 interp_points: int = int_points,
                 montecarlo_tests: int = None,
                 number_of_batches: int = None):
        # Control on input
        assert multiple_access in available_multiple_access
        # GENERAL ENVIRONMENT
        super(SlicingEnvironment, self).__init__(rb=Resources([(central_frequency, 0, frequencies)], 1, minislot, slot_time),
                                                 rng=np.random.RandomState(seed), verbose=verbose)
        # Create cell
        r_outer = np.max(cell_radii)
        r_inner = np.min(cell_radii)
        self.place_cell(1, r_outer=r_outer, r_inner=r_inner, pl_exp=pathloss_exponent, sh_std=-np.inf)
        self.cell[0].place_bs(1, 'mBS', 'HD', noise_power=noise_power)
        # Compute the distance from input mean snr and noise
        gain = self.nodes[0].gain + 2.15
        pathloss = - (mean_snr + (noise_power - 30))
        distance = compute_distance(pathloss, pathloss_exponent, gain, self.RB.wavelength.mean(), self.frau)
        # Insert users
        _, angle = circ_uniform(2, r_outer, r_inner, self.rng)
        # New axis is needed because the coordinates must be (1,2)
        coords_urllc = np.hstack((distance[0] * np.cos(angle[0]), distance[0] * np.sin(angle[0])))[np.newaxis]
        coords_embb = np.hstack((distance[1] * np.cos(angle[1]), distance[1] * np.sin(angle[1])))[np.newaxis]
        # place eMBB
        self.cell[0].place_user(1, 'UE', 'DL', coord=coords_embb, traffic='eMBB', ant=1, qos={'rate': embb_mean_rate}, noise_power=noise_power)
        # place URLLC
        self.cell[0].place_user(1, 'UE', 'DL', coord=coords_urllc, traffic='URLLC', ant=1, noise_power=noise_power,
                                qos={'rate': urllc_mean_rate, 'latency': urllc_tolerable_latency, 'outage': urllc_target_outage})
        # Insert mean snr
        self.urllc_mean_snr_db = mean_snr[0]
        # elif simu_model == 'urllc_fixed':
        #     _, angle = circ_uniform(1, r_outer, r_inner, self.rng)
        #     coords = np.hstack((distance * np.cos(angle), distance * np.sin(angle)))
        #     # place eMBB
        #     self.cell[0].place_user(1, 'UE', 'DL', traffic='eMBB', ant=1, qos={'rate': embb_mean_rate}, noise_power=noise_power)
        #     # place URLLC
        #     self.cell[0].place_user(1, 'UE', 'DL', coord=coords, traffic='URLLC', ant=1, noise_power=noise_power,
        #                             qos={'rate': urllc_mean_rate, 'latency': urllc_tolerable_latency, 'outage': urllc_target_outage})
        #     # Compute mean snr
        #     self.urllc_mean_snr_db = mean_snr
        # else:  # simu_model == 'embb_fixed':
        #     _, angle = circ_uniform(1, r_outer, r_inner, self.rng)
        #     coords = np.hstack((distance * np.cos(angle), distance * np.sin(angle)))
        #     # place eMBB
        #     self.cell[0].place_user(1, 'UE', 'DL', coord=coords, traffic='eMBB', ant=1, qos={'rate': embb_mean_rate}, noise_power=noise_power)
        #     # place URLLC: the distance must be decided from the possible ones in the outage table
        #     self.urllc_mean_snr_db = self.rng.choice(np.arange(40, 91, 5))
        #     _, angle = circ_uniform(1, r_outer, r_inner, self.rng)
        #     pathloss = - (self.urllc_mean_snr_db + (noise_power - 30))
        #     distance = compute_distance(pathloss, pathloss_exponent, gain, self.RB.wavelength.mean(), self.frau)
        #     coords = np.hstack((distance * np.cos(angle), distance * np.sin(angle)))
        #     self.cell[0].place_user(1, 'UE', 'DL', coord=coords, traffic='URLLC', ant=1, noise_power=noise_power,
        #                             qos={'rate': urllc_mean_rate, 'latency': urllc_tolerable_latency, 'outage': urllc_target_outage})
        #     # Compute mean snr
        #     # self.urllc_mean_snr_db = compute_pathloss(np.linalg.norm(self.nodes[2].coord), pathloss_exponent, gain, self.RB.wavelength.mean(), self.frau)
        self.urllc_mean_snr_linear = 10 ** (self.urllc_mean_snr_db / 10)
        # Build gain channel tensor
        self.build_chan_gain()
        if self.verbose:
            self.plot_scenario()
        # Multiple access involved
        self.multiple_access = multiple_access
        # Allocation grid
        self.embb_allocation = np.ones((minislot, frequencies), dtype=bool)
        self.urllc_allocation = np.ones(frequencies, dtype=bool)    # The URLLC got only one dimension because of the regular grid
        self.urllc_freqs = urllc_frequencies
        self.urllc_minis = urllc_minislots
        # Resources cardinality
        self.total_resources = frequencies * minislot
        self.urllc_resources = urllc_frequencies * minislot     # FDD multiplexing i.e. the frequencies are reserved fr every minislot
        self.embb_resources = self.total_resources - self.urllc_resources if self.multiple_access == 'OMA' else self.total_resources
        # eMBB attributes
        self.embb_id = self.get_user_id('eMBB')[0]
        self.embb_mean_rate = embb_mean_rate
        self.embb_chan_snr = np.broadcast_to(np.abs(self.chan_gain[:, 0, self.embb_id].flatten()) ** 2 / self.noise_power[self.embb_id], (minislot, frequencies))
        self.embb_power = np.zeros((minislot, frequencies))  # Collect the power allocation of embb user
        self.embb_rate = np.zeros((minislot, frequencies))  # Collect the rate achieved by embb user
        # URLLC common attributes
        self.urllc_id = self.get_user_id('URLLC')[0]
        self.urllc_mean_rate = urllc_mean_rate
        self.urllc_total_rate = urllc_mean_rate * self.total_resources  # Transformation because of the regular grid
        self.target_outage = urllc_target_outage
        self.urllc_chan_snr = np.abs(self.chan_gain[:, 0, self.urllc_id].flatten()) ** 2 / self.noise_power[self.urllc_id]
        # URLLC SIC attributes
        self.urllc_rate_sic = np.zeros(frequencies)  # rate for the minimum allocation for SIC decoding
        self.urllc_power_sic = np.zeros(frequencies)  # minimum power needed for SIC decoding
        # URLLC Interference Limited attributes
        self.urllc_rate_il = np.zeros(frequencies)  # rate for the minimum allocation for SIC decoding
        self.urllc_power_il = np.zeros(frequencies)  # minimum power needed for SIC decoding
        # Batch attributes
        if montecarlo_tests is None:
            montecarlo_tests = default_num_test
        if number_of_batches is None:
            number_of_batches = default_batches
        self.montecarlo_tests = montecarlo_tests
        self.number_of_batches = number_of_batches
        self.interp_points = interp_points
        # Margin
        self.margin = urllc_target_outage / 10

    @property
    def alpha_inf(self):
        return 1 - 2 ** (-self.urllc_mean_rate)  # Infimum alpha needed for eMBB decoding

    @property
    def power_spent_sic(self):
        return np.sum(self.embb_power) + self.urllc_minis * np.sum(self.urllc_power_sic)

    @property
    def power_spent_il(self):
        return np.sum(self.embb_power) + self.urllc_minis * np.sum(self.urllc_power_il)

    def embb_power_allocation(self, kind: str = 'adaptive') -> np.ndarray:
        """eMBB power allocation process using classical water-filling approach.

        :param kind: str in {'fixed', 'adaptive'}. In the first case, the first self.embb_reserved_FR are taken
                to perform power allocation, otherwise, the best channel are given to the same purpose.
        :return: np.array containing the power allocated
        """
        assert kind in available_embb_allocation
        if kind == 'fixed':
            # Take the first embb_reserved_FR for eMBB. The remaining are given to URLLC
            if self.multiple_access == 'OMA':   # If NOMA all resources are also given to eMBB
                # self.embb_allocation[:self.urllc_minis, (self.FR - self.urllc_freqs):] = False
                self.embb_allocation[:, (self.FR - self.urllc_freqs):] = False
            self.urllc_allocation[:(self.FR - self.urllc_freqs)] = False
        else:  # adaptive
            # Take the strongest embb_reserved_FR for eMBB. The remaining are given to URLLC
            descent_order = np.argsort(self.embb_chan_snr[0])[::-1]
            if self.multiple_access == 'OMA':   # If NOMA all resources are also given to eMBB
                # self.embb_allocation[:self.urllc_minis, descent_order[(self.FR - self.urllc_freqs):]] = False
                self.embb_allocation[:, descent_order[(self.FR - self.urllc_freqs):]] = False
            self.urllc_allocation[descent_order[:(self.FR - self.urllc_freqs)]] = False
        self.embb_power[self.embb_allocation], self.embb_rate[self.embb_allocation] = perform_wf('minimum power', self.embb_chan_snr[self.embb_allocation], self.embb_mean_rate * self.total_resources)
        return self.embb_power

    def urllc_sic_power_allocation(self) -> np.ndarray:
        """URLLC power allocation process for SIC requirement.
        Computed only if multiple access is NOMA, np.zeros otherwise.

        :return: np.array containing the power allocated
        """
        if self.multiple_access == 'NOMA':
            # Only the first mini slot because URLLC power is the same on each mini slot and the target ru is already been divided by Mu
            chan = self.embb_chan_snr[0, self.urllc_allocation] / (1 + self.embb_chan_snr[0, self.urllc_allocation] * self.embb_power[0, self.urllc_allocation])
            self.urllc_power_sic[self.urllc_allocation], self.urllc_rate_sic[self.urllc_allocation] = perform_wf('minimum power', chan, self.urllc_total_rate)
        return self.urllc_power_sic

    def urllc_il_power_allocation(self) -> np.array:
        """URLLC power allocation process in case of Interfered Limited scenario.
        Computed only if multiple access is NOMA, np.zeros otherwise.

        :return: np.array containing the power allocated
        """
        if self.multiple_access == 'NOMA':
            chan = 1 / np.min(self.embb_power[self.embb_power > 0]) * np.ones(self.urllc_freqs)
            chan[self.embb_power[0, self.urllc_allocation] > 0] = 1 / self.embb_power[0, self.urllc_allocation & (self.embb_power[0] > 0)]
            if np.sum(chan) > 0:
                self.urllc_power_il[self.urllc_allocation], self.urllc_rate_il[self.urllc_allocation] = perform_wf('minimum power', chan, self.urllc_total_rate)
        return self.urllc_power_il

    def estimate_outage(self,
                        urllc_power_vector: np.array,
                        montecarlo_tests: int or None = None,
                        number_of_batches: int or None = None) -> float:
        """Montecarlo estimation of the outage probability given the urllc_power_vector.

        :param urllc_power_vector: np.array 1-D of length equal to self.FR. Only the power of the urllc_allocation
            variable are considered to the estimation
        :param montecarlo_tests: int, total number of test to perform
        :param number_of_batches: int, number of different batches. Each batch of simulation will have
                montecarlo_tests/number_of_batches trials.
        """
        if montecarlo_tests is None:
            montecarlo_tests = self.montecarlo_tests
        if number_of_batches is None:
            number_of_batches = self.number_of_batches
        # Control the real outage
        return batch_montecarlo_outage(montecarlo_tests, number_of_batches, self.urllc_total_rate,
                                       np.repeat(self.urllc_mean_snr_linear, self.urllc_freqs),
                                       urllc_power_vector[self.urllc_allocation],
                                       self.embb_power[0, self.urllc_allocation])
        # Only the first mini slot because URLLC power is the same on each mini slot and the target ru is already been divided by Mu

    def interpolate_outage(self, x_to_interp: np.array, x_points: np.array, fval_to_interp: np.array,
                           y_to_interp: np.array = None, y_points: np.array = None, threshold: float = 1e-9):
        """Evaluate outage through linear interpolation.
        In case of OMA the interpolation is 1-D linear.
        In case of NOMA the interpolation involves a 2-D bi-linear process.

        :param x_to_interp: np.array, point of x to be interpolated.
        :param x_points: np.array, points where the interpolation take place.
        :param fval_to_interp: np.array, evaluation of f(x, [y]) on x_to_interpolate [y_to_interpolate].
        :param y_to_interp: np.array, point of y to be interpolated.
        :param y_points: np.array, points where the interpolation take place.
        :param threshold: float, interpolated values below this threshold are imposed to zero.
        """
        # OMA 1-D
        if self.multiple_access == 'OMA':
            output = np.interp(x_points, x_to_interp, fval_to_interp)
        # NOMA 2-D
        else:
            interp = interpolate.RectBivariateSpline(x_to_interp, y_to_interp, fval_to_interp, kx=3, ky=3)
            output = interp(x_points, y_points)
        # Set zero for values below threshold
        output[output <= threshold] = 0
        return output

    def extract_data_from_csv(self, data):
        # Take outage and powers for current normalized pathloss and frequency
        outage_df = data.loc[(np.around(data['mean_snr']) == self.urllc_mean_snr_db) & (data['FR'] == self.urllc_freqs)]
        u_power_possible = np.hstack((-np.inf, np.unique(outage_df['urllc_power'].to_numpy())))
        e_power_possible = np.unique(outage_df['embb_power'].to_numpy())
        # Impose monotonicity in values
        outage_possible = np.zeros((len(e_power_possible), len(u_power_possible)))
        for i, e in enumerate(e_power_possible):
            outage_possible[i] = np.hstack((1, outage_df.loc[outage_df['embb_power'] == e].outage.cummin().to_numpy()))
        return u_power_possible, e_power_possible, outage_possible
