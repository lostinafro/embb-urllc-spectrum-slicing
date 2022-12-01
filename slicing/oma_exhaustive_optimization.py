#!/usr/bin/env python3
# file: embb_urllc_noma.py

# Import packages
import os

import numpy as np
import pandas as pd

from common.method import watt2dbm
from optimization.opt_variable import OptVariableData
from slicing.environment import command_parser, standard_bar, read_outage_csv, output_csv_path
# Local
from slicing.oma_fixed_preemption import OMAPreemption


class OMAExhaustive(OMAPreemption):
    """Class for the simulation environment"""
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
        super(OMAExhaustive, self).__init__(seed=seed,
                                            embb_mean_rate=embb_mean_rate,
                                            urllc_mean_rate=urllc_mean_rate,
                                            urllc_target_outage=urllc_target_outage,
                                            mean_snr=mean_snr,
                                            urllc_frequencies=urllc_frequencies,
                                            urllc_minislots=urllc_minislots,
                                            outage_database=outage_database,
                                            **kwargs)
        # Exhaustive search attributes
        self.urllc_power_exhaustive = np.zeros(self.FR)
        self.outage_estimated_exhaustive = 1.0

    @property
    def power_spent_exhaustive(self):
        return np.sum(self.embb_power) + self.urllc_minis * np.sum(self.urllc_power_exhaustive)

    # INITIAL POINT EVALUATION
    def get_feasible_initial_point(self):
        # 1. eMBB power allocation through water filling
        self.embb_power_allocation(kind='adaptive')
        # 2. URLLC power allocation for SIC purpose
        self.urllc_sic_power_allocation()
        # 3. Compute feasible solution
        self.feasible_urllc_power_allocation()
        # 4. starting points
        self.urllc_power_feasible = np.maximum(self.urllc_power_feasible, self.urllc_power_sic)
        self.outage_estimated_feasible = self.estimate_outage(self.urllc_power_feasible)
        # Add power if tabulated solution is unfeasible (depending on the snr it could happen)
        iteration = 0
        while self.outage_estimated_feasible > self.target_outage:
            self.urllc_power_feasible = self.urllc_power_feasible * 1.1
            self.outage_estimated_feasible = self.estimate_outage(self.urllc_power_feasible)
            iteration += 1
            if iteration > 20:
                break
        self.urllc_power_exhaustive = self.urllc_power_feasible.copy()
        self.outage_estimated_exhaustive = self.outage_estimated_feasible
        return self.urllc_power_feasible.copy(), self.outage_estimated_exhaustive

    # EXHAUSTIVE SEARCH ALGORITHM
    def step_searching(self, starting_point: np.array, step: float, direction: np.array) -> tuple:
        """Take a step in a direction.

        :param starting_point: np.array, the initial feasible solution
        :param step: float, quantity of the step
        :param direction: np.array bool, give the direction (1 if direction 0 otherwise)
        """
        point = starting_point.copy()
        point[direction] = np.maximum(0, point[direction] - step)
        outage_estimated_current = self.estimate_outage(point)
        if outage_estimated_current <= self.target_outage:
            return True, point, outage_estimated_current
        else:
            return False, point, outage_estimated_current

    def exhaustive_search(self,
                          initial_step_size: float = 1e-3,
                          max_iteration: int = 100,
                          step_threshold: float = 1e-6) -> OptVariableData:
        """Exhaustive search algorithm of the minimum power needed

        :param max_iteration: int, maximum number of iteration allowed
        :param initial_step_size: float, the initial step size of the descent algorithm
        :param step_threshold: float, the minimum step size untile the algorithm stop

        The idea of the algorithm is the following:
        Starting from the best channel (less interference) take a step in that direction:
        - if the step lead to an unfeasible solution, take a step back else continue;
        - if all the steps leads to unfeasible solution the step size is reduced and the algorithm continues.
        The algorithms stops when the step_size is lower of step_threshold
        """
        # List of point, value, gradient and outage at each iteration assuming I want to save 10 iteration
        saving_list = OptVariableData(4, max_iteration + 1, var_names=['x', 'f(x)', 'outage', 'step_size'])
        # Algorithm initialization
        iteration = 0
        step_size = initial_step_size
        stop_condition = False
        # Direction matrix
        order = np.flatnonzero(self.urllc_allocation)
        direction_mat = np.eye(self.FR, dtype=bool)
        # Power variables
        urllc_power_current = self.urllc_power_exhaustive.copy()
        outage_current = self.outage_estimated_exhaustive
        saving_list.update([urllc_power_current, np.sum(urllc_power_current), outage_current, step_size], render=self.verbose)
        # Exit if initial solution is unfeasible
        if outage_current > self.target_outage:
            return saving_list
        possible_directions = order
        while not stop_condition:
            iteration += 1
            count_unfeasible = 0
            for f in possible_directions:
                is_feasible, power_obtained, outage_estimated = self.step_searching(urllc_power_current, step_size, direction_mat[f])
                if is_feasible:
                    outage_current = outage_estimated
                    urllc_power_current = power_obtained.copy()
                    # If power current is less than the one needed for the SIC
                    if urllc_power_current[f] <= self.urllc_power_sic[f]:
                        urllc_power_current[f] = self.urllc_power_sic[f]
                        possible_directions = possible_directions[possible_directions != f]
                else:   # if not is_feasible:
                    count_unfeasible += 1
                    if count_unfeasible == len(possible_directions):
                        step_size = step_size / 2
                        if step_size <= step_threshold:
                            stop_condition = True
            # Save data
            saving_list.update([urllc_power_current, np.sum(urllc_power_current), outage_current, step_size], render=self.verbose)
            if (iteration >= max_iteration) or (not possible_directions.size):
                stop_condition = True
        # When exit the loop
        self.urllc_power_exhaustive = urllc_power_current.copy()
        # self.outage_estimated_exhaustive = outage_current
        self.outage_estimated_exhaustive = self.estimate_outage(urllc_power_current, montecarlo_tests=int(1e8), number_of_batches=10)
        # Logging data
        if self.verbose:
            print('\n')
            print('_________________Exhaustive ended________________')
            print(f'Feasible URLLC power: {watt2dbm(np.sum(self.urllc_power_feasible)):.3f} dBm')
            print(f'Total URLLC power: {watt2dbm(np.sum(self.urllc_power_exhaustive)):.3f} dBm')
            print(f'Final outage esti: {self.outage_estimated_exhaustive:.2e}')
            print(f'Power saved: {watt2dbm(np.sum(self.urllc_power_feasible) - np.sum(self.urllc_power_exhaustive)):.3f} dBm')
            print('_________________________________________________')
        return saving_list


if __name__ == "__main__":
    # Command parser
    iterations, mean_snr_list, Fu, Mu, verbose, render, num_tests, batches, suffix = command_parser()
    # Immutable parameters
    from slicing.environment import outage_target, ru, re, F
    # Read csv
    outage_df = read_outage_csv(ru)
    # Output file
    if render:
        output_csv = output_csv_path('oma_exhaustive_metrics', 'optimization', Fu, Mu, suffix)

    # Simulation parameter
    threshold = 1e-6
    step_size = 1e-3
    max_iter = 100
    # Starting simulations
    for snr in mean_snr_list:
        print(f'SNR_u={snr[0]}; SNR_e={snr[1]}')
        for s in standard_bar(iterations):
            # Generate environment
            env = OMAExhaustive(s, re, ru, outage_target, snr, F, Fu, outage_df, verbose=verbose,
                                montecarlo_tests=num_tests, number_of_batches=batches)
            # Get a feasible solution and its outage
            feasible_power, initial_estimated_outage = env.get_feasible_initial_point()
            # exhaustive search solution
            ex_data = env.exhaustive_search(initial_step_size=step_size, max_iteration=max_iter, step_threshold=threshold)
            # Printing results
            if verbose:
                print('\n')
                print('__________________________FINAL RESULTS_____________________________')
                print(f'Exhaustive algorithm stopped at iteration {ex_data.current_iteration}:')
                print(f'\tInitial power: {watt2dbm(env.power_spent_feasible):.3f} dBm')
                print(f'\tInitial outage: {initial_estimated_outage:.3e}')
                print(f'\tEnd power: {watt2dbm(env.power_spent_exhaustive):.3f} dBm')
                print(f'\tEnd outage: {env.outage_estimated_exhaustive:.3e}')
                print('\n')
            if render:
                # Print csv
                dic = [{'snr_u': snr[0],
                        'snr_e': snr[1],
                        'seed': s,
                        'power_spent_sic': env.power_spent_sic,
                        'power_spent_il': env.power_spent_il,
                        'power_spent_init': env.power_spent_feasible,
                        'power_spent_final': env.power_spent_exhaustive,
                        'outage_init': env.outage_estimated_feasible,
                        'outage_final': env.outage_estimated_exhaustive,
                        'urllc_power_init': env.urllc_minis * np.sum(env.urllc_power_feasible),
                        'urllc_power_final': env.urllc_minis * np.sum(env.urllc_power_exhaustive),
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
