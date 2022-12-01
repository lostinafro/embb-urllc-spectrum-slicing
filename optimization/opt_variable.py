# file: opt_variable.py

import warnings
from collections import deque

import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate


class OptVariableData:
    """Create a deque list which gives at index [-i] the variables of iteration t-i of the algorithm.
       The indexing is NOT PYTHONIC, it serves as a translator to always have the right values.
    """
    # TODO: make the printing per each iteration, make the get_item working as dictionary
    def __init__(self, var_number: int, iteration_number: int = None, var_names: list or None = None, *args):
        """Constructor of the class

        :param var_number: int, number of variable to save per iteration
        :param iteration_number: int, number of iteration to keep track of
        :param var_names: list of str, name of the variable to be saved, for presentation purpose
        """
        # Control on input
        assert isinstance(var_number, int)
        # Attributes
        self.var_number = var_number
        self.var_names = var_names if var_names is not None else [str('var')] * var_number
        self.iteration_number = iteration_number
        self._current_iteration = 0
        self.var = deque(*args, maxlen=self.iteration_number)
        warnings.filterwarnings("ignore", category=FutureWarning, message='elementwise comparison failed; returning scalar instead')

    @property
    def straight(self):
        list_me = list(self.var)
        return list_me[1::] + list_me[:1:]

    @property
    def current_iteration(self):
        return self._current_iteration

    def update(self, variables: tuple or list, render: bool = False):
        """Insert the new variables at time 0. Eliminate the last element if needed.

        :param variables: tuple or list, new element to be inserted
        :param render: bool, set if print the results of the current iteration
        """
        assert len(variables) == self.var_number
        self.var.rotate(-1)
        if len(self.var) < self.iteration_number:
            self.var.appendleft(variables)
        else:      # maxlen has been violated
            self.var[0] = variables
        self._current_iteration += 1
        if render:
            print(tabulate([self.straight[-1]], headers=self.var_names, showindex=[self._current_iteration], numalign="right", tablefmt="plain"))

    def plot(self, var_name: str, filename: str or None = None, **kwargs):
        """Plot a variable as a function of the algorithm iterations. It works only for 1-D variables right now.

        :param var_name: str, name of the variable to be plotted. Must be the same of the pone in self.var_names
        :param filename: str or None, if present it defines the filename (absolute or relative path) of where saving
                        the plot.
        """
        plt.plot(np.arange(self._current_iteration), self[var_name], **kwargs)
        if filename is None:
            plt.show(block=False)
        else:
            plt.savefig(filename + ".png", dpi=300)

    def __getitem__(self, item: int or str):
        if type(item) == int:
            """Meant to be used for iteration extraction, ordered from current time backward"""
            return self.var[item]
        elif type(item) == str:
            """Meant to be used for variable extraction, ordered by iteration"""
            return [self.straight[ind][self.var_names.index(item)] for ind in range(len(self))]
        else:
            raise IndexError(f'Index item can be int or str not {type(item)}')

    def __len__(self):
        return len(self.var)

    def __str__(self):
        if not len(self):
            index_id = [0]
            values = [[np.nan] * self.var_number]
        else:
            index_id = np.arange(-len(self) + 1, 1)
            values = self.straight
        return tabulate(values, headers=self.var_names, showindex=index_id, numalign="right", tablefmt="plain")

    def __repr__(self):
        return f'OptVariableData: iteration saved {min(self._current_iteration, self.iteration_number):d}, number of var per iteration: {self.var_number}'


if __name__ == '__main__':
    a = OptVariableData(2, 3)
    for i in range(4):
        a.update([i, i+2*0.5])
        print(a)
