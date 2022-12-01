#!/usr/bin/env python3
# filename "resources.py"

from collections import UserList

import numpy as np
from scipy.constants import c


class Resources(UserList):
    """Frequency resources available in the simulation.
    TODO:
        - THE INPUT MUST BE TUPLE OF ARRAY NOT LIST OF TUPLE U PIECE OF SHIT
        - I need to learn how to subclass numpy I guess
        - overlapping frequencies should not be allowed
        - consider the timing according to numerology and NR standard
    Parameters
    ----------
    :param blocks: list of tuple [(f_0,mu_0,sub_i), ..., (f_i,mu_i,sub_i), ...] where
        f_i, mu_i, sub_i are initial frequency [Hz], numerology and number of
        subcarriers for resource block i, respectively.
    :param slot: int, number of slot considered.
    :param minislot: number of minislot in each slot.
    :param slot_time: duration of a slot in seconds.
    """
    def __init__(self, blocks: list, slot: int, minislot: int = 14, slot_time: float = 1e-3):
        super().__init__()
        try:
            for f in range(len(blocks)):
                self.data.append(ResourceBlock(blocks[f][0], blocks[f][1], blocks[f][2]))
            self.slot = slot
            self.minislot = minislot
            self.slot_time = slot_time
            self.time = self.slot_time * self.slot
            self.wavelength = c / np.array([r.f0 + np.arange(-r.N_sc // 2, r.N_sc // 2) * r.d_f for r in self.data]).flatten()
        except TypeError:
            raise TypeError(f'positional argument RB must be a list of tuples')

    # if (mu in range(4)) and (N_RB > 275):
    #     raise ValueError("For numerology mu in {0,..,3}, max N_RB = 275")
    # elif (mu == 4) and (N_RB > 138):
    #     raise ValueError("For numerology mu = 4, max N_RB = 138")
    # elif (mu == 5) and (N_RB > 69):
    #     raise ValueError("For numerology mu = 5, max N_RB = 69")
    # else:
    def __repr__(self):
        return '\n'.join([f'{i}  {obj}' for i, obj in enumerate(self)])


class ResourceBlock:
    """5G-NR resource block definition."""
    def __init__(self, f0=2e9, mu=0, sub=1):
        if type(f0) is not float:
            raise TypeError('Initial frequency must be float')
        elif type(mu) is not int or mu < 0 or mu > 5:
            raise TypeError('Numerology must be int from 0 to 5')
        elif type(sub) is not int or sub < 1:
            raise TypeError('# of subcarrier must be int > 0')
        self.f0 = f0
        self.mu = mu
        self.N_sc = sub
        self.d_f = 2 ** mu * 180 * 1e3      # TODO: solve this issue. Makes no sense now...
        self.d_t = 1 / 2 ** mu  # ms
        self.BW = self.d_f * self.N_sc

    def __lt__(self, other):
        return True if self.f0 < other.f0 else False

    def __repr__(self):
        return (f'(f0={self.f0:.2e}, mu={self.mu:1}, '
                f'N_sc={self.N_sc:02}, BW={self.BW * 1e-3:.0f} [kHz])')


class Codeword:
    """Codeword definition? Work in progress
    TODO: each codeword has some property regarding the maximum number of portion of it that can be canceled without
     losing the whole codeword. The portion is in term of minislot (or generally the minimum quantum of time
     considered in the resource allocation process)
    Parameters
    __________
    cla : int,
        class of codeword representing the maximum portions of codeword which can be loose before going into outage.
    size : int,
        the length of the codeword given in minislot.
    """
    def __init__(self, cla: int, size: int):
        assert cla <= size
        # Force instances to be integers
        self.cla = int(cla)
        self.size = int(size)
        self.pun_count = 0

    def pun(self):
        self.pun_count += 1

    @property
    def isoutage(self):
        return True if self.pun_count > self.cla else False

    def __repr__(self):
        return (f'cw({self.cla}, {self.size}, {self.pun_count}, {self.isoutage})')



# for i in range(prob1):
# l.append(class1)
# for i in range(prob2):
# l.append(class2)
# for i in range...
#     for class_idx in range(classes):
#         for i in range(prob[class_idx]):
#             l.append(
#
#
#         class [i])
# normalizzi per sicurezza
# valori normalizzati * 100
# random.choice(list)