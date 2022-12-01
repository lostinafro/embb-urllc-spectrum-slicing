#!/usr/bin/env python3
# file: waterfilling.py

import numpy as np


def perform_wf(algorithm: str,
               channel_snr: np.ndarray,
               bound: float, weight: np.ndarray = None) -> tuple:
    # TODO:
    #   - implement the different partition solution for bound (both min power or max rate) (Palomar?)
    #   - implement min power with weight solution
    """Water filling algorithm.

    :param algorithm: str, define the kind of algorithm to perform.
    :param channel_snr:  np.ndarray (1-D), signal-to-noise ratio at each channel of interest.
    :param bound: float, aggregate constraint value.
    :param weight: np.ndarray 1-D, user specific weight for maximization of sum rate; weight.shape must be = chan.shape.

    :returns p: numpy.ndarray 1-D, power allocated for each channel; p.shape = chan.shape.
    """
    supported_algorithms = {'minimum power', 'maximum rate'}
    # Control on input
    if algorithm not in supported_algorithms:
        raise ValueError(f'algorithm {algorithm} not supported. Supported ones are {supported_algorithms}')
    elif weight is None:
        weight = np.ones(channel_snr.shape)
    elif channel_snr.shape != weight.shape:
        raise ValueError('chan.shape and weight.shape must be equal')
    # Init
    num = channel_snr.shape[0]
    p = np.zeros(num)
    # Ordering channels
    order = np.flip(np.argsort(weight * channel_snr))
    d = 1 / weight[order] / channel_snr[order]
    if algorithm == 'maximum rate':
        # The loop commented is replaced by numpy operations ----
        # for i in range(len(chan)):
        #     p[i] = bound - np.sum((d[i] - d[:i]) * weight[:i])
        # -------------------------------------------------------
        d_vec = np.repeat(d * weight[np.newaxis], num, axis=0)
        w_vec = np.repeat(weight[np.newaxis], num, axis=0)
        d_vec[np.triu_indices(num, k=1)] = 0
        w_vec[np.triu_indices(num, k=1)] = 0
        p_max_vec = bound - d * np.sum(w_vec, axis=1) + np.sum(d_vec, axis=1)
        opt = max(np.flatnonzero(p_max_vec > 0))
        p[:opt + 1] = (p_max_vec[opt] / sum(weight[:opt + 1]) + d[opt] - d[:opt + 1]) * weight[:opt + 1]
    elif algorithm == 'minimum power':
        # The loop commented is replaced by numpy operations ----
        # for i in range(len(chan)):
        #     p[i] = (2 ** bound * np.prod(d[i])) ** (1/i) - d[i]
        # -------------------------------------------------------
        # Create the matrix containing the repetition of vector d
        d_vec = np.repeat(d[np.newaxis], num, axis=0)
        # Impose the upper triangular matrix equal to 1 for product purpose
        d_vec[np.triu_indices(num, k=1)] = 1
        hyp_vec = (2 ** bound * np.prod(d_vec, axis=1)) ** (1/np.arange(1, num + 1)) - d
        opt = max(np.flatnonzero(hyp_vec > 0))
        p[:opt + 1] = hyp_vec[opt] + d[opt] - d[:opt + 1]
    # Compute rate per user
    r = np.log2(1 + p * channel_snr[order])
    # To unsort a sorted array it is possible to use order.argsort() as indexes
    return p[order.argsort()], r[order.argsort()]
