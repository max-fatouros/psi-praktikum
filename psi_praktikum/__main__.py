"""
# psi-praktikum
"""
import logging

import matplotlib.pyplot as plt
import mjaf
import numpy as np
import scipy

mjaf.logging.set_handlers(
    logger_name=__name__,
)

from psi_praktikum._utils import (
    constants,
    paths,
)
from psi_praktikum._utils.mplstyles import PAPER

log = logging.getLogger(__name__)
plt.style.use(PAPER)


# plt.rcParams['text.usetex'] = True


def double_exponential(t, t_mu, t_pi):
    t_shifted = t
    return (
        np.exp(- (t_shifted / t_mu))
        - np.exp(- (t_shifted / t_pi))
    )

# def double_exponential(t, N_0, t0, t_mu, t_pi):
#     t_shifted = t - t0
#     return N_0 * (
#         np.exp(- (t_shifted / t_mu))
#         - np.exp(- (t_shifted / t_pi))
#     )


def sample_double_exponential(N_0, t_mu, t_pi):
    """
    Generated an array of N_0 samples from the double exponential distribution.
    """
    return (
        np.random.exponential(t_mu, N_0)
        - np.random.exponential(t_pi, N_0)
    )


def parse_data(filename: str) -> np.ndarray:
    """
    Parses the data from the *.Spe files.
    Returns a numpy array.
    """
    output = []

    path = paths.DATA_DIR / filename
    with open(path) as f:
        for line in f.readlines()[12:-16]:
            line = line.strip()
            output.append(int(line))

    return np.asarray(output)


def fit_simulated(N_0):
    samples = sample_double_exponential(N_0, constants.T_MU, constants.T_PI)

    samples = samples[samples < 1e-5]
    samples = samples[1e-15 < samples]

    hist, bin_edges = np.histogram(samples, 5000)
    xs = (bin_edges[:-1] + bin_edges[1:]) / 2

    # popt, pcov = scipy.optimize.curve_fit(
    #     double_exponential,
    #     xs,
    #     hist,
    #     p0=(1,1,1,1),
    #     bounds=(
    #         (0, 0, 0, 0),
    #         (np.inf, np.inf, np.inf, np.inf)
    #     )
    # )

    popt, pcov = scipy.optimize.curve_fit(
        double_exponential,
        xs,
        hist / (np.sum(hist) * (bin_edges[1] - bin_edges[0])),
        # p0=(1,1),
        bounds=(
            (0, 0),
            (np.inf, np.inf),
        ),
    )

    print(f'{popt=}')

    plt.bar(xs, hist, 0.8 * (bin_edges[1:] - bin_edges[:-1]))

    # ts = np.linspace(0, 1e-5, num=10_000)
    # true_data = 500 * double_exponential(ts, constants.T_MU, constants.T_PI)
    # hist, bin_edges = np.histogram(true_data, )
    # plt.scatter(ts, true_data)

    # plt.plot(xs, double_exponential(xs, *popt))

    plt.show()


def fit_data(filename):
    data = parse_data(filename)

    bin_edges = range(len(data) + 1)
    bin_edge_times = apply_fit(bin_edges, constants.P_1, constants.P_2)[:, 0]
    times = bin_edge_times[:-1]

    plt.bar(
        times,
        data,
        0.8 * (bin_edge_times[1:] - bin_edge_times[:-1]),
    )
    plt.xlabel('Decay time [microseconds]')

    popt, pcov = scipy.optimize.curve_fit(
        double_exponential,
        times,
        data / np.sum(data),
        bounds=(
            (0, 0),
            (np.inf, np.inf),
        ),
    )

    print(popt)



def smooth_peaks(data, sigma=10):
    return scipy.ndimage.gaussian_filter(data, sigma)


def find_peaks(data):
    peak_bin_indices, _ = scipy.signal.find_peaks(data)
    return peak_bin_indices


def linear_fit(X, Y) -> np.ndarray:
    """
    Fits the curve
    AP = Y
    where A = [X, [1,...,1]]

    Returns:
        P = [p_1, p_2]
    """
    A = np.vstack([X, np.ones(len(X))]).T
    P = np.linalg.inv(A.T @ A) @ A.T @ Y
    return P


def apply_fit(X, p_1, p_2):
    A = np.vstack([X, np.ones(len(X))]).T
    P = np.vstack([p_1, p_2])
    return A @ P


def fit_calibration(
    filename,
    times,
    visualize=False,
) -> np.ndarray:
    """
    Calibrates the signal from the time-to-digital converter (TDC).

    Args:
        filename: name of the calibration file.
        times: the known time differences fed into the TDC.
        visualize: wether or not to make plots along the way.

    Returns:
        The parameters of the fit, ordered from highest to lowest polynomial order.
    """
    data = parse_data(filename)
    data = smooth_peaks(data)
    bin_numbers = np.arange(0, len(data))
    peak_bin_indices = find_peaks(data)

    if visualize:
        plt.bar(bin_numbers, data)
        plt.vlines(
            peak_bin_indices,
            -0.1 * max(data),
            0,
            color='red',
            linestyle='dotted',
        )
        plt.xlabel('bin number')
        plt.ylabel('counts')
        plt.xlim(
            1.2 * min(peak_bin_indices),
            (1 / 1.2) * max(peak_bin_indices),
        )
        plt.show()

    parameters = linear_fit(peak_bin_indices, times)

    if visualize:
        plt.scatter(peak_bin_indices, times)
        plt.plot(
            peak_bin_indices,
            apply_fit(peak_bin_indices, *parameters),
            color='green',
            linestyle='--',
        )
        plt.xlabel('bin number')
        plt.ylabel('time [microseconds]')
        plt.show()

    return parameters


def main():
    # fit_data(
    #     "stop_S6andS7_delay_1_5_mus_fs12_50and100mm_30min.Spe",
    #     limits=(None, 3)
    # )

    fit_simulated(250_000)

    # fit_calibration(
    #     "TimeCalibration_delaytrigger_05to7us.Spe",
    #     times = [
    #         0.5,
    #         1,
    #         1.5,
    #         2,
    #         3,
    #         4,
    #         5,
    #         6,
    #         7,
    #     ]
    # )



if __name__ == '__main__':
    print('---Starting---')
    main()
    print('---Done---')
