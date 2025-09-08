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


def ff3(t, N_0, t_mu, t_pi):
    t_shifted = t
    return N_0 * (
        np.exp(- (t_shifted / t_mu))
        - np.exp(- (t_shifted / t_pi))
    )

def ff4a(t, t0, N_0, t_mu, t_pi):
    t_shifted = t - t0
    f1 = N_0 * (
        np.exp(- (t_shifted / t_mu))
        - np.exp(- (t_shifted / t_pi))
    )
    return f1

def ff4b(t, p0, N_0, t_mu, t_pi):
    t_shifted = t
    f1 = N_0 * (
        np.exp(- (t_shifted / t_mu))
        - np.exp(- (t_shifted / t_pi))
    )
    f2 = f1 + p0
    return f1


def ff(ts: np.ndarray, p0, p1, p2, p3, N0, t0, t_mu, t_pi):
    """
    The whole time-range must be passed,
    otherwise the convolutions probably won't work.
    """
    ts = ts - t0
    noise_gaussian = p0 * np.exp(-(ts**2) / (2 * p1**2))
    double_exponential = N0 * (
        np.exp(- (ts / t_mu))
        - np.exp(- (ts / t_pi))
    )
    f1 = np.convolve(double_exponential, noise_gaussian, mode='same')
    f2 = f1 + p2  # offset for random coincidences
    hadronic_gaussian = p3 * np.exp(-(ts**2) / (2 * p1**2))
    f3 = f2 + hadronic_gaussian
    return f3

def sample_double_exponential(N_0, t_mu, t_pi):
    """
    Generated an array of N_0 samples from the double exponential distribution.
    """
    return (
        np.random.exponential(t_mu, N_0)
        + np.random.exponential(t_pi, N_0)
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


def fit_simulated(
    N_0=100_000,
    fit_function=ff,
    bounds=(
        (0,0,-np.inf,0, 0,0,0,0),
        (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
    ),
):
    samples = sample_double_exponential(N_0, constants.T_MU, constants.T_PI)
    # samples += 1.75e-6
    # samples += 25e-6
    # samples += np.random.normal(0, 0.1, N_0)

    # samples = samples[samples < 3e-6]
    # samples = samples[1e-6 < samples]

    # TODO: set range explicitly
    hist, bin_edges = np.histogram(samples, 8192)
    xs = (bin_edges[:-1] + bin_edges[1:]) / 2

    # TODO: give uncertainty in the y-values
    #       (square root of the number of entries)
    # NOTE: the minimization method changes if you pass initial parameters (p0).
    #       I've had better luck by not setting it.

    # sigma = np.sqrt(
    #     [
    #         e if e != 0 else 1
    #         for e in hist
    #     ]
    # )


    # hist[hist==0] = np.nan
    fhist = hist.astype(float)
    # fhist[hist==0] = np.nan
    # sigma = np.sqrt(fhist)

    def f(popts):
        residuals = fit_function(xs, *popts) - fhist
        return np.sum(residuals**2)

    # popt, pcov = scipy.optimize.curve_fit(
    #     fit_function,
    #     xs,
    #     fhist,
    #     bounds=bounds,
    #     sigma=sigma,
    #     nan_policy='omit',
    # )


    fit = scipy.optimize.dual_annealing(
        f,
        bounds=(
            (0., 1e10),
            (0., 1e-2),
            (0., 1e-5),
        )
    )

    print(fit.x)

    # print(f'{popt=}')
    # plt.bar(xs, hist, 0.8 * (bin_edges[1:] - bin_edges[:-1]))
    # plt.plot(xs, fit_function(xs, *popt))
    # plt.show()


def fit_data(
    filename,
    fit_function=ff,
    bounds=(
        (0,0,-np.inf,0, 0,0,0,0),
        (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
    ),
):
    # TODO: cut the right hand overflow off
    data = parse_data(filename)[100:-200]

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
        ff,
        times,
        data,
        bounds=bounds,
        # bounds=(
        #     (0, 0, 0, 0, 0, -np.inf, 0, 0),
        #     (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
        #     # (0,)*8,
        #     # (np.inf,)*8,
        # ),
        # bounds=(
        #     (0, 0, 0, 0, 0),
        #     (np.inf, np.inf, np.inf, np.inf, np.inf),
            # (0,)*8,
            # (np.inf,)*8,
        # ),
    )

    print(popt)


    plt.plot(times, fit_function(times, *popt))


    plt.show()



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
        # plt.xlim(
        #     1.2 * min(peak_bin_indices),
        #     (1 / 1.2) * max(peak_bin_indices),
        # )
        plt.show()

    # TODO: find error
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

    print(">>> sim")
    fit_simulated(
        fit_function=ff3,
        # bounds=(
        #     (0,0,0),
        #     (np.inf, np.inf, np.inf)
        # )
    )
    # fit_simulated(
    #     fit_function=ff4a,
    #     bounds=(
    #         (0,0,0,0),
    #         (5e-6, np.inf, np.inf, np.inf)
    #     )
    # )

    # print(">>> data")
    # fit_data(
    #     "stop_S6andS7_delay_1_5_mus_fs12_50and100mm_30min.Spe",
    #     # "PSI_lab_2025/stop_S6andS7_delay_1_5_mus_fs12_135mm_60min_timinggivenbys7.Spe"
    #     # "PSI_lab_2025/stop_S6andS7_delay_1_5_mus_fs12_135mm_timinggivenbys7_CFD_allstat.Spe",
    # )

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
