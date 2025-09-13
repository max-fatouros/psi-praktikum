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


def ff3(t, N0, t_mu, t_pi):
    t_shifted = t
    return N0 * (
        np.exp(- (t_shifted / t_mu))
        - np.exp(- (t_shifted / t_pi))
    )

def ff4a(t, t0, N0, t_mu, t_pi):
    t_shifted = t - t0
    f1 = N0 * (
        np.exp(- (t_shifted / t_mu))
        - np.exp(- (t_shifted / t_pi))
    )
    return f1

def ff4b(t, p0, N0, t_mu, t_pi):
    t_shifted = t
    f1 = N0 * (
        np.exp(- (t_shifted / t_mu))
        - np.exp(- (t_shifted / t_pi))
    )
    f2 = f1 + p0
    return f2

def ff5(t, p0, t0, N0, t_mu, t_pi):
    t_shifted = t - t0
    f1 = N0 * (
        np.exp(- (t_shifted / t_mu))
        - np.exp(- (t_shifted / t_pi))
    )
    f2 = f1 + p0
    return f2


def pexp(x):
    return np.piecewise(x, x > 0, [0, np.exp])

def ff(ts: np.ndarray, p1, t0, N0, t_mu, t_pi):
    """
    The whole time-range must be passed,
    otherwise the convolutions probably won't work.
    """
    t_shifted = ts - t0
    # f1 = (
    #     (
    #         (1/2)
    #         * pexp(
    #             -(1/t_mu)
    #             * (
    #                 t_shifted
    #                 - ((p1**2) / (2*t_mu))
    #             )
    #         )
    #         * (
    #             1
    #             + scipy.special.erf(
    #                 (t_shifted - ((p1**2)/t_mu))
    #                 / (np.sqrt(2) * p1)
    #             )
    #         )
    #     ) - (
    #         (1/2)
    #         * pexp(
    #             -(1/t_pi)
    #             * (
    #                 t_shifted
    #                 - ((p1**2)/(2*t_pi))
    #             )
    #         )
    #         * (
    #             1
    #             + scipy.special.erf(
    #                 (t_shifted - ((p1**2)/t_pi))
    #                 / (np.sqrt(2) * p1)
    #             )
    #         )
    #     )
    # )


    double_exponential = N0 * (
        pexp(- (t_shifted / t_mu))
        - pexp(- (t_shifted / t_pi))
    )
    f1 = scipy.ndimage.gaussian_filter1d(double_exponential, p1)
    return f1


    # f2 = f1 + p2  # offset for random coincidences
    # hadronic_gaussian = p3 * np.exp(-(t_shifted**2) / (2 * p4**2))
    # f2 = f1 + hadronic_gaussian
    # return f2

def sample_double_exponential(N0, t_mu, t_pi):
    """
    Generated an array of N0 samples from the double exponential distribution.
    """
    return (
        np.random.exponential(t_mu, N0)
        + np.random.exponential(t_pi, N0)
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


def chi_squared(fit_function, xs, ys, dys=None):
    def residual(parameters):
        residuals = fit_function(xs, *parameters) - ys
        if dys is not None:
            residuals *= (1/dys)
        return np.sum(residuals**2)

    return residual



def fit_simulated(
    N0=100_000,
    fit_function=ff,
    bounds=(
        (0,0,-np.inf,0, 0,0,0,0),
        (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
    ),
):
    samples = sample_double_exponential(N0, constants.T_MU, constants.T_PI)
    samples += 1.75e-6



    uniform_background = np.random.uniform(high=1e-5, size=N0 // 10)
    samples = np.hstack(
        [
            samples,
            uniform_background,
        ]
    )

    # TODO: set range explicitly
    # hist, bin_edges = np.histogram(samples, 8192)
    data, bin_edges = np.histogram(samples, 8192, range=(0, 1e-5))
    data = data.astype(float)


    mean_background = np.mean(data[100:500])

    cut = 1000
    data = data[cut:]
    bin_edges = bin_edges[cut:]
    # bin_edge_times = apply_fit(bin_edges, constants.P_1, constants.P_2)[:, 0]
    # bin_edge_times *= 1e6  # microseconds to seconds
    times = bin_edges[:-1]


    mask = data > 0
    data = data[mask]
    times = times[mask]
    sigmas = 1/np.sqrt(data)

    data -= mean_background


    # data = scipy.ndimage.gaussian_filter1d(data, 5)



    plt.bar(
        times,
        data,
        0.8 * (bin_edges[1:][mask] - bin_edges[:-1][mask])
    )

    # TODO: give uncertainty in the y-values
    #       (square root of the number of entries)
    # NOTE: the minimization method changes if you pass initial parameters (p0).
    #       I've had better luck by not setting it.


    # hist[hist==0] = np.nan
    # fhist -= mean_uniform
    # fhist[hist==0] = np.nan


    # popt, pcov = scipy.optimize.curve_fit(
    #     fit_function,
    #     xs,
    #     hist,
    #     p0=(1, 1e-5, 50, 0, 0),
    #     # bounds=bounds,
    #     # sigma=sigma,
    #     # nan_policy='omit',
    # )
    # print(popt)


    # fit = scipy.optimize.least_squares(
    #     lambda p: fit_function(xs, *p) - fhist,
    #     (2e-6,1,1e-6,1e-6),
    #     bounds=(
    #         (1e-6, 0, 1e-10, 1e-10),
    #         (5e-6, 1e3, 1e-5, 1e-5),
    #         # (0., 10),
    #         # (1e-6, 5e-6),
    #         # (0., 1e3),
    #         # (1e-10, 1e-5),
    #         # (1e-10, 1e-5),
    #     ),
    #     # maxiter=100_000
    # )
    # fit = scipy.optimize.dual_annealing(
    #     # chi_squared(fit_function, times, data),
    #     chi_squared(fit_function, times, data, dys=sigmas),
    #     bounds=bounds,
    #     # maxiter=10_000,
    # )
    # fit = scipy.optimize.minimize(
    #     chi_squared(fit_function, xs, hist, dys=sigmas),
    #     x0=(24, 8.6e-6, 6.8e2, 4.6e-6, 5.12e-6),
    # )



    popt = fit.x
    print(fit)
    plt.plot(times, fit_function(times, *popt), color='red')
    plt.show()

    plt.bar(
        xs,
        (fit_function(xs, *popt) - hist) / sigma,
        0.8 * (bin_edges[1:] - bin_edges[:-1])
    )

    return popt


def fit_data(
    filename,
    fit_function=ff,
    bounds=(
        (0,0,-np.inf,0, 0,0,0,0),
        (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
    ),
):
    # TODO: cut the right hand overflow off
    # data = parse_data(filename)[:-200]
    data = parse_data(filename)[:-3000]

    bin_edges = range(len(data) + 1)

    data = data.astype(float)



    mean_background = np.mean(data[100:1000])

    cut = 200
    data = data[cut:]
    bin_edges = bin_edges[cut:]
    bin_edge_times = apply_fit(bin_edges, constants.P_1, constants.P_2)[:, 0]
    bin_edge_times *= 1e-6  # microseconds to seconds
    times = bin_edge_times[:-1]

    # mask = data > 0
    # data = data[mask]
    # times = times[mask]
    # sigmas = 1/np.sqrt(data)
    # sigmas = data**2

    data -= mean_background





    # data = scipy.ndimage.gaussian_filter1d(data, 5)


    plt.bar(
        times,
        data,
        0.8 * (bin_edge_times[1:] - bin_edge_times[:-1]),
        # 0.8 * (bin_edge_times[1:][mask] - bin_edge_times[:-1][mask]),
    )
    plt.xlabel('Decay time [microseconds]')
    # plt.show()
    # exit()

    fit = scipy.optimize.dual_annealing(
        chi_squared(fit_function, times, data),
        # chi_squared(fit_function, times, data, dys=sigmas),
        bounds=bounds,
    )

    popt = fit.x
    print(fit)
    plt.plot(times, fit_function(times, *popt), color='red')
    xs = np.linspace(0, 1e7)
    # plt.plot(xs, fit_function(xs, 0,0,popt[-3],popt[-2],popt[-1]), color='red')
    plt.show()
    return popt



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

    # print(">>> sim")
    # fit_simulated(
    #     N0=1_000_000,
    #     fit_function=ff4a,
    #     bounds=(
    #         (0., 1e-5),
    #         (0., 1000),
    #         (0, 1e-5),
    #         (0, 1e-5),
    #     ),
    # )

    # fit_simulated(
    #     N0=1_000_000,
    #     fit_function=ff4b,
    #     bounds=(
    #         (0., 5),
    #         (0., 1000),
    #         (0, 1e-5),
    #         (0, 1e-5),
    #     ),
    # )
    # fit_simulated(
    #     N0=1_000_000,
    #     fit_function=ff5,
    #     bounds=(
    #         (0., 30),
    #         (1.5e-6, 1e-5),
    #         (0., 1000),
    #         (0, 1e-5),
    #         (0, 1e-5),
    #     ),
    # )
    # fit_data(
    #     # "stop_S6andS7_delay_1_5_mus_fs12_50and100mm_30min.Spe",
    #     "PSI_lab_2025/stop_S6andS7_delay_1_5_mus_fs12_135mm_60min_timinggivenbys7.Spe",
    #     # "PSI_lab_2025/stop_S6andS7_delay_1_5_mus_fs12_135mm_timinggivenbys7_CFD_allstat.Spe",
    #     fit_function=ff4a,
    #     bounds=(
    #         (0., 1e-5),
    #         (0., 1000),
    #         (0, 1e-5),
    #         (0, 1e-5),
    #     ),
    # )
    # exit()
    # fit_simulated(
    #     N0=1_000_000,
    #     fit_function=ff,
    #     bounds=(
    #         (0., 100),  # time gaus sigma
    #         # (0., 5),  # y shift
    #         # (0., 50),  # hadronic gaus amplitude
    #         # (0., 100),  # hadronic gaus sigma
    #         (0, 1e-5),  # t0
    #         (0., 1e3),  # N0
    #         (0., 1e-5),  # t_mu
    #         (0., 1e-7),  # t_pi
    #     ),
    # )

    # exit()
    print(">>> data")
    # fit_data(
    #     # "stop_S6andS7_delay_1_5_mus_fs12_50and100mm_30min.Spe",
    #     "PSI_lab_2025/stop_S6andS7_delay_1_5_mus_fs12_135mm_60min_timinggivenbys7.Spe",
    #     # "PSI_lab_2025/stop_S6andS7_delay_1_5_mus_fs12_135mm_timinggivenbys7_CFD_allstat.Spe",
    #     fit_function=ff,
    #     bounds=(
    #         # (0, 10),
    #         # (0, 1e-5),
    #         # (0., 1e2),
    #         # (0, 1e-2),
    #         # (0, 1e-5),
    #         (0., 100),  # time gaus sigma
    #         # (0., 5),  # y shift
    #         # (0., 50),  # hadronic gaus amplitude
    #         # (0., 100),  # hadronic gaus sigma
    #         (1e-6, 2e-6),  # t0
    #         (0., 1e3),  # N0
    #         (0., 1e-5),  # t_mu
    #         (0., 1e-7),  # t_pi
    #     ),
    # )

    # exit()

    fit_data(
        "stop_S6andS7_delay_1_5_mus_fs12_50and100mm_30min.Spe",
        # "PSI_lab_2025/stop_S6andS7_delay_1_5_mus_fs12_135mm_60min_timinggivenbys7.Spe",
        # "PSI_lab_2025/stop_S6andS7_delay_1_5_mus_fs12_135mm_timinggivenbys7_CFD_allstat.Spe",
        fit_function=ff,
        bounds=(
            # (0, 10),
            # (0, 1e-5),
            # (0., 1e2),
            # (0, 1e-2),
            # (0, 1e-5),
            (0., 100),  # time gaus sigma
            # (0., 5),  # y shift
            # (0., 100),  # hadronic gaus amplitude
            # (0., 100),  # hadronic gaus sigma
            (0, 5e-6),  # t0
            (0., 100),  # N0
            (0., 1e-5),  # t_mu
            (0., 1e-5),  # t_pi
        ),
    )


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
