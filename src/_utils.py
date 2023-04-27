from pathlib import Path

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal
from skimage import filters
from typing import Union


def fit(array, deg, x_vals=None):
    """Polyfit of array with given degree. Can take an interval of x-values in which to calculate output.

    Parameters
    ----------
    array : ndarray
        Array in format [points:axis] - first x then y axis, to be used for calculation,
    deg : int
        Degree of interpolation (i.e.: 3 == cubic interpolation)
    x_vals : array-like, optional
        Interval of x_values in which to work with, by default None where the min() and max() of the orig array is used

    Returns
    -------
    fitted_array: ndarray
        Array in format [points:axis] - first x then y
    """
    if x_vals is None:
        x_vals = np.linspace(min(array[:, 0]), max(array[:, 0]), len(array))
    reg = np.polyfit(array[:, 0], array[:, 1], deg)
    poly = np.poly1d(reg)
    return np.vstack((x_vals, poly(x_vals))).T


def calc_len(array: np.array) -> np.array:
    array = np.cumsum(np.linalg.norm(np.gradient(array, axis=0), axis=-1), axis=0)
    return array


def find_idx_nearest(array: np.array, value: Union[int, float]) -> int:
    """Find the index in a 1D-array closest to a given value.

    Parameters
    ----------
    array : array-like
        Array where to search in
    value : float
        Value to search for closest match

    Returns
    -------
    idx: int
        Index of array with closest match to value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def spline(array: np.array, x_new: np.array) -> np.array:
    """Creates a cubic spline of a given 2D-array and returns an array: [points:axis].
    Also takes x positions for starting and stopping and the number of points to return.

    Parameters
    ----------
    array : array-like
        2D-array which is used for the spline in the format [points:axis]
    start : int
        Start value of x
    stop : int
        Stop value of x
    x_points : int, optional
        Number of values the array should have, by default 200

    Returns
    -------
    spline_array: ndarray
        Array in format [points:axis]
    """
    spline_obj = interpolate.interp1d(array[:, 0], array[:, 1], kind="quadratic")
    return np.vstack((x_new, spline_obj(x_new))).T


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def mkdir(path: Path):
    path.mkdir(mode=0o774, exist_ok=True, parents=True)


def plot_image(array: np.array, dir: Path, filename: str, frame_nb: int, dpi: int):
    mkdir(dir)
    plt.ioff()
    plt.style.use("seaborn-whitegrid")
    plt.close()
    fig = plt.figure()
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    array = np.flipud(array).T
    array = np.where(array == 1, 0, 1)
    frame_nb = frame_nb
    plt.imshow(array, aspect="equal")
    savestr = dir / f"{filename}{frame_nb:04d}.png"
    plt.savefig(savestr, dpi=dpi)
    plt.close()


def local_thresh(frame, block_size, offset):
    # FIXME Gehört das nicht in Segmenter?
    local_thresholds = filters.threshold_local(frame, block_size, "gaussian", offset)
    thresh_f = frame - local_thresholds
    thresh_f = np.where(thresh_f < 0, 0, thresh_f)
    return thresh_f


def rot_matrix(arc_len):
    return np.array(
        ((np.cos(arc_len), -np.sin(arc_len)), (np.sin(arc_len), np.cos(arc_len)))
    ).T
