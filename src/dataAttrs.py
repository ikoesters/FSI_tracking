from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DataAttrs:

    # GetData
    """
    Parameters
    ----------
    path_film : str
        Path to the high-speed film
    path_angles : str
        Path to the file containing the position feedback from the resolver in .dat format
    angle_to_film_time_resolution : int
        Capture frequency difference between film and position feedback, by default 4
    stop_frame : int
        Process only until a specific frame is reached, by default 100
    intermediates_save_path: str
        Where to save the intermediate steps which are helpful for debugging. If None, no saving occurs.
    denoise_to_binarize_percentile : float
        Percentile at which a pixel brightness value is considered signal, everything below will be set to zero, by default 99
    """
    data_folder: Path = None
    cine_file: Path = None
    angle_file: Path = None
    stop_frame: int = None
    angle_to_film_time_factor: int = 4
    intermediates_save_path: Path = Path("intermediates/")
    denoise_to_binarize_percentile: float = 99
    plot_intermediates: bool = True
    frame_shape: tuple = None
    pos: np.array = None

    # Data2Midline
    """
    Parameters
    ----------
    measurement_obj : type[Measurement]
        Measurement object located in measurement.py
    dattr_obj : type[dattr]
        dattr object located in dattr.py
    sum_how_many_frames : int
        Amount of frames to be stacked to increase the seed count, by default 6
    crop_window_x : tuple
        X-coordinates to crop to region where foil is located; to decrease computation effort, by default (300, 1150)
    crop_window_y : tuple
        Y-coordinates to crop to region where foil is located; to decrease computation effort, by default (0, 800)
    """
    sum_how_many_frames: int = 6
    conv_diam: int = 20
    dilation_diam: int = 10
    nb_flex_sample_points: int = 4  # 4 points for 3rd degree fit
    nb_rigid_sample_points: int = 3
    sample_points: int = nb_flex_sample_points + nb_rigid_sample_points
    # Values for outline -> midline fit
    # It is better to hardcode, than min(), max() as outline shape is not very stable
    # fit_tail_until_x MUST be longer than the visible tail, as it can only be cut not extended
    # The values are in reference to a the outline, centered at 0 at the pivot and rotated to 0° AOA
    fit_tail_until_x: int = -550
    fit_head_until_x: int = 150
    crop_x: tuple = (300, 1150)
    crop_y: tuple = (0, 800)
    crop_shape: tuple = (
        crop_x[1] - crop_x[0],
        crop_y[1] - crop_y[0],
    )
    pivot_point: tuple = None

    # Legacy to interact w/ stefan_post.py
    midline_save_path: Path = Path("midlines.npy")
    angle_save_path: Path = Path("angles.npy")
    angle_diff_save_path: Path = Path("angle_diffs.npy")

    # Evaluate
    evaluate_error: bool = None

    # Model
    """
    Parameters
    ----------
    naca_model : tuple
        Type of 4-digit NACA foil, individual values above 9 are valid, by default (0, 0, 18)
    chord_len : int
        Length of the chord line or midline, by default 720
    pivot_point : tuple
        Pixel location of the pivot point within the high-speed footage, by default (845, 438)
    c_pivot : float
        Location of the pivot point along the normalized midline, by default 0.25
    c_flex_rigid_interface : float
        Location of the interface between the rigid, aluminum head and the flexible body of the foil along the normalized midline, by default 0.326
    model_resolution : int
        Number of points that the NACA model is made of - must be even!, by default 500
    """
    naca_4digit_code: tuple = (0, 0, 18)
    chord_len: float = 720.0
    c_pivot: float = 0.25
    c_flex_rigid_interface: float = 0.326
    model_resolution: int = 500  # must be even
    thickening_offset_at_tail: int = 30

    # Mask
    masked_ims_save_path = Path("masked/")
    mask_ims_save_path = Path("masks/")
    undil_mask_save_path = Path("undilated_mask")
    h5_data_save_path = Path("computed_data")
