# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm, trange

import _utils as ut
from measurement import Measurement
from model import Model


class Segmenter:
    def __init__(
        self,
        measurement_obj: type[Measurement],
        model_obj: type[Model],
        sum_how_many_frames: int = 6,
        crop_window_x: tuple = (300, 1150),
        crop_window_y: tuple = (0, 800),
        denoise_to_binarize_percentile: float = 99,
    ):
        """Takes in frames, preprocesses and segments them. Provides the midline (or meanline/chordline) for subsequent steps.

        Parameters
        ----------
        measurement_obj : type[Measurement]
            Measurement object located in measurement.py
        model_obj : type[Model]
            Model object located in model.py
        sum_how_many_frames : int, optional
            Amount of frames to be stacked to increase the seed count, by default 6
        crop_window_x : tuple, optional
            X-coordinates to crop to region where foil is located; to decrease computation effort, by default (300, 1150)
        crop_window_y : tuple, optional
            Y-coordinates to crop to region where foil is located; to decrease computation effort, by default (0, 800)
        denoise_to_binarize_percentile : float, optional
            Percentile at which a pixel brightness value is considered signal, everything below will be set to zero, by default 99
        """
        self.meas = measurement_obj
        self.model = model_obj
        self.sum_nb = sum_how_many_frames
        self.crop_x = crop_window_x
        self.crop_y = crop_window_y
        self.denoise_to_binarize_percentile = denoise_to_binarize_percentile

        self.midline_save_path = Path("midlines.npy")
        self.angle_save_path = Path("angles.npy")
        self.angle_diff_save_path = Path("angle_diffs.npy")

        self.nb_flex_sample_points = 4
        self.nb_rigid_sample_points = 3
        self.sample_points = self.nb_flex_sample_points + self.nb_rigid_sample_points
        self.crop_shape = (
            self.crop_x[1] - self.crop_x[0],
            self.crop_y[1] - self.crop_y[0],
        )
        self.first_conv_diam = 20
        self.cropped_pivot = (
            self.model.pivot_point[0] - crop_window_x[0],
            self.model.pivot_point[1] - crop_window_y[0],
        )
        # Values for outline -> midline fit
        # It is better to hardcode, than min(), max() as outline shape is not very stable
        # fit_tail_until_x MUST be longer than the visible tail, as it can only be cut not extended
        # The values are in reference to a the outline, centered at 0 at the pivot and rotated to 0° AOA
        self.fit_tail_until_x = -550
        self.fit_head_until_x = 150

    def compute_midlines(self) -> np.array:
        self.meas.act_frame = 0
        midlines = np.zeros((0, self.sample_points, 2))
        print("Compute Midlines")
        for framenb in trange(0, self.meas.stop_frame, self.sum_nb):
            midline = self.compute_midline(framenb=framenb)
            midline = np.expand_dims(midline, axis=0)
            midlines = np.vstack((midlines, midline))
        return midlines

    def rotate_points(self, points: np.array, inclination_in_arc: float) -> np.array:
        rot_mat = ut.rot_matrix(inclination_in_arc)
        rotated = np.zeros_like(points)
        for idx, (rot, pts) in enumerate(zip(rot_mat, points)):
            rotated[idx] = rot @ pts
        return rotated

    def improve_midline(self, midline: np.array, outline: np.array) -> np.array:
        arc = np.arctan(np.gradient(midline[:, 1], midline[:, 0]))
        arc_resampled = np.interp(outline[:, 0], midline[:, 0], arc)

        outline_rotated = self.rotate_points(outline, arc_resampled)
        midline = ut.fit(
            outline_rotated,
            3,
            np.linspace(outline_rotated[:, 0].min(), outline_rotated[:, 0].max(), 1000),
        )
        midline_backrotated = self.rotate_points(midline, -arc_resampled)
        return midline_backrotated

    def shift_and_rotate(self, array: np.array, angle: float, shift: float) -> np.array:
        array_shifted = array + shift
        array_rotated = array_shifted @ ut.rot_matrix(np.deg2rad(angle))
        return array_rotated

    def rotate_and_shift(self, array: np.array, angle: float, shift: float) -> np.array:
        array_rotated = array @ ut.rot_matrix(np.deg2rad(angle))
        array_shifted = array_rotated + shift

        return array_shifted

    def fit_midline(self, outline: np.array, framenb: int) -> np.array:
        pivot_point_crop = np.array((self.model.pivot_point)) - np.array(
            (self.crop_x[0], self.crop_y[0])
        )
        rotation = self.meas.pos[framenb]
        outline_rotated = self.shift_and_rotate(outline, -rotation, -pivot_point_crop)
        midline = ut.fit(
            outline_rotated,
            3,
            np.linspace(self.fit_tail_until_x, self.fit_head_until_x, 1000),
        )
        midline_backrotated = self.rotate_and_shift(midline, rotation, pivot_point_crop)
        return midline_backrotated

    def compute_midline(self, framenb: int) -> np.array:

        sumframe = self.sum_bin_crop(framenb=framenb)
        outline = self.compute_outline(sumframe, framenb=framenb)

        # Compute midline
        midline = self.fit_midline(outline, framenb)

        if self.meas.plot_intermediates == True:
            # Image output
            mid_im = self.crop(np.ones((2000, 2000)))
            mid_im[midline.astype("uint")[:, 0], midline.astype("uint")[:, 1]] = 0
            mid_im[outline.astype("uint")[:, 0], outline.astype("uint")[:, 1]] = 0
            ut.plot_image(
                mid_im,
                self.meas.intermediates_save_path / "midline/",
                "mid",
                framenb,
                100,
            )

        # Get values of flexible part of midline
        midline_len = ut.calc_len(midline)
        flex_sample_positions = np.linspace(
            0, 1 - self.model.c_flexrig, self.nb_flex_sample_points, endpoint=False
        )  # 4 points for 3rd degree fit
        flex_midline_points = np.zeros((0, 2))
        for sample_p in flex_sample_positions:
            point = midline[
                ut.find_idx_nearest(midline_len, sample_p * self.model.chord_len)
            ]
            flex_midline_points = np.vstack((flex_midline_points, point))

        # Correction of tail tip where first order midline can be assumed
        flex_grads = np.gradient(flex_midline_points, axis=0)
        flex_midline_points[0, :] += flex_grads[0, :] - flex_grads[1, :]

        # Get values of rigid part of midline

        pos = self.meas.pos[framenb]
        rigid_sample_positions = np.linspace(
            self.model.c_pivot - self.model.c_flexrig,
            self.model.c_pivot,
            self.nb_rigid_sample_points,
            endpoint=True,
        )
        rigid_midline_points = np.zeros((0, 2))
        for sample_p in rigid_sample_positions:
            point = np.array(
                (
                    np.cos(np.deg2rad(pos)) * sample_p * self.model.chord_len
                    + self.cropped_pivot[0],
                    np.sin(np.deg2rad(pos)) * sample_p * self.model.chord_len
                    + self.cropped_pivot[1],
                )
            )
            rigid_midline_points = np.vstack((rigid_midline_points, point))

        midline_points = np.vstack((flex_midline_points, rigid_midline_points))

        return midline_points

    def compute_outline(self, sumframe: np.array, framenb: int) -> np.array:

        if self.meas.plot_intermediates == True:
            # Image output
            sumf = np.where(sumframe == 0, 1, 0)
            ut.plot_image(
                sumf,
                self.meas.intermediates_save_path / "raw_frame/",
                "raw",
                framenb,
                100,
            )

        corr = ndi.correlate(sumframe, Segmenter._round_kernel(20))
        corr = np.where(corr == 0, 0, 1)

        if self.meas.plot_intermediates == True:
            # Image output
            ut.plot_image(
                np.where(corr == 1, 0, 1),
                self.meas.intermediates_save_path / "correlate/",
                "corr",
                framenb,
                100,
            )

        bin_dilate = ndi.binary_closing(corr, Segmenter._round_kernel(10), 2)
        bin_dilate = np.where(bin_dilate == 0, 1, 0)
        struct, nb_structs = Segmenter._get_biggest_structure(bin_dilate)
        # if nb_structs > 100:
        bin_dilate = ndi.binary_opening(
            bin_dilate, Segmenter._round_kernel(self.first_conv_diam), 2
        )
        struct, nb_structs = Segmenter._get_biggest_structure(bin_dilate)

        # bin_dilate = np.where(bin_dilate == 0, 1, 0)
        struct = ndi.binary_fill_holes(struct, Segmenter._round_kernel(10))

        if self.meas.plot_intermediates == True:
            # Image output
            ut.plot_image(
                np.where(struct == 0, 0, 1),
                self.meas.intermediates_save_path / "dilate/",
                "dil",
                framenb,
                100,
            )

        outline = ndi.filters.laplace(struct)  # Make 2-Pixel Outline
        # Extract Outlinevalue Indices
        outline = np.array(np.where(outline == True)).T
        return outline

    def sum_bin_crop(self, framenb: int) -> np.array:
        frame = np.zeros_like(self.crop_and_preprocess(self.meas.get_frame(0)))
        for i in range(self.sum_nb):
            next_frame = self.meas.get_frame(framenb + i)
            next_frame = self.crop_and_preprocess(next_frame)
            frame += next_frame
        return np.where(frame > 0, 1, 0)

    def crop_and_preprocess(self, frame: int) -> np.array:
        frame = self.crop(frame)
        frame = self.denoise(frame)
        frame = np.where(frame > 0, 1, 0)
        return frame

    def crop(self, frame: np.array) -> np.array:
        """Crops a frame to the window specified in the object."""
        cropped_frame = frame[
            self.crop_x[0] : self.crop_x[1], self.crop_y[0] : self.crop_y[1]
        ]
        return cropped_frame

    def split_outline(self, outline: np.array) -> np.array:
        outline = outline[outline[:, 0].argsort()]  # Sort for x
        lower_side = np.zeros((0, 2))
        upper_side = np.zeros((0, 2))
        for k in range(outline[0, 0], outline[-1, 0]):
            # grab all entries where x=i
            x_equal_k = outline[(outline[:, 0] == k)]
            mean_at_x = np.mean(x_equal_k, axis=0)

            if len(x_equal_k) == 1:  # if only one point
                continue

            lower_than_mean = x_equal_k[(x_equal_k[:, 1] < mean_at_x[1])]
            lower_than_mean = np.mean(lower_than_mean, axis=0)
            lower_side = np.vstack((lower_side, lower_than_mean))

            higher_than_mean = x_equal_k[(x_equal_k[:, 1] > mean_at_x[1])]
            higher_than_mean = np.mean(higher_than_mean, axis=0)
            upper_side = np.vstack((upper_side, higher_than_mean))
        return lower_side, upper_side

    def denoise(self, frame: np.array) -> np.array:
        """Reduce noise by filtering for values which are above a threshold,
        determined by a one-sided percentile (no upper, only lower limit).
        Remaining values are reduced by that that number, to remove offset.

        Parameters
        ----------
        frame : ndarray
            Array to denoise
        percentile : float, optional
            Percentile of values which should be above threshold, by default 99

        Returns
        -------
        denoised_frame: ndarray, uint16
            Denoised array, values below set to 0 - above: orig. value minus threshold value
        """
        threshold = np.percentile(frame, self.denoise_to_binarize_percentile)
        prepro_frame = np.where(frame < threshold, 0, frame - threshold).astype("uint16")
        return prepro_frame

    @staticmethod
    def _round_kernel(size: int) -> np.array:
        """Creates a square array with zeros with a circular shape of ones, which is as large as it can fit the array.

        Parameters
        ----------
        size : int
            Length in elements along each axis

        Returns
        -------
        out: ndarray
            Square sized array containing only zeros and ones
        """
        a, b, r = size // 2, size // 2, size // 3  # a,b = midpoint; r = radius

        y, x = np.ogrid[-a : size - a, -b : size - b]
        mask = x**2 + y**2 <= r**2
        array = np.zeros((size, size))
        array[mask] = 1
        return array

    @staticmethod
    def _get_biggest_structure(bin_frame: np.array) -> np.array:
        # Label objects in image and filter for size of airfoil
        label_objects, nb_structs = ndi.label(bin_frame)
        sizes = np.bincount(label_objects.ravel())
        sizes[0] = 0  # Deleting background
        mask_sizes = sizes == np.max(sizes)
        return mask_sizes[label_objects].astype("uint8"), nb_structs

    def reset_midlines(self) -> None:
        self.midlines = np.empty((0, self.sample_points, 2))

# %%
if __name__ == "__main__":
    import os

    from measurement import Measurement
    from model import Model

    root = os.path.dirname(__file__)
    meas = Measurement(root + "/testdata/film.cine", root + "/testdata/pos_feedback.dat")
    model = Model(meas)
    seg = Segmenter(meas, model)
    sumframe = seg.sum_bin_crop()
    outline = seg.compute_outline(sumframe)
    midline = seg.compute_midline()

    print("Ende")
