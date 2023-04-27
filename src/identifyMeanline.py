# %%
import numpy as np
from scipy import ndimage as ndi

import _utils as ut
from getData import GetData
from plotIntermediates import PlotIntermediates


class identifyMeanline:
    def __init__(
        self,
        data_providing_obj: type[GetData],
    ):
        """Takes in frames, preprocesses and segments them.
        Provides the midline (or meanline/chordline) for subsequent steps.
        """
        self.gd = data_providing_obj
        self.pli = PlotIntermediates(self.gd)

    def fit_meanline(self, outline: np.array, framenb: int) -> np.array:
        pivot_point_crop = np.array((self.gd.pivot_point)) - np.array(
            (self.gd.crop_x[0], self.gd.crop_y[0])
        )  # FIXME: Pivot point shift nur ein wert oder ein Paar?
        rotation = self.gd.pos[framenb]
        outline_rotated = self.shift_and_rotate(outline, -rotation, -pivot_point_crop)
        midline = ut.fit(
            outline_rotated,
            3,
            np.linspace(self.gd.fit_tail_until_x, self.gd.fit_head_until_x, 1000),
        )
        midline_backrotated = self.rotate_and_shift(midline, rotation, pivot_point_crop)
        return midline_backrotated

    def shift_and_rotate(self, array: np.array, angle: float, shift: float) -> np.array:
        array_shifted = array + shift
        array_rotated = array_shifted @ ut.rot_matrix(np.deg2rad(angle))
        return array_rotated

    def rotate_and_shift(self, array: np.array, angle: float, shift: float) -> np.array:
        array_rotated = array @ ut.rot_matrix(np.deg2rad(angle))
        array_shifted = array_rotated + shift

        return array_shifted

    def compute_midline(self, sumframe: np.array, framenb: int) -> np.array:
        outline = self.compute_outline(sumframe, framenb=framenb)
        midline = self.fit_meanline(outline, framenb)
        self.pli.plot_meanline(midline, outline, framenb)
        flex_midline_points = self.get_flex_points(midline)
        flex_midline_points = self.correct_tailtip(flex_midline_points)
        rigid_midline_points = self.get_rigid_points(framenb)
        midline_points = np.vstack((flex_midline_points, rigid_midline_points))
        # FIXME: kann framenb von data_attr kommen? letzer wert, nicht ungenau?
        return midline_points

    def get_flex_points(self, midline: np.array) -> np.array:
        # Get values of flexible part of midline
        midline_len = ut.calc_len(midline)
        flex_sample_positions = np.linspace(
            0,
            1 - self.gd.c_flex_rigid_interface,
            self.gd.nb_flex_sample_points,
            endpoint=False,
        )
        flex_midline_points = np.zeros((0, 2))
        for sample_p in flex_sample_positions:
            point = midline[
                ut.find_idx_nearest(midline_len, sample_p * self.gd.chord_len)
            ]
            flex_midline_points = np.vstack((flex_midline_points, point))
        return flex_midline_points

    def correct_tailtip(self, flex_midline_points: np.array) -> np.array:
        # Correction of tail tip where first order midline can be assumed
        flex_grads = np.gradient(flex_midline_points, axis=0)
        flex_midline_points[0, :] += flex_grads[0, :] - flex_grads[1, :]
        return flex_midline_points

    def get_rigid_points(self, framenb: int) -> np.array:
        # Get values of rigid part of midline
        pos = self.gd.pos[framenb]
        rigid_sample_positions = np.linspace(
            self.gd.c_pivot - self.gd.c_flex_rigid_interface,
            self.gd.c_pivot,
            self.gd.nb_rigid_sample_points,
            endpoint=True,
        )
        rigid_midline_points = np.zeros((0, 2))
        for sample_p in rigid_sample_positions:
            point = np.array(
                (
                    np.cos(np.deg2rad(pos)) * sample_p * self.gd.chord_len
                    + self.gd.cropped_pivot[0],
                    np.sin(np.deg2rad(pos)) * sample_p * self.gd.chord_len
                    + self.gd.cropped_pivot[1],
                )
            )
            rigid_midline_points = np.vstack((rigid_midline_points, point))
        return rigid_midline_points

    def compute_outline(self, sumframe: np.array, framenb: int) -> np.array:
        self.pli.plot_raw_frame(sumframe, framenb)
        corr = self.correlate_frame(sumframe)
        self.pli.plot_correlate(corr, framenb)
        bin_dilate = self.dilate_frame(corr)
        struct = self.extract_shape(bin_dilate)
        self.pli.plot_dilate(struct, framenb)
        outline_idc = self.extract_outline_idc(struct)
        return outline_idc

    def correlate_frame(self, frame: np.array) -> np.array:
        corr = ndi.correlate(frame, self._round_kernel(self.gd.conv_diam))
        corr = np.where(corr == 0, 0, 1)
        return corr

    def dilate_frame(self, frame: np.array) -> np.array:
        bin_dilate = ndi.binary_closing(
            frame, self._round_kernel(self.gd.dilation_diam), 3
        )
        bin_dilate = np.where(bin_dilate == 0, 1, 0)
        return bin_dilate

    def extract_shape(self, frame: np.array) -> np.array:
        struct, _ = self._get_biggest_structure(frame)
        bin_dilate = ndi.binary_opening(struct, self._round_kernel(self.gd.conv_diam), 2)
        struct, _ = self._get_biggest_structure(bin_dilate)

        struct = ndi.binary_fill_holes(struct, self._round_kernel(self.gd.dilation_diam))
        return struct

    def extract_outline_idc(self, frame: np.array) -> np.array:
        outline = ndi.filters.laplace(frame)
        outline = np.array(np.where(outline == True)).T
        return outline

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
        self.midlines = np.empty((0, self.gd.sample_points, 2))


# %%
if __name__ == "__main__":
    pass
