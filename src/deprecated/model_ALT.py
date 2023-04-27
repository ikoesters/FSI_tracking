import matplotlib.pyplot as plt
import numpy as np

import _utils as ut
from measurement import Measurement


class Model:
    def __init__(
        self,
        measurement_obj: type[Measurement],
        naca_model: tuple = (0, 0, 18),
        chord_len: int = 720,
        pivot_point: tuple = (845, 438),
        c_pivot: float = 0.25,
        c_flex_rigid_interface: float = 0.326,
        model_resolution: int = 500,  # must be even
        thickening_offset_at_tail: int = 30,
    ):
        """Generates outlines from a midline computed by segmenter.py based on the assumption of a 4-digit NACA foil.

        Parameters
        ----------
        measurement_obj : type[Measurement]
            Measurement object located in measurement.py
        naca_model : tuple, optional
            Type of 4-digit NACA foil, individual values above 9 are valid, by default (0, 0, 18)
        chord_len : int, optional
            Length of the chord line or midline, by default 720
        pivot_point : tuple, optional
            Pixel location of the pivot point within the high-speed footage, by default (845, 438)
        c_pivot : float, optional
            Location of the pivot point along the normalized midline, by default 0.25
        c_flex_rigid_interface : float, optional
            Location of the interface between the rigid, aluminum head and the flexible body of the foil along the normalized midline, by default 0.326
        model_resolution : int, optional
            Number of points that the NACA model is made of - must be even!, by default 500
        """
        self.meas = measurement_obj
        self.naca_model = naca_model
        self.chord_len = chord_len
        self.pivot_point = pivot_point
        self.c_pivot = c_pivot
        self.c_flexrig = c_flex_rigid_interface
        self.model_res = model_resolution
        self.thickening_offset_at_tail = thickening_offset_at_tail

    def scaled_naca(self):
        camber, naca_upper, naca_lower = self.naca4()
        camber *= self.chord_len
        naca_upper *= self.chord_len
        naca_lower *= self.chord_len
        return camber, naca_upper, naca_lower

    def naca4(self):
        m = self.naca_model[0]
        p = self.naca_model[1]
        t = self.naca_model[2]
        nb_points = self.model_res

        if np.any((m, p)) == 0:
            m, p = (
                1e-8,
                1e-2,
            )  # diff for small output on yc_first; (m / p ** 2)-part would otherwise explode
            sym_naca = True
        else:
            sym_naca = False
        m /= 100
        p /= 10
        t /= 100
        if sym_naca == True:
            c_first = np.linspace(0, 0.5, nb_points // 2, endpoint=False)
            c_last = np.linspace(0.5, 1, nb_points // 2)
        else:
            c_first = np.linspace(0, p, nb_points // 2)
            c_last = np.linspace(p, 1, nb_points // 2)
        yc_first = (m / p**2) * (2 * p * c_first - c_first**2)
        yc_last = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * c_last - c_last**2)
        yc = np.hstack((yc_first, yc_last))
        xc = np.hstack((c_first, c_last))
        camber = np.vstack((xc, yc)).T

        # Compute outline
        dy_first = (2 * m) / (p**2) * (p - c_first)
        dy_last = ((2 * m) / (1 - p) ** 2) * (p - c_last)
        dydx = np.hstack((dy_first, dy_last))
        angle = np.arctan(dydx)
        yt = (
            5
            * t
            * (
                0.2969 * np.sqrt(xc)
                - 0.1260 * xc
                - 0.3516 * xc**2
                + 0.2843 * xc**3
                - 0.1015 * xc**4
            )
        )
        upper = np.array((xc - yt * np.sin(angle), yc + yt * np.cos(angle))).T
        lower = np.array((xc + yt * np.sin(angle), yc - yt * np.cos(angle))).T

        return camber, upper, lower

    def _filter_midlines(
        self, midlines: np.array, order: int = 2, fs: int = 1000, cutoff: int = 100
    ) -> np.array:
        # midlines = np.copy(midlines)
        for mid_point in range(midlines.shape[1]):
            for coo in range(midlines.shape[2]):
                midlines[:, mid_point, coo] = ut.butter_lowpass_filter(
                    midlines[:, mid_point, coo], cutoff, fs, order
                )
        return midlines

    def _interpolate_midlines_timewise(self, midlines: np.array, factor: int) -> np.array:
        interp_midlines = np.zeros(
            (midlines.shape[0] * factor, midlines.shape[1], midlines.shape[2])
        )
        for mid_point in range(midlines.shape[1]):
            for coo in range(midlines.shape[2]):
                interp_midlines[:, mid_point, coo] = np.interp(
                    np.linspace(
                        0,
                        1,
                        midlines.shape[0] * factor,
                    ),
                    np.linspace(
                        0,
                        1,
                        midlines.shape[0],
                    ),
                    midlines[:, mid_point, coo],
                )
        return interp_midlines

    def interpolate_midline_lengthwise(
        self, midline_points: np.array, degree: int = 4
    ) -> np.array:
        # Generate midline from points
        midline = ut.fit(
            midline_points,
            degree,
            np.linspace(midline_points[0, 0], midline_points[-1, 0], self.model_res),
        )
        return midline

    def filter_and_interp_midlines(
        self, unfiltered_midlines: np.array, sum_nb: int
    ) -> np.array:
        midlines = self._filter_midlines(unfiltered_midlines)
        midlines = self._interpolate_midlines_timewise(midlines, factor=sum_nb)
        return midlines

    def _make_naca(self, midline_points: np.array) -> tuple:
        # Generate Naca foil
        midline = self.interpolate_midline_lengthwise(midline_points)

        # Generate Naca foil
        camber, naca_upper, naca_lower = self.scaled_naca()
        # dreht y-Werte -> das Profil um 180°
        naca_upper[:, 1] = np.flip(naca_upper[:, 1])
        naca_lower[:, 1] = np.flip(naca_lower[:, 1])
        camber[:, 1] = np.flip(camber[:, 1])

        naca_upper -= camber
        naca_lower -= camber

        dydx = np.gradient(midline[:, 1], midline[:, 0])
        angle = np.arctan(dydx)

        # Put foil outline around midline
        def outline_upper(correction_value: int):
            return np.array(
                (
                    midline[:, 0] - (naca_upper[:, 1] + correction_value) * np.sin(angle),
                    midline[:, 1] + (naca_upper[:, 1] + correction_value) * np.cos(angle),
                )
            ).T

        def outline_lower(correction_value: int):
            return np.array(
                (
                    midline[:, 0]
                    + (-naca_lower[:, 1] + correction_value) * np.sin(angle),
                    midline[:, 1]
                    - (-naca_lower[:, 1] + correction_value) * np.cos(angle),
                )
            ).T

        undil_corr_val = np.linspace(0, 0, self.model_res)
        upper_undil = outline_upper(undil_corr_val)
        lower_undil = outline_lower(undil_corr_val)

        dil_corr_val = np.linspace(
            self.thickening_offset_at_tail,
            self.thickening_offset_at_tail / 6,
            self.model_res,
        )
        upper_dil = outline_upper(dil_corr_val)
        lower_dil = outline_lower(dil_corr_val)

        return (upper_dil, lower_dil), (upper_undil, lower_undil)

    def compute_aoa_multiple(self, midlines: np.array) -> np.array:
        # From head to tail
        angles = np.rad2deg(
            np.arctan((midlines[:,-1, 1] - midlines[:,0, 1]) / (midlines[:,-1, 0] - midlines[:,0, 1]))
        )
        return angles
    
    def compute_aoa_single(self, midline: np.array) -> np.array:
        # From head to tail
        angle = np.rad2deg(
            np.arctan((midline[-1, 1] - midline[0, 1]) / (midline[-1, 0] - midline[0, 1]))
        )
        return angle

    def compute_abs_aoa_error(
        self, measured_angle: np.array, reference_angle: np.array
    ) -> np.array:
        return np.absolute(measured_angle - reference_angle)

    def compute_aoa_error(
        self, measured_angle: np.array, reference_angle: np.array
    ) -> np.array:
        return measured_angle - reference_angle


if __name__ == "__main__":
    import os
    from measurement import Measurement

    root = os.path.dirname(__file__)
    meas = Measurement(root + "/testdata/film.cine", root + "/testdata/pos_feedback.dat")
    model = Model(meas)
    print("Ende")
