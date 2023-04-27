import numpy as np

import _utils as ut
from getData import GetData


class KinematicModel:
    def __init__(self, data_providing_obj: type[GetData]):
        """ """
        self.gd = data_providing_obj

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
            np.linspace(
                midline_points[0, 0], midline_points[-1, 0], self.gd.model_resolution
            ),
        )
        return midline

    def filter_and_interp_midlines(
        self, unfiltered_midlines: np.array, sum_nb: int
    ) -> np.array:
        midlines = self._filter_midlines(unfiltered_midlines)
        midlines = self._interpolate_midlines_timewise(midlines, factor=sum_nb)
        return midlines
