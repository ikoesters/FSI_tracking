# %%
from getData import GetData

import numpy as np


class DataPreProc:
    def __init__(self, data_providing_obj: type[GetData]):
        """ """
        self.gd = data_providing_obj

    def provide_frames(
        self,
        start_frame: int,
        sum_how_many_frames: int,
        crop_window_x: tuple,
        crop_window_y: tuple,
    ) -> np.array:
        sum_frame = self._sum_frames(start_frame, sum_how_many_frames)
        sum_frame = self._preprocess_frame(sum_frame)
        sum_frame = self._crop(sum_frame, crop_window_x, crop_window_y)
        return sum_frame

    def _sum_frames(self, start_frame: int, how_many_frames: int) -> np.array:
        summed_frame = np.zeros((self.gd.frame_shape))
        for framenb in range(how_many_frames):
            frame = self._normalize_frame(self.gd.get_frame(start_frame + framenb))
            summed_frame += frame
        return summed_frame / how_many_frames

    @staticmethod
    def _normalize_frame(frame: np.array) -> np.array:
        return frame / 65_536  # 2**16

    def _preprocess_frame(self, frame: np.array) -> np.array:
        frame = self._denoise(frame)
        frame = np.where(frame > 0, 1, 0)
        return frame

    def _denoise(self, frame: np.array) -> np.array:
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
        threshold = np.percentile(frame, self.gd.denoise_to_binarize_percentile)
        prepro_frame = np.where(frame < threshold, 0, frame - threshold)
        return prepro_frame

    @staticmethod
    def _crop(frame, crop_window_x, crop_window_y):
        cropped_frame = frame[
            crop_window_x[0] : crop_window_x[1], crop_window_y[0] : crop_window_y[1]
        ]
        return cropped_frame


# %%
if __name__ == "__main__":
    from testdata import flexsin750

    getdata = DataPreProc(flexsin750)
    print()

# %%
