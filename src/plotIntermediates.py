# %%
import _utils as ut
import numpy as np
from getData import GetData


class PlotIntermediates:
    def __init__(self, data_providing_obj: type[GetData]) -> None:
        self.gd = data_providing_obj

    def plot_meanline(self, meanline, outline, framenb):
        if self.gd.plot_intermediates:
            canvas = np.ones(self.gd.frame_shape)
            ix = np.argwhere((meanline[:, 0] < canvas.shape[0]) & (meanline[:, 0] > 0))
            iy = np.argwhere((meanline[:, 1] < canvas.shape[1]) & (meanline[:, 1] > 0))
            indice = np.intersect1d(ix, iy)
            meanline = meanline[indice]

            canvas[meanline.astype("uint")[:, 0], meanline.astype("uint")[:, 1]] = 0
            canvas[outline.astype("uint")[:, 0], outline.astype("uint")[:, 1]] = 0
            ut.plot_image(
                canvas,
                self.gd.intermediates_save_path / "meanline/",
                "mid",
                framenb,
                100,
            )

    def plot_raw_frame(self, frame, framenb):
        if self.gd.plot_intermediates:
            sumf = np.where(frame == 0, 1, 0)
            ut.plot_image(
                sumf,
                self.gd.intermediates_save_path / "raw_frame/",
                "raw",
                framenb,
                100,
            )

    def plot_correlate(self, corr_frame, framenb):
        if self.gd.plot_intermediates:
            ut.plot_image(
                np.where(corr_frame == 1, 0, 1),
                self.gd.intermediates_save_path / "correlate/",
                "corr",
                framenb,
                100,
            )

    def plot_dilate(self, frame_struct, framenb):
        if self.gd.plot_intermediates:
            ut.plot_image(
                np.where(frame_struct == 0, 0, 1),
                self.gd.intermediates_save_path / "dilate/",
                "dil",
                framenb,
                100,
            )
