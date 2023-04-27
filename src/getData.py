# %%
import pims
import json
from dataAttrs import DataAttrs
import numpy as np


class GetData(DataAttrs):
    def init(self):
        self.film = self._get_film()
        self.frame_shape = self.get_frame(0).shape
        self.pos = self._pos_reader()
        self.cropped_pivot = (
            self.pivot_point[0] - self.crop_x[0],
            self.pivot_point[1] - self.crop_y[0],
        )

    def get_frame(self, frame_number: int) -> np.array:
        """Wraps pims methods that extracts a frame from the cine file"""
        frame = self.film.get_frame(frame_number).T
        return frame

    def _get_film(self) -> type[pims.Cine]:
        """read cinefile in specified path"""
        film = pims.open(str(self.cine_file))
        return film

    def _pos_reader(self) -> np.array:
        """Reads airfoil position from Json file, returns angle in deg"""
        with open(self.angle_file, "r") as f:
            raw = json.load(f)
        rawpos = np.array(raw[1])
        triggertime = np.argmin(raw[10])
        pos = rawpos[triggertime:]
        pos = np.interp(
            np.linspace(0, 1, len(pos) * self.angle_to_film_time_factor),
            np.linspace(0, 1, len(pos)),
            pos,
        )
        pos = pos[: self.film.image_count]
        return -pos


# %%
if __name__ == "__main__":
    from testdata import rigsin750

    rigsin750


# %%
