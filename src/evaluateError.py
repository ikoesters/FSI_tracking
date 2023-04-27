# %%
from getData import GetData
import numpy as np


class EvaluateError:
    def __init__(self, data_providing_obj: type[GetData]):
        self.gd = data_providing_obj

    def compute_aoa(self, midlines: np.array) -> np.array:
        if len(midlines) == 2:
            midlines = np.expand_dims(midlines, axis=0)
        # From head to tail
        angles = np.rad2deg(
            np.arctan(
                (midlines[:, -1, 1] - midlines[:, 0, 1])
                / (midlines[:, -1, 0] - midlines[:, 0, 1])
            )
        )
        return angles

    def compute_aoa_error(
        self, midlines: np.array, reference_angle: np.array
    ) -> np.array:
        measured_angle = self.compute_aoa(midlines)
        return measured_angle - reference_angle
