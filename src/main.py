# %%
from getData import GetData
from dataPreProc import DataPreProc
from identifyMeanline import identifyMeanline
from kinematicModel import KinematicModel
from evaluateError import EvaluateError
from bodyshape import Bodyshape
from mask import Mask
import _utils as ut
from typing import Union


class Main:
    def __init__(
        self,
        data_providing_obj: type[GetData],
    ) -> None:
        self.gd = data_providing_obj
        self._preproc = DataPreProc(self.gd)
        self._seg = identifyMeanline(self.gd)
        self._kinmod = KinematicModel(self.gd)
        self._eval = EvaluateError(self.gd)
        self._body = Bodyshape(self.gd)
        self._mask = Mask(self.gd)

    def main(self):
        self.setup()
        midlines = self.compute_all_midlines()
        # TODO: save raw midlines; dim: framenb, x, y
        midlines = self.apply_kinematic_model()
        # TODO: save midlines; dim: framenb, x, y
        if self.gd.evaluate_error is True:
            error = self.evaluate_error_rigid_case(midlines, self.gd.pos)
            # TODO: save error
        outlines = self.midline_to_outline(midlines)
        self.mask_frames(outlines)

    def setup(self):
        # TODO: setup h5
        ut.mkdir(self.gd.intermediates_save_path)
        # Legacy
        # ut.mkdir(self.dattr.masked_ims_save_path)
        # ut.mkdir(self.dattr.undil_mask_save_path)
        # ut.mkdir(self.dattr.h5_data_save_path)

    def compute_all_midlines(self):
        all_midlines = []
        for framenb in range(self.gd.stop_frame // self.gd.sum_how_many_frames):
            sumframe = self._preproc.provide_frames(
                framenb * self.gd.sum_how_many_frames,
                self.gd.sum_how_many_frames,
                self.gd.crop_x,
                self.gd.crop_y,
            )
            midline = self._seg.compute_midline(sumframe, framenb)
            all_midlines.append(midline)
            # TODO: add metada to xarray
        return all_midlines

    def apply_kinematic_model(self, midlines):
        m = self._kinmod.apply_kinematic_model(midlines)
        m = self._kinmod.low_pass_filter_in_time(midlines)
        return m

    def evaluate_error_rigid_case(self, midlines, angles):
        aoa_error = self._eval.compute_aoa_error(midlines, angles)
        return aoa_error

    def midline_to_outline(self, midlines):
        # TODO: For loop oder vectorisierung?
        pass

    def mask_frames(self, outlines):
        # TODO: For loop
        pass


# %%
if __name__ == "__main__":
    from testdata import rigsin750

    main = Main(rigsin750)
    mid = main.compute_all_midlines()

# %%
