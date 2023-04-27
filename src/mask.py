# %%
import imageio
import numpy as np
import numpy.typing as npt
import skimage.draw as draw

import _utils as ut
from getData import GetData


class Mask:
    def __init__(self, data_providing_obj: type[GetData]):
        """Governing script that calls methods in the
        other objects to generate masks.
        """
        self.gd = data_providing_obj

    def _indices_to_maskarray(
        self, naca_upper: np.array, naca_lower: np.array
    ) -> np.array:
        im = np.ones(self.seg.crop_shape)
        perimiter = np.vstack((naca_lower, np.flipud(naca_upper)))
        rr, cc = draw.polygon(perimiter[:, 0], perimiter[:, 1])
        im[rr, cc] = 0
        return im

    def outline_to_mask(self, naca_upper: np.array, naca_lower: np.array) -> np.array:
        mask = self._indices_to_maskarray(naca_upper, naca_lower)
        mask = self.pad_ones(mask)
        return mask

    def generate_mask(self, midline: npt.NDArray) -> tuple:
        (dil_naca_upper, dil_naca_lower), (
            undil_naca_upper,
            undil_naca_lower,
        ) = self.model._make_naca(midline)
        dil_mask = self.outline_to_mask(dil_naca_upper, dil_naca_lower)
        undil_mask = self.outline_to_mask(undil_naca_upper, undil_naca_lower)
        return dil_mask, undil_mask

    def generate_masks(self, filtered_midlines):
        masks = np.expand_dims(np.zeros(self.meas.frame_shape), axis=0)
        midlines = self.seg.compute_midlines()
        midlines = self.model.filter_and_interp_midlines(
            unfiltered_midlines=midlines, sum_nb=self.seg.sum_nb
        )

        print("Generate Masks")
        for framenb in range(0, self.meas.stop_frame):
            midline_points = midlines[framenb]
            (dil_naca_upper, dil_naca_lower), (
                undil_naca_upper,
                undil_naca_lower,
            ) = self.model._make_naca(midline_points)
            dil_mask = self.generate_mask(dil_naca_upper, dil_naca_lower)
            undil_mask = self.generate_mask(undil_naca_upper, undil_naca_lower)
            savestr = self.undil_mask_save_path / f"mask_{self.meas.act_frame:04d}.png"
            imageio.imwrite(
                savestr,
                mask.T.astype("uint8"),
                format="PNG",
            )
            mask = np.expand_dims(mask, axis=0)
            masks = np.vstack((masks, mask))
            self.meas.act_frame += 1
        # np.save(self.undil_mask_save_path.stem + ".npy", masks)
        return masks

    def pad_ones(self, cropped_mask: np.array) -> np.array:
        padded_mask = np.ones(self.meas.frame_shape)
        padded_mask[
            self.seg.crop_x[0] : self.seg.crop_x[1],
            self.seg.crop_y[0] : self.seg.crop_y[1],
        ] = cropped_mask
        return padded_mask

    def reset_masks(self) -> None:
        self.masks = np.empty(self.meas.frame_shape)
        self.masks = np.expand_dims(self.masks, axis=0)


# %%
if __name__ == "__main__":
    from testdata import rigsin750
