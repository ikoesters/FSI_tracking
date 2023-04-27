from ast import Mod
from pathlib import Path
from typing import Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import skimage.draw as draw
from tqdm import tqdm, trange

import _utils as ut
from measurement import Measurement
from model import Model
from segmenter import Segmenter


class Mask:
    def __init__(
        self,
        measurement_obj: type[Measurement],
        model_obj: type[Model],
        segmentation_obj: type[Segmenter],
    ):
        """Governing script that calls methods in the other objects to generate masks.jjjjjj

        Parameters
        ----------
        measurement_obj : type[Measurement]
            Measurement object located in measurement.py
        model_obj : type[Model]
           Model object located in model.py
        segmentation_obj : type[Segmenter]
            Segmentation object located in segmenter.py
        """

        self.meas = measurement_obj
        self.model = model_obj
        self.seg = segmentation_obj

    def main(self) -> None:
        # h5 save structure
        save_arrays_dict = {
            "midline": {
                "name": "midlines",
                "shape": (self.meas.stop_frame, self.seg.sample_points, 2),
                "dtype": "float32",
            },
            "aoa_reference": {
                "name": "aoa_reference",
                "shape": (self.meas.stop_frame),
                "dtype": "float32",
            },
            "aoa_measured": {
                "name": "aoa_measured",
                "shape": (self.meas.stop_frame),
                "dtype": "float32",
            },
            "aoa_error": {
                "name": "aoa_error",
                "shape": (self.meas.stop_frame),
                "dtype": "float32",
            },
            "dil_mask": {
                "name": "dilated_masks",
                "shape": (self.meas.stop_frame, *self.meas.frame_shape),
                "dtype": "uint8",
            },
            "undil_mask": {
                "name": "undilated_masks",
                "shape": (self.meas.stop_frame, *self.meas.frame_shape),
                "dtype": "uint8",
            },
            "masked_frame": {
                "name": "masked_frame",
                "shape": (self.meas.stop_frame, *self.meas.frame_shape),
                "dtype": "uint16",
            },
        }
        self.h5_data_save_path = ut.create_h5_file(
            save_arrays_dict, self.h5_data_save_path
        )
        self.set_all_setup_attrs()
        midlines = self.seg.compute_midlines()
        midlines = self.model.filter_and_interp_midlines(
            unfiltered_midlines=midlines, sum_nb=self.seg.sum_nb
        )

        for framenb in trange(0, self.meas.stop_frame):
            # Get frame and compute mask
            frame = self.meas.get_frame(framenb)
            midline = midlines[framenb]
            dil_mask, undil_mask = self.generate_mask(midline)
            masked_frame = self.mask_frame(frame, dil_mask)
            # Save masked frame and plot visible overlays
            ut.save_fullsized_PNG_to_folder(
                self.masked_ims_save_path, "masked", masked_frame, framenb
            )
            self.plot_visible_overlay("overlay_undilated/", frame, undil_mask, framenb)
            self.plot_visible_overlay("overlay_dilated/", frame, dil_mask, framenb)
            # Compute angle of attack error
            aoa_measured = self.model.compute_aoa_single(midline)
            aoa_reference = self.meas.pos[framenb]
            aoa_error = self.model.compute_aoa_error(aoa_measured, aoa_reference)
            # Append to h5
            variables_to_save = (
                midline,
                aoa_reference,
                aoa_measured,
                aoa_error,
                dil_mask,
                undil_mask,
                masked_frame,
            )  # must be in the same order as in save_array
            savedict = ut.create_data_entry_dict(variables_to_save, save_arrays_dict)
            ut.save_to_h5(self.h5_data_save_path, savedict, framenb)

    def compute_angles_only(self) -> None:
        save_arrays_dict = {
            "midline": {
                "name": "midlines",
                "shape": (self.meas.stop_frame, self.seg.sample_points, 2),
                "dtype": "float32",
            },
            "aoa_reference": {
                "name": "aoa_reference",
                "shape": (self.meas.stop_frame),
                "dtype": "float32",
            },
            "aoa_measured": {
                "name": "aoa_measured",
                "shape": (self.meas.stop_frame),
                "dtype": "float32",
            },
            "aoa_error": {
                "name": "aoa_error",
                "shape": (self.meas.stop_frame),
                "dtype": "float32",
            },
        }
        self.h5_data_save_path = ut.create_h5_file(
            save_arrays_dict, self.h5_data_save_path
        )
        self.set_all_setup_attrs()
        midlines = self.seg.compute_midlines()
        midlines = self.model.filter_and_interp_midlines(
            unfiltered_midlines=midlines, sum_nb=self.seg.sum_nb
        )

        aoa_measured = self.model.compute_aoa_multiple(midlines)
        aoa_reference = self.meas.pos[: self.meas.stop_frame]
        aoa_error = self.model.compute_aoa_error(aoa_measured, aoa_reference)
        variables_to_save = (
            midlines,
            aoa_reference,
            aoa_measured,
            aoa_error,
        )
        savedict = ut.create_data_entry_dict(variables_to_save, save_arrays_dict)
        ut.save_to_h5(self.h5_data_save_path, savedict)

    def set_all_setup_attrs(self) -> None:
        # Set attributes
        ut.dump_attrs(
            self.h5_data_save_path,
            ut.create_attrs_dict(self, ["meas", "model", "seg"]),
            "mask",
        )
        ut.dump_attrs(
            self.h5_data_save_path,
            ut.create_attrs_dict(self.meas, ["pos", "film"]),
            "meas",
        )
        ut.dump_attrs(
            self.h5_data_save_path, ut.create_attrs_dict(self.model, ["meas"]), "model"
        )
        ut.dump_attrs(
            self.h5_data_save_path,
            ut.create_attrs_dict(self.seg, ["meas", "model"]),
            "seg",
        )

    def mask_frame(
        self, frame: npt.NDArray[np.int_], mask: npt.NDArray[np.int_]
    ) -> np.array:
        masked_frame = frame * mask
        return masked_frame

    def plot_visible_overlay(
        self, save_folder: str, frame: np.array, mask: np.array, framenb: int
    ) -> None:
        overlay_dilated = np.where(mask == 0, 1, 0) + np.where(
            self.seg.denoise(frame) > 0, 1, 0
        )
        ut.plot_image(
            overlay_dilated,
            self.meas.intermediates_save_path / save_folder,
            "visible_overlay",
            framenb,
            100,
        )

    def mask_ims(self, masks_dil, masks_undil):
        self.undilated_masks = self.generate_masks(dilated=False)
        # unelegant solution for design choice of using the attribute self.masks as storage for later steps (side-effect)
        self.dilated_masks = self.generate_masks(dilated=True)

        print("Mask frames")
        pbar = tqdm(total=self.meas.stop_frame)  # show progress
        self.meas.act_frame = -1  # -1 for .next_frame() to start at 0
        while self.meas.act_frame < self.seg.stop_frame:
            frame = self.meas.next_frame()
            dilated_mask = self.dilated_masks[self.meas.act_frame]
            undilated_mask = self.undilated_masks[self.meas.act_frame]
            masked_frame = frame * dilated_mask
            savestr = self.masked_ims_save_path / f"masked_{self.meas.act_frame:04d}.png"
            imageio.imwrite(
                savestr,
                masked_frame.T.astype("uint16"),
                format="PNG",
            )
            if self.meas.plot_intermediates == True:
                # Save masked overlay of raw image for each mask
                overlay_undilated = np.where(undilated_mask == 0, 1, 0) + np.where(
                    self.seg.denoise(frame) > 0, 1, 0
                )
                ut.plot_image(
                    overlay_undilated,
                    self.meas.intermediates_save_path / "overlay_undilated/",
                    "overlay",
                    self.meas.act_frame,
                    100,
                )
            if self.meas.plot_intermediates == True:
                # Save masked overlay of raw image for each mask
                overlay_dilated = np.where(dilated_mask == 0, 1, 0) + np.where(
                    self.seg.denoise(frame) > 0, 1, 0
                )
                ut.plot_image(
                    overlay_dilated,
                    self.meas.intermediates_save_path / "overlay_dilated/",
                    "overlay",
                    self.meas.act_frame,
                    100,
                )
            pbar.update(1)
        pbar.close()

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


if __name__ == "__main__":
    import os

    # from post_process import PostfromPIVPNG

    root = Path(os.path.dirname(__file__))
    meas = Measurement(
        root / "Sinus_750_13000_30_1235_500_500_4_40_4000_flex.cine",
        root / "Sinus_750_13000_30_750_13000_-30_4_4_20171102-123513.dat",
        stop_frame=60,  # 60  # 16500 # must be devidable by nb_sum_frames
    )
    model = Model(meas)
    seg = Segmenter(meas, model, sum_how_many_frames=6)
    mask = Mask(
        meas,
        model,
        seg,
    )  # midlines_filepath="./midlines.npy")
    mask.compute_angles_only()

    # mask.main()
    # piv = PIVfromPNG()
