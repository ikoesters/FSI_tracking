import pathlib

import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import seaborn as sns
import skimage.draw as draw
from tqdm import tqdm

import _utils as ut


class Evaluate_SynthData:
    def __init__(
        self,
        measurement_obj,
        model_obj,
        segmentation_obj,
        mask_obj,
        evaluations_save_path="evaluations/",
        seed_density_sweep=np.linspace(250, 500, 11, endpoint=True),
        segmentation_conv_diam_sweep=np.arange(10, 31, 10),
    ):
        self.meas = measurement_obj
        self.model = model_obj
        self.seg = segmentation_obj
        self.mask = mask_obj
        self.masked_ims_save_path = pathlib.Path.cwd() / evaluations_save_path
        self.seed_density_sweep = seed_density_sweep
        self.segmentation_conv_diam_sweep = segmentation_conv_diam_sweep

        # self.seg.first_conv_diam = self.segmentation_conv_diam

    def seed_density(self, seed_thresh: int) -> int:
        # evaluate seed density BEFORE illusiory contour is applied
        # gets called by correspondece fcts for indice names
        self.meas.seed_threshold = seed_thresh
        im = self.meas.naive_background_gen()
        _, sizes = meas.count_structures(im)
        nb_sizes = sizes.shape[0]
        sizes_h_p_h = nb_sizes / (
            (self.meas.frame_shape[0] * self.meas.frame_shape[1] / (100 * 100))
        )  # seeds per 100x100px
        return int(sizes_h_p_h)

    def angle_correspondence(self) -> None:
        synth_midline, _, _ = self.meas.naca4()
        synth_angle = np.arctan(
            (synth_midline[-1, 1] - synth_midline[0, 1])
            / (synth_midline[-1, 0] - synth_midline[0, 1])
        )
        synth_angle = np.expand_dims(synth_angle, axis=0)
        synth_angles = np.repeat(synth_angle, self.meas.stop_frame - 1, axis=0)

        results = np.empty((0, 3))
        keys = ("Seed", "Diameter", "Angle_Difference")
        np.save("angle_keys", keys)

        for ind_seed, seed in enumerate(self.seed_density_sweep):
            thresh = self.meas.find_seed_density(seed)
            self.meas.seed_threshold = thresh
            self.meas.generate_synth_png()
            seed_dens = seed
            for ind_diam, diam in enumerate(self.segmentation_conv_diam_sweep):
                print(
                    f"Seed: {seed}\nConvolve Diameter: {diam}\nRun {len(self.segmentation_conv_diam_sweep)*ind_seed + ind_diam +1} of {len(self.segmentation_conv_diam_sweep)* len(self.seed_density_sweep)}\n"
                )
                # make masks
                self.seg.first_conv_diam = diam  # diameter of summation window for seeds
                self.seg.compute_midlines()
                self.model.filter_and_interp_midlines(self.seg)
                angles = np.rad2deg(
                    np.arctan(
                        (self.model.midlines[:, -1, 1] - self.model.midlines[:, 0, 1])
                        / (self.model.midlines[:, -1, 0] - self.model.midlines[:, 0, 1])
                    )
                )
                # Use previous masks - for testing
                # self.mask.masked_ims_save_path = pathlib.Path.cwd() / "bin_masks.npy"
                # self.mask.masks = np.load(self.mask.masked_ims_save_path)

                diff_angles = np.absolute(synth_angles - angles)
                result = np.array(
                    (
                        np.repeat(int(seed_dens), self.meas.stop_frame - 1)[10:],
                        np.repeat(int(diam), self.meas.stop_frame - 1)[10:],
                        diff_angles[10:],
                    )
                ).T
                results = np.vstack((results, result))
                np.save("angle_diffs", results)

                self.seg.reset_midlines()

        return results, keys

    def angle_correspondence_wo_diam_sweep(self) -> tuple:
        synth_midline, _, _ = self.meas.naca4()
        synth_angle = np.arctan(
            (synth_midline[-1, 1] - synth_midline[0, 1])
            / (synth_midline[-1, 0] - synth_midline[0, 1])
        )
        synth_angle = np.expand_dims(synth_angle, axis=0)
        synth_angles = np.repeat(synth_angle, self.meas.stop_frame - 1, axis=0)

        results = np.empty((0, 3))
        keys = ("Seed", "Diameter", "Angle_Difference")
        np.save("angle_keys", keys)

        for ind_seed, seed in enumerate(self.seed_density_sweep):
            thresh = self.meas.find_seed_density(seed)
            self.meas.seed_threshold = thresh
            seed_dens = seed
            diam = self.seg.first_conv_diam
            print(
                f"Seed: {seed}\nConvolve Diameter: {diam}\nRun {ind_seed+1} of {len(self.seed_density_sweep)}\n"
            )
            self.meas.generate_synth_png()
            self.seg.compute_midlines()
            self.model.filter_and_interp_midlines(self.seg)
            angles = np.rad2deg(
                np.arctan(
                    (self.model.midlines[:, -1, 1] - self.model.midlines[:, 0, 1])
                    / (self.model.midlines[:, -1, 0] - self.model.midlines[:, 0, 1])
                )
            )

            diff_angles = np.absolute(synth_angles - angles)
            result = np.array(
                (
                    np.repeat(int(seed_dens), self.meas.stop_frame - 1)[10:],
                    np.repeat(int(diam), self.meas.stop_frame - 1)[10:],
                    diff_angles[10:],
                )
            ).T
            results = np.vstack((results, result))
            np.save("angle_diffs", results)

            self.seg.reset_midlines()

        return results, keys


if __name__ == "__main__":
    from mask import Mask
    from measurement import Measurement
    from model import Model
    from segmenter import Segmenter
    from synthetic_images import SyntheticImages

    naca_model = (15, 5, 18)
    meas = SyntheticImages(naca_model=naca_model, seed_dens_in_wake=300)
    # meas.generate_synth_png()
    meas.make_filelist()
    pivot = meas.find_pivot()
    model = Model(
        meas,
        pivot_point=(pivot[0], pivot[1]),
        naca_model=naca_model,
        thickening_offset_at_tail=0,
    )
    seg = Segmenter(meas, model, sum_how_many_frames=1)
    mask = Mask(
        meas,
        model,
        seg,
    )
    meas.make_filelist()
    eval = Evaluate_SynthData(meas, model, seg, mask)
    result, keys = eval.angle_correspondence_wo_diam_sweep()
