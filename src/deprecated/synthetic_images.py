import pathlib

import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import skimage.draw as draw
import tqdm
from scipy import interpolate

import _utils as ut


class SyntheticImages:
    def __init__(
        self,
        save_path_raw_frames="synth_ims/",
        intermediates_save_path="intermediates/",
        path_angles=None,
        angle_to_film_time_resolution=1,
        filetype="png",
        stop_frame=100,
        frame_shape=(1280, 800),
        camber_start_point=(350, 450),
        chord_len=720,
        naca_model=(0, 0, 18),
        model_resolution=500,  # should be the same as in model
        c_pivot=0.25,
        seed_threshold=0.95,
        seed_dens_in_wake=None,  # e.g. 200
    ):
        self.save_path_raw_frames = (
            pathlib.Path.cwd() / save_path_raw_frames / f"seed_{seed_threshold}/"
        )
        self.intermediates_save_path = pathlib.Path.cwd() / intermediates_save_path
        self.path_angles = path_angles
        self.filetype = filetype
        self.angle_to_film_time_resolution = angle_to_film_time_resolution
        self.stop_frame = stop_frame
        self.frame_shape = frame_shape
        self.camber_start_point = camber_start_point
        self.chord_len = chord_len
        self.naca_model = naca_model
        self.model_res = model_resolution
        self.c_pivot = c_pivot
        self.seed_threshold = seed_threshold

        self.act_frame = 0
        self.frame_shape = frame_shape
        self.pos = self.pos_reader()
        self.scale_fct = self.find_scale_fct()
        self.c_pivot = c_pivot
        self.pivot = self.find_pivot()
        ut.make_folder(self.save_path_raw_frames)
        ut.make_folder(self.intermediates_save_path)
        self.plot_intermediates = (
            True if self.intermediates_save_path is not None else False
        )
        self.wake_case_seed_thresh = None  # needed for seed_density_correlation
        self.seed_dens_corr = self.seed_density_correlation()
        if seed_dens_in_wake is not None:
            self.wake_case_seed_thresh = self.find_seed_density(seed_dens_in_wake)

    def next_frame(self):
        self.act_frame += 1
        frame = self.get_frame(self.act_frame)
        return frame

    def generate_synth_png(self):
        print("Generate Synthetic Images")
        for im in tqdm.trange(0, self.stop_frame):
            image = self.naive_background_gen()
            mask = self.naca_ill_contour()
            masked_image = image * mask
            scale_fct = int(2 ** 16) // np.max(masked_image)
            masked_image *= scale_fct
            imageio.imwrite(
                self.save_path_raw_frames / f"synth_im_{im:04d}.png",
                masked_image.astype("uint16").T,
            )
            self.make_filelist()

    def naca_ill_contour(self):
        _, naca_upper, naca_lower = self.naca4()
        naca_upper = self.shift_and_scale(naca_upper)
        naca_lower = self.shift_and_scale(naca_lower)

        naca_upper[:, 1] = np.flipud(naca_upper[:, 1])
        naca_lower[:, 1] = np.flipud(naca_lower[:, 1])
        mask = np.ones(self.frame_shape)
        perimiter = np.vstack((naca_lower, np.flipud(naca_upper)))
        rr, cc = draw.polygon(perimiter[:, 0], perimiter[:, 1])
        mask[rr, cc] = 0
        return mask

    def shift_and_scale(self, naca):
        naca *= self.scale_fct * self.chord_len
        naca[:, 0] += self.camber_start_point[0]
        naca[:, 1] += self.camber_start_point[1]
        return naca

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
        yc_first = (m / p ** 2) * (2 * p * c_first - c_first ** 2)
        yc_last = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * c_last - c_last ** 2)
        yc = np.hstack((yc_first, yc_last))
        xc = np.hstack((c_first, c_last))
        camber = np.vstack((xc, yc)).T

        # Compute outline
        dy_first = (2 * m) / (p ** 2) * (p - c_first)
        dy_last = ((2 * m) / (1 - p) ** 2) * (p - c_last)
        dydx = np.hstack((dy_first, dy_last))
        angle = np.arctan(dydx)
        yt = (
            5
            * t
            * (
                0.2969 * np.sqrt(xc)
                - 0.1260 * xc
                - 0.3516 * xc ** 2
                + 0.2843 * xc ** 3
                - 0.1015 * xc ** 4
            )
        )
        upper = np.array((xc - yt * np.sin(angle), yc + yt * np.cos(angle))).T
        lower = np.array((xc + yt * np.sin(angle), yc - yt * np.cos(angle))).T

        return camber, upper, lower

    def get_frame(self, frame_nb):
        frame_path = self.file_list[frame_nb]
        frame = imageio.imread(frame_path).T
        return frame

    def naive_background_gen(self, threshold=None):
        if threshold is None:
            threshold = self.seed_threshold
        if self.wake_case_seed_thresh is None:
            image = np.random.rand(self.frame_shape[0], self.frame_shape[1])
            image = np.where(image > threshold, 1, 0)  # generate sparse seed
        else:
            image_nowake = np.random.rand(self.frame_shape[0], int(self.pivot[1]) - 25)
            image_nowake = np.where(
                image_nowake > threshold, 1, 0
            )  # generate sparse seed
            image_wake = np.random.rand(
                self.frame_shape[0], self.frame_shape[1] - int(self.pivot[1] - 25)
            )
            image_wake = np.where(
                image_wake > self.wake_case_seed_thresh, 1, 0
            )  # generate sparse seed
            image = np.hstack((image_nowake, image_wake))

        dilate_seed = ndi.correlate(image, np.ones((2, 2)))
        dilate_seed = np.where(
            dilate_seed < 2, 0, dilate_seed
        )  # only retain cores of the seeds
        return dilate_seed

    def seed_density_correlation(self, low=0.8, high=0.99):
        density = np.empty((0, 1))

        vals = np.linspace(low, high, 20)

        for val in vals:
            im = self.naive_background_gen(val)
            _, st = self.count_structures(im)
            nb_str = st[1:].shape[0]
            dens = nb_str / (1080 * 800 / 10_000)
            density = np.vstack((density, dens))

        interp_obj = interpolate.interp1d(vals, density[:, 0], kind="quadratic")
        x_new = np.linspace(low, high, 1000)
        return np.array((x_new, interp_obj(x_new))).T

    def find_seed_density(self, seed_dens):
        nearest = ut.find_idx_nearest(self.seed_dens_corr[:, 1], seed_dens)
        thresh = self.seed_dens_corr[nearest, 0]

        return thresh

    def make_filelist(self):
        self.file_list = sorted(
            pathlib.Path(self.save_path_raw_frames).glob(f"**/*.{self.filetype}")
        )

    def pos_reader2(self):
        pivot = self.find_pivot()
        angle = np.arctan(
            (pivot[1] - self.camber_start_point[1])
            / (pivot[0] - self.camber_start_point[0])
        )
        pos = np.repeat(
            np.rad2deg(angle),
            self.stop_frame,
        )
        return pos

    def pos_reader(self):
        c, _, _ = self.naca4()
        idx_pivot = int(self.c_pivot * self.model_res)
        angle = np.arctan((c[idx_pivot, 1] - c[0, 1]) / (c[idx_pivot, 0] - c[0, 0]))
        pos = np.repeat(
            np.rad2deg(angle),
            self.stop_frame,
        )

        return pos

    def find_pivot(self):
        camber, _, _ = self.naca4()
        camber = self.shift_and_scale(camber)
        pivot = camber[int((1 - self.c_pivot) * self.model_res)]
        return pivot

    def find_scale_fct(self):
        camber, _, _ = self.naca4()
        len_camber = ut.calc_len(camber)[-1]
        fct = 1 / len_camber
        return fct

    @staticmethod
    def count_structures(bin_frame):
        # Label objects in image and filter for size of airfoil
        label_objects, _ = ndi.label(bin_frame)
        sizes = np.bincount(label_objects.ravel())
        sizes[0] = 0  # Deleting background
        # mask_sizes = sizes == np.max(sizes)
        return label_objects, sizes


if __name__ == "__main__":
    from pathlib import Path

    from mask import Mask
    from model import Model
    from segmenter import Segmenter

    naca_model = (15, 5, 18)
    seed_dens_in_wake = 250
    meas = SyntheticImages(naca_model=naca_model, seed_dens_in_wake=seed_dens_in_wake)
    ut.make_folder("plots")
    for seed in range(200, 501, 50):
        thresh = meas.find_seed_density(seed)
        image = meas.naive_background_gen(thresh)
        mask = meas.naca_ill_contour()
        masked_image = image * mask
        masked_image = (np.where(masked_image > 0, 1, 0) * 255).astype("uint8")
        imageio.imwrite(f"plots/wake{seed_dens_in_wake}_seed{seed}.png", masked_image.T)

    print("Ende")


"""
    naca_model = (0, 0, 18)
    meas = SyntheticImages(naca_model=naca_model)
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
    mask.generate_masks()
    mask.mask_ims()
    print("Ende")
"""
