import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.interpolate import griddata
import scipy as sp
import skimage
import matplotlib.image as img
from scipy import ndimage
import cv2
from imageio import imwrite
from pathlib import Path


def get_data(path, path2, start, end, two_d=None, replace_nan=None):
    vec_x = []
    vec_y = []
    paths = sorted(glob(path + 'Masked/' + path2 + 'piv_*.h5'))
    for path in paths:
        with h5py.File(path) as f:
            vec_x.append(np.array(f['piv2']['deltaxs_final']))
            vec_y.append(np.array(f['piv2']['deltays_final']))
    vec_x = np.array(vec_x)
    vec_y = np.array(vec_y)
    if two_d == True:
        path_to_dim = paths[0][:-14]
        vec_x = np.array(post.build_2d_array(vec_x, path_to_dim))
        vec_y = np.array(post.build_2d_array(vec_y, path_to_dim))
    return vec_x, vec_y


def build_stack_and_compute(path, path2, count_max, start, end, two_d=None,
                            replace_nan=None):
    '''func to stack and average the results with a sliding a window of 
    size count_max in order to fill empty spaces in the vector field'''
    paths = sorted(glob(path + path2 + 'piv_*.h5'))
    count = 1
    frame = 1
    vec_x_med = []
    vec_y_med = []
    corr_med = []
    with h5py.File(paths[0]) as f:
        correl = np.array(f['piv2']['correls_max'])
        vec_x = np.array(f['piv2']['deltaxs_final'])
        vec_y = np.array(f['piv2']['deltays_final'])
        ixvecs_final = np.array(f['piv2']['ixvecs_final'][:])
        iyvecs_final = np.array(f['piv2']['iyvecs_final'][:])

    # stack the frames
    while frame < len(paths[start:end]):
        while count < count_max:
            with h5py.File(paths[start + frame]) as f:
                frame += 1
                count += 1
                correl = np.vstack(
                    (correl, np.array(f['piv2']['correls_max'])))
                vec_x = np.vstack(
                    (vec_x, np.array(f['piv2']['deltaxs_final'])))
                vec_y = np.vstack(
                    (vec_y, np.array(f['piv2']['deltays_final'])))
            correl_map = correl > 0.3
            vec_x_cor = vec_x*correl_map
            vec_y_cor = vec_y*correl_map
            correl_cor = correl*correl_map
        vec_x_med.append([np.mean(val[val != 0]) for val in vec_x_cor.T])
        vec_y_med.append([np.mean(val[val != 0]) for val in vec_y_cor.T])
        corr_med.append([np.mean(val[val != 0]) for val in correl_cor.T])
        vec_x = vec_x[1:]
        vec_y = vec_y[1:]
        correl = correl[1:]
        count = count-1
    vec_x_med = np.array(vec_x_med)
    vec_y_med = np.array(vec_y_med)

    if replace_nan == True:
        corr_med = np.nan_to_num(corr_med)
        vec_x_med = np.nan_to_num(vec_x_med)
        vec_y_med = np.nan_to_num(vec_y_med)
    else:
        vec_x_med = np.array(vec_x_med)
        vec_y_med = np.array(vec_y_med)
        corr_med = np.array(corr_med)

    if two_d == True:
        path_to_dim = paths[0][:-17]
        # print(path_to_dim)
        vec_x_med = build_2d_array(vec_x_med, path_to_dim)
        vec_y_med = build_2d_array(vec_y_med, path_to_dim)
        corr_med = build_2d_array(corr_med, path_to_dim)
    mask = []
    path_masks = sorted(glob(path + 'Mask/Mask_*.png'))
    for f in path_masks[start:end]:
        frame = img.imread(f)
        kernel = np.ones((6, 6), np.uint8)
        #frame = cv2.dilate(frame, kernel, iterations=6)
        resized = skimage.transform.resize(frame, vec_x_med[0].shape)
        masked = np.ones(resized.shape)
        masked[resized == 0] = np.nan
        mask.append(masked)
    masks = np.array(mask)

    return np.array(vec_y_med, dtype='float64'), np.array(vec_x_med, dtype='float64'), corr_med, masks


def build_2d_array(data, path_to_dim):
    '''func to generate 2d data in image format of the single array stored 
    data'''
    # get a reference (y-coords) to determine the shape of the 2d array
    paths = sorted(glob(path_to_dim + '/piv_*.h5'))
    # print(paths)
    with h5py.File(paths[0]) as f:
        y = np.array(f['piv2']['ys'])
    # find the first max index in the field to determine the shape
    y_i = y.astype(np.uint16)
    s1 = np.argmax(y_i)+1
    s2 = int(len(y)/s1)
    # reshape the data
    data = [var.reshape(s2, s1).T for var in data]
    return data


def plot_quiver(vec_x, vec_y, mask, count_max, stepnumber):
    vec_xf = gaussian_smooth_replace_nan(vec_x, sigma=0.8)*mask[:len(vec_x)]
    vec_yf = gaussian_smooth_replace_nan(vec_y, sigma=0.8)*mask[:len(vec_y)]
    vec_mag = np.sqrt(vec_xf**2+vec_yf**2)
    x = np.arange(vec_x[0][:, 10:].shape[1])
    y = np.arange(vec_x[0][:, 10:].shape[0])

    for frame, vec in enumerate(vec_mag):
        fig = plt.figure()  # figsize=(8,5))
        ax = fig.add_subplot(111)
        plt.quiver(x, y, vec_yf[frame][:, 10:], vec_xf[frame][:, 10:],
                   vec_mag[frame][:, 10:],
                   scale=150)  # 190
        set_axes(fig, ax)
        plt.clim(0, 8)
        plt.colorbar(
            shrink=0.5,
            orientation='vertical',
        )
        plt.text(73, 45, 'U(mag)\n  [m/s]')
        ax.set_aspect('equal')
        plt.tight_layout()
        # plt.savefig(path +'quiver_sliding_mean_size=%i_%i.svg' %(count_max,
        #                                                   frame), format='svg')
        plt.savefig(path + 'quiver/quiver_sliding_mean_size=%i_%03d.png' % (count_max,
                                                                            frame+stepnumber), format='png')
        plt.close('all')


def streamlines(vec_x, vec_y, mask, count_max, stepnumber):
    vec_xf = gaussian_smooth_replace_nan(vec_x, sigma=1.2)*mask[len(vec_x)]
    vec_yf = gaussian_smooth_replace_nan(vec_y, sigma=1.2)*mask[len(vec_y)]
    vec_mag = [np.sqrt(x**2+vec_y[i]**2) for i, x in enumerate(vec_x)]
    x = np.arange(vec_x[0, :, 10:].shape[1])
    y = np.arange(vec_x[0, :, 10:].shape[0])
    for frame, vec in enumerate(vec_mag):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.streamplot(
            x,
            y,
            vec_y[frame][:, 10:],
            vec_x[frame][:, 10:],
            color=vec_mag[frame][:, 10:],
            density=1.5,
            arrowsize=0.4,
            arrowstyle='->',
            linewidth=1,
        )

        set_axes(fig, ax)
        plt.clim(0, 8)
        plt.colorbar(
            shrink=0.5,
            orientation='vertical',
        )
        plt.text(72, 45, 'U(mag)\n  [m/s]')
        ax.set_aspect('equal')
        plt.xlim(0, 69)
        plt.ylim(0, 49)
        plt.tight_layout()
        # plt.savefig(path  +'streamline_sliding_mean_size=%i_%i.svg' %(count_max,
        #                                                   frame), format='svg')
        plt.savefig(path + 'streamlines/streamline_sliding_mean_size=%i_%03d.png' % (count_max,
                                                                                     frame+stepnumber), format='png')
        plt.close('all')


def plot_corr_map(corr, count_max, stepnumber):
    for frame, corr in enumerate(corr):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        plt.imshow(corr[:, 10:], origin='lower')
        plt.clim(0, 1)
        plt.colorbar(orientation='vertical', shrink=0.5)
        set_axes(fig1, ax1)
        plt.tight_layout()
        ax1.set_aspect('equal')
        plt.text(70, 44, 'correlation\n       [-]')
        # plt.savefig(path + _'corr_map_sliding_mean_size=%i_%i.svg' %(count_max,
        #                                                   frame), format='svg')
        plt.savefig(path + 'correlation/corr_map_sliding_mean_size=%i_%03d.png' % (count_max,
                                                                                   frame+stepnumber), format='png')
        plt.close('all')


def vorticity(vec_x, vec_y, mask, count_max, stepnumber, filtered=True):
    curl = -(np.gradient(vec_y, axis=1) - np.gradient(vec_x, axis=0))

    for frame, curli in enumerate(curl):

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        plt.imshow(curli[:, 10:], origin='lower', vmin=-3, vmax=3)
        set_axes(fig1, ax1)
        plt.colorbar(shrink=0.5)
        plt.text(70, 43, 'vorticity\n   [1/s]')
        ax1.set_aspect('equal')
        plt.tight_layout()
        # plt.savefig(path + '_vorticity_sliding_mean_size=%i_%i.svg' %(count_max,
        #                                                   frame), format='svg')
        plt.savefig(path + 'vorticity/vorticity_sliding_mean_size=%i_%03d.png' % (count_max,
                                                                                  frame+stepnumber), format='png')
        if filtered == True:

            curli = gaussian_smooth_replace_nan(curli, sigma=1)
            curli = cv2.resize(
                np.array(curli),
                (
                    mask.shape[2],
                    mask.shape[1]
                ),
                fx=0,
                fy=0,
                interpolation=cv2.INTER_NEAREST
            )*mask[frame, :, :]
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            plt.imshow(curli[:, 200:], origin='lower', vmin=-2.5, vmax=2.5)
            set_axes_HD(fig1, ax1)
            plt.colorbar(shrink=0.5)
            plt.text(1110, 710, 'vorticity\n   [1/s]')
            ax1.set_aspect('equal')
            plt.tight_layout()
            # plt.savefig(path + '_vorticity_filtered_sliding_mean_size=%i_%i.svg' %(count_max,
            #                                                   frame), format='svg')
            plt.savefig(path + 'vorticity/vorticity_filtered_sliding_mean_size=%i_%03d.png' % (count_max,
                                                                                               frame+stepnumber), format='png')
        plt.close('all')


def calibrate_piv(vec_x, vec_y, mask, acq_rate, path, count_max, save_folder, stepnumber, plot=True, filtered=True):
    vec_mag = np.sqrt(vec_x**2 + vec_y**2)
    cam_res = (800, 1280)
    trail_edge = (439, 372)
    lead_edge = (439, 1086)

    origin_ph = (439, 907)
    shape = (800, 1280)
    distance = 0.066

    dist_px_x = lead_edge[0]-trail_edge[0]
    dist_px_y = lead_edge[1]-trail_edge[1]
    dist_px = np.sqrt(dist_px_x**2 + dist_px_y**2)

    calib = distance/dist_px

    U_x = calib * vec_x * acq_rate
    U_y = calib * vec_y * acq_rate

    U_mag = np.sqrt(U_x**2 + U_y**2)

    U_xf = gaussian_smooth_replace_nan(U_x)
    U_yf = gaussian_smooth_replace_nan(U_y)
    U_magf = np.sqrt(U_xf**2 + U_yf**2)
    U_mean = np.mean(np.nan_to_num(U_magf))
    if plot == True:
        for frame, U in enumerate(U_mag):
            fig1 = plt.figure()
            ax1 = fig1.gca()
            plt.imshow(U[:, 10:], origin='lower', vmin=0, vmax=8)
            ax1.set_clim = (0, 8)
            plt.colorbar(shrink=0.5)
            set_axes(fig1, ax1)
            plt.text(70, 45, 'U(mag)\n  [m/s]')
            ax1.set_aspect('equal')
            plt.tight_layout()
            # plt.savefig(path + path[-20:-1] +'_vorticity_sliding_mean_size=%i_%i.svg' %(count_max,
            #                                                   frame), format='svg')
            plt.savefig(path + save_folder + '/U_sliding_mean_size=%i_%05d.png' % (count_max,
                                                                                   frame+stepnumber), format='png')
            plt.close()
            if filtered == True:

                # print(mask.shape)
                U_ff = cv2.resize(
                    np.array(U_magf[frame]),
                    (
                        mask.shape[2],
                        mask.shape[1]
                    ),
                    fx=0,
                    fy=0,
                    interpolation=cv2.INTER_NEAREST
                )*mask[frame, :, :]

                fig2 = plt.figure()
                ax2 = fig2.gca()
                plt.imshow(U_ff[:, 200:]*mask[frame, :, 200:] /
                           U_mean, origin='lower', vmin=0, vmax=2)
                plt.colorbar(shrink=0.5)
                ax2.set_clim = (0, 2)
                set_axes_HD(fig2, ax2)
                plt.text(1120, 715, r'$\frac{U}{\bar{U}}$', fontsize=16)
                plt.text(1170, 710, '[-]', fontsize=13)
                ax2.set_aspect('equal')
                plt.tight_layout()
                '''plt.savefig(   
                                path +
                                'U/U_filtered_sliding_mean_size=%i_%05d.svg' 
                                %(count_max, frame), 
                                format='svg'
                            )'''
                plt.savefig(
                    path + save_folder +
                    '/U_filtered_sliding_mean_size=%i_%05d.png'
                    % (count_max, frame + stepnumber),
                    format='png'
                )
            plt.close('all')

    return U_x, U_y, U_mag


def set_axes_HD(fig, ax):
    ax.set_xticks([57.9, 220.18, 382.5, 544.7, 707, 869.3, 1031.5])
    ax.set_xticklabels(np.round((np.array(
        [57.9, 220.18, 382.5, 544.7, 707, 869.3, 1031.5])-707)*9.243697478991598e-02, 0))
    ax.set_yticks([114.5, 276.7, 439, 601.3, 763.5])
    ax.set_yticklabels(np.round(
        (np.array([114.5, 276.7, 439, 601.3, 763.5])-439)*9.243697478991598e-02, 0))
    ax.set_ylabel('Y-axis [mm]')
    ax.set_xlabel('X-axis [mm]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def set_axes(fig, ax):
    ax.set_xticks([6, 16, 26, 36, 46, 56, 66])
    ax.set_xticklabels((np.array([6, 16, 26, 36, 46, 56, 66])-46)*1.5)
    ax.set_yticks([7, 17, 27, 37, 47])
    ax.set_yticklabels((np.array([7, 17, 27, 37, 47])-27)*1.5)
    ax.set_ylabel('Y-axis [mm]')
    ax.set_xlabel('X-axis [mm]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def gaussian_smooth_replace_nan(U, sigma=1.2):
    U_zero = U.copy()
    U_zero[np.isnan(U)] = 0
    U_zero = sp.ndimage.gaussian_filter(U_zero, sigma)
    U_nan = np.ones(U.shape)
    U_nan[np.isnan(U)] = 0
    U_nan = sp.ndimage.gaussian_filter(U_nan, sigma)

    return U_zero/U_nan


def streamlines_U(vec_x, vec_y, mask, count_max,stepnumber):
    vec_xf = gaussian_smooth_replace_nan(vec_x)
    vec_yf = gaussian_smooth_replace_nan(vec_y)

    vec_xf = np.array([cv2.resize(
        np.array(vec),
        (
            mask.shape[2],
            mask.shape[1]
        ),
        fx=0,
        fy=0,
        interpolation=cv2.INTER_NEAREST
    )*mask[i, :, :] for i, vec in enumerate(vec_xf)])

    vec_yf = np.array([cv2.resize(
        np.array(vec),
        (
            mask.shape[2],
            mask.shape[1]
        ),
        fx=0,
        fy=0,
        interpolation=cv2.INTER_NEAREST
    )*mask[i, :, :] for i, vec in enumerate(vec_yf)])

    vec_mag = np.sqrt(vec_xf**2 + vec_yf**2)
    y = np.arange(vec_xf.shape[1])
    x = np.arange(vec_xf.shape[2])
    U_mean = np.mean(np.nan_to_num(vec_mag))
    # print('x:',len(x))
    # print('y:',len(y))
    # print('u:',vec_xf.shape)
    # print('v:',vec_yf.shape)

    for frame, vec in enumerate(vec_mag):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.imshow(
            vec*mask[frame]/U_mean,
            origin='lower',
            vmin=0,
            vmax=8
        )

        plt.clim(0, 2)
        plt.colorbar(
            shrink=0.5,
            orientation='vertical',
        )

        plt.streamplot(
            x,
            y,
            vec_yf[frame],
            vec_xf[frame],
            color='w',
            density=1,
            arrowsize=0.4,
            arrowstyle='->',
            linewidth=0.75,
        )

        ax.set_xticks(
            np.array(
                [
                    57.9,
                    220.18,
                    382.5,
                    544.7,
                    707,
                    869.3,
                    1031.5
                ]
            )
            + 200
        )

        ax.set_xticklabels(
            np.round(
                (
                    np.array(
                        [
                            57.9,
                            220.18,
                            382.5,
                            544.7,
                            707,
                            869.3,
                            1031.5
                        ]
                    )-707
                )*9.243697478991598e-02, 0)
        )

        ax.set_yticks([114.5, 276.7, 439, 601.3, 763.5])
        ax.set_yticklabels(
            np.round(
                (np.array(
                    [
                        114.5,
                        276.7,
                        439,
                        601.3,
                        763.5
                    ]
                )-439)*9.243697478991598e-02, 0)
        )

        ax.set_ylabel('Y-axis [mm]')
        ax.set_xlabel('X-axis [mm]')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.text(1330, 715, r'$\frac{U}{\bar{U}}$', fontsize=16)
        plt.text(1380, 710, '[-]', fontsize=13)
        ax.set_aspect('equal')
        plt.xlim(200, 1280)
        plt.ylim(0, 800)
        plt.tight_layout()
        # plt.savefig(path + 'streamline_sliding_mean_size=%i_%i.svg' %(count_max,
        #                                                   frame), format='svg')

        plt.savefig(
            path + 'streamlines_U/'
            'streamline_U_sliding_mean_size=%i_%03d.png'
            % (count_max, frame + stepnumber),
            format='png'
        )

        plt.close('all')


def get_masks(path, start, end, shape):
    mask = []
    mask_org = []
    path_masks = sorted(
        glob(str(Path(path).parent / 'undilated_mask/mask_*.png')))
    for f in path_masks[start:end]:
        frame = img.imread(f)
        #kernel = np.ones((6,6), np.uint8)
        #frame = cv2.dilate(frame, kernel, iterations=6)
        resized = skimage.transform.resize(frame, shape)
        masked = np.ones(resized.shape)
        masked[resized == 0] = np.nan
        mask.append(masked)
        masked_org = np.ones(frame.shape)
        masked_org[frame == 0] = np.nan
        mask_org.append(masked_org)
    masks = np.array(mask)
    masks_org = np.array(mask_org)
    return masks, masks_org


if __name__ == "__main__":

    plt.close('all')
    # plt.ion()
    plt.ioff()

    start = 4
    end = 16500
    stepsize = 500
    corr = [0.3]
    overlap = [0.5]
    count_max = 4
    acq_rate = 4000.

    path1 = '/home/users/kosters3ir/useful/project/17OSCILLATING/FSI_Benchmark_case/'
    path2 = ["flex_sinus_30_2017_11_02_1238/"]
    # path2 = [
    #    'Darrieus_42_6000_30_1609_500_500_4_40_4000_flex/',
    #    'Darrieus_84_6000_30_1604_500_500_4_40_4000_flex/',
    #    'Darrieus_235_9000_30_1553_500_500_4_40_4000_flex/',
    #    'Darrieus_470_9000_30_1543_500_500_4_40_4000_flex/',
    #    'Darrieus_500_9000_30_1525_500_500_4_40_4000_flex/',
    #    'Darrieus_765_9000_30_1514_500_500_4_40_4000_flex/',
    # ]
#
    path3 = 'masked/'

    for p in path2:
        print(p[:12])
        path = path1 + p + path3
        path_2 = 'piv_correl0.3_overlap0.5_window_size128/'

        save_folder = [
            'streamlines',
            'quiver',
            'U',
            'vorticity',
            'streamlines_U',
            'correlation'
        ]

        for folder in save_folder:
            if not os.path.exists(path + folder):
                os.makedirs(path + folder)
            os.chmod(path + folder, 0o774)
        stepnumber = 0
        for i in range(start, end, stepsize):
            print(np.round(i/end*100, 1), '% of the files processed\n')
            vec_x, vec_y, corr, masks = build_stack_and_compute(
                path,
                path_2,
                count_max,
                i-4,
                i+stepsize,
                two_d=True
            )
            masks, masks_org = get_masks(path, i-2, i+stepsize, vec_x[0].shape)

            U_x, U_y, U_mag = calibrate_piv(
                vec_x,
                vec_y,
                masks_org,
                acq_rate,
                path,
                count_max,
                save_folder[2],
                stepnumber
            )
            plt.close('all')
            plot_quiver(U_x, U_y, masks, count_max, stepnumber)
            plt.close('all')
            #streamlines(U_x, U_y, masks, count_max, stepnumber)
            # plt.close('all')
            #plot_corr_map(corr, count_max, stepnumber)
            # plt.close('all')
            vorticity(U_x, U_y, masks_org, count_max, stepnumber)
            plt.close('all')
            streamlines_U(U_x, U_y, masks_org, count_max, stepnumber)
            stepnumber += stepsize
    # os.chdir('/home/sthoerner/Promotion/PIV_2019_05_21_test_results/code/')
