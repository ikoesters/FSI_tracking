import numpy as np

from getData import GetData


class Bodyshape:
    def __init__(self, data_providing_obj: type[GetData]):
        """Generates outlines from a midline computed by segmenter.py
        based on the assumption of a 4-digit NACA foil."""
        self.gd = data_providing_obj

    def scaled_naca(self):
        camber, naca_upper, naca_lower = self.naca4()
        camber *= self.gd.chord_len
        naca_upper *= self.gd.chord_len
        naca_lower *= self.gd.chord_len
        return camber, naca_upper, naca_lower

    def naca4(self):
        m, p, t = self.gd.naca_4digit_code
        nb_points = self.gd.model_resolution

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
        yc_first = (m / p**2) * (2 * p * c_first - c_first**2)
        yc_last = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * c_last - c_last**2)
        yc = np.hstack((yc_first, yc_last))
        xc = np.hstack((c_first, c_last))
        camber = np.vstack((xc, yc)).T

        # Compute outline
        dy_first = (2 * m) / (p**2) * (p - c_first)
        dy_last = ((2 * m) / (1 - p) ** 2) * (p - c_last)
        dydx = np.hstack((dy_first, dy_last))
        angle = np.arctan(dydx)
        yt = (
            5
            * t
            * (
                0.2969 * np.sqrt(xc)
                - 0.1260 * xc
                - 0.3516 * xc**2
                + 0.2843 * xc**3
                - 0.1015 * xc**4
            )
        )
        upper = np.array((xc - yt * np.sin(angle), yc + yt * np.cos(angle))).T
        lower = np.array((xc + yt * np.sin(angle), yc - yt * np.cos(angle))).T

        return camber, upper, lower

    def _make_naca(self, midline_points: np.array) -> tuple:
        # Generate Naca foil
        midline = self.interpolate_midline_lengthwise(midline_points)

        # Generate Naca foil
        camber, naca_upper, naca_lower = self.scaled_naca()
        # dreht y-Werte -> das Profil um 180°
        naca_upper[:, 1] = np.flip(naca_upper[:, 1])
        naca_lower[:, 1] = np.flip(naca_lower[:, 1])
        camber[:, 1] = np.flip(camber[:, 1])

        naca_upper -= camber
        naca_lower -= camber

        dydx = np.gradient(midline[:, 1], midline[:, 0])
        angle = np.arctan(dydx)

        # Put foil outline around midline
        def outline_upper(correction_value: int):
            return np.array(
                (
                    midline[:, 0] - (naca_upper[:, 1] + correction_value) * np.sin(angle),
                    midline[:, 1] + (naca_upper[:, 1] + correction_value) * np.cos(angle),
                )
            ).T

        def outline_lower(correction_value: int):
            return np.array(
                (
                    midline[:, 0]
                    + (-naca_lower[:, 1] + correction_value) * np.sin(angle),
                    midline[:, 1]
                    - (-naca_lower[:, 1] + correction_value) * np.cos(angle),
                )
            ).T

        undil_corr_val = np.linspace(0, 0, self.model_res)
        upper_undil = outline_upper(undil_corr_val)
        lower_undil = outline_lower(undil_corr_val)

        dil_corr_val = np.linspace(
            self.gd.thickening_offset_at_tail,
            self.gd.thickening_offset_at_tail / 6,
            self.gd.model_resolution,
        )
        upper_dil = outline_upper(dil_corr_val)
        lower_dil = outline_lower(dil_corr_val)

        return (upper_dil, lower_dil), (upper_undil, lower_undil)
