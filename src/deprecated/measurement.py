import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pims
from matplotlib import rc

import _utils as ut


class Measurement:
    def __init__(
        self,
        path_film:str="./film.cine",
        path_angles:str="./pos_feedback.dat",
        angle_to_film_time_resolution:int=4,
        stop_frame:int=100,
    ):
        """Provide functionality to access the measuremt data: high-speed film in .cine format 
        and feedback of pitch angle from the motor resolver in .dat format.

        Parameters
        ----------
        path_film : str, optional
            Path to the high-speed .cine film, by default "./film.cine"
        path_angles : str, optional
            Path to the file containing the position feedback from the resolver in .dat format, by default "./pos_feedback.dat"
        angle_to_film_time_resolution : int, optional
            Capture frequency difference between film and position feedback, by default 4
        stop_frame : int, optional
            Process only until a specific frame is reached, by default 100
        """
        self.path_film = path_film
        self.path_angles = path_angles
        self.angle_to_film_time_resolution = angle_to_film_time_resolution
        self.stop_frame = stop_frame
        self.intermediates_save_path = Path("intermediates/")

        ut.mkdir(self.intermediates_save_path)
        self.act_frame = 0
        self.film = self.get_file()
        self.pos = self.pos_reader()
        self.frame_shape = self.get_frame(0).shape
        self.plot_intermediates = (
            True if self.intermediates_save_path is not None else False
        )

    def next_frame(self)->np.array:
        self.act_frame += 1
        frame = self.film.get_frame(self.act_frame).T
        return frame

    def get_file(self)->type[pims]:
        """read cinefile in specified path"""
        film = pims.open(str(self.path_film))
        return film

    def get_frame(self, frame_number:int)->np.array:
        """Extracts a frame from the cine file"""
        frame = self.film.get_frame(frame_number).T
        return frame

    def pos_reader(self)->np.array:
        """Reads airfoil position from Json file, returns angle in deg"""
        with open(self.path_angles, "r") as f:
            raw = json.load(f)
        rawpos = np.array(raw[1])
        triggertime = np.argmin(raw[10])
        pos = rawpos[triggertime:]
        pos = np.interp(
            np.linspace(0, 1, len(pos) * self.angle_to_film_time_resolution),
            np.linspace(0, 1, len(pos)),
            pos,
        )
        pos = pos[: self.film.image_count]
        return -pos


if __name__ == "__main__":
    import os

    root = os.path.dirname(__file__)
    meas = Measurement(
        root + "/testdata/film.cine", root + "/testdata/pos_feedback.dat"
    )
    print("Ende")