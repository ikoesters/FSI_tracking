from pathlib import Path
from typing import Union

import h5py
import imageio
import numpy as np
import xarray as xr

from getData import GetData


class SaveData:
    def __init__(self, data_providing_obj: type[GetData]):
        """ """
        self.gd = data_providing_obj

    def create_h5_file(self, dataset_dict: dict, savepath: Path = Path.cwd()):
        # TODO: include attributes: cwd/measurement, date&time, total nb of images,
        # f.attrs["time"] = now
        savestr = savepath / f"{self.gd.create_timestamp()}.h5"
        with h5py.File(savestr, mode="w") as f:
            for _, kwargs in dataset_dict.items():
                f.create_dataset(**kwargs, compression="gzip")
        return savestr

    def save_to_h5(savestr, savedict, framenb=None):
        with h5py.File(savestr, mode="r+") as f:
            for _, kwargs in savedict.items():
                name = kwargs["name"]
                data = kwargs["data"]
                if framenb is None:
                    f[name][:] = data
                else:
                    f[name][framenb] = data

    def create_attrs_dict(class_object, exclude_list=None):
        if exclude_list is None:
            exclude_list = []
        class_attrs = {
            key: value
            for key, value in class_object.__dict__.items()
            if key not in exclude_list
        }
        return class_attrs

    def dump_attrs(h5_savepath, attrs_dict, class_name):
        with h5py.File(h5_savepath, mode="r+") as f:
            for key, val in attrs_dict.items():
                try:
                    f.attrs[f"{class_name}.{key}"] = val
                except TypeError:
                    f.attrs[f"{class_name}.{key}"] = str(val)

    def variable_to_string(variable):
        return f"{variable=}".split("=")[0]

    def create_data_entry_dict_NOT_WORKING(variable_list, save_dict):
        save_dict = save_dict.copy()
        for variable in variable_list:
            save_dict[variable_to_string(variable)]["data"] = variable
        return save_dict

    def create_data_entry_dict(variable_list, save_dict):
        save_dict = save_dict.copy()
        for key, value in zip(save_dict, variable_list):
            save_dict[key]["data"] = value
        return save_dict

    def save_fullsized_PNG_to_folder(
        savepath: Path, filename_base: str, frame: np.array, framenb: int
    ):
        savestr = savepath / f"{filename_base}_{framenb:04d}.png"
        imageio.imwrite(savestr, frame.T.astype("uint16"), format="PNG")

    def create_timestamp(formatstring: str = "%y%m%d-%H%M"):
        import datetime

        return datetime.datetime.now().strftime(formatstring)

    def save_xarray(data: Union[list, tuple, xr.DataArray]) -> None:
        # TODO: save xarray iteratively/recursively?
        pass
