import datetime
import os
from math import log
from pathlib import Path

from fluidimage.topologies.piv import TopologyPIV


class PIVfromPNG:
    def __init__(
        self,
        processing_threads: int = 1,
        save_path: str = "masked",
        image_path: str = "masked/",
        correl_val: float = 0.3,
        overlap_val: float = 0.5,
        window_size_val: int = 128,
    ) -> None:
        """_summary_

        Parameters
        ----------
        processing_threads : int, optional
            _description_, by default 1
        save_path : str, optional
            _description_, by default "masked"
        image_path : str, optional
            _description_, by default "masked/"
        correl_val : float, optional
            _description_, by default 0.3
        overlap_val : float, optional
            _description_, by default 0.5
        window_size_val : int, optional
            _description_, by default 128
        """
        os.environ["OMP_NUM_THREADS"] = str(processing_threads)
        self.save_path = Path(save_path)
        self.image_folder = Path(image_path)

        self.correl_val = correl_val
        self.overlap_val = overlap_val
        self.window_size_val = window_size_val

        self.image_path = image_path
        self.save_path = (
            self.save_path
            / f"piv_correl{correl_val}_overlap{overlap_val}_window_size{window_size_val}"
        )

        self.params = self.create_params()
        self.compute_piv(self.params)

    def create_params(self):
        postfix = datetime.datetime.now().isoformat()

        params = TopologyPIV.create_default_params()

        params.series.path = self.image_folder
        params.series.ind_start = 1
        params.series.ind_step = 1
        params.series.strcouple = "i:i+2"

        params.piv0.shape_crop_im0 = self.window_size_val
        params.piv0.grid.overlap = self.overlap_val
        params.multipass.number = int(
            log(self.window_size_val, 2) - 4
        )  # last window is 32x32
        params.multipass.use_tps = "True"
        params.fix.correl_min = self.correl_val

        params.saving.how = "recompute"
        params.saving.path = self.save_path
        params.saving.postfix = postfix
        return params

    def compute_piv(self, params, logging_level: str = "warning") -> None:
        self.save_path.mkdir(exist_ok=True)
        try:
            topology = TopologyPIV(params, logging_level=logging_level)
            print("Compute PIV")
            topology.compute(sequential=False)
        except ValueError:
            topology = TopologyPIV(params)
            print("Compute PIV")
            topology.compute(sequential=False)


if __name__ == "__main__":
    PIVfromPNG()
