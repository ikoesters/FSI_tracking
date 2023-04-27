# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import pandas as pd
import scipy.signal as sig

# %%
# file = Path("rigid_sinus_30_2017_11_02_1121/computed_data/220810-1636.h5")
file = Path("computed_data/220814-0100.h5")

# %%
# %%
with h5py.File(file, mode="r") as f:
    df = pd.DataFrame(data={"aoa_ref": f["aoa_reference"][:, 0],
                      "aoa_meas": f["aoa_measured"][:, 0], "aoa_error": f["aoa_error"][:, 0]})

# %%
with h5py.File(file, mode="r") as f:
    df = pd.DataFrame(data={"aoa_ref": f["aoa_reference"][:],
                      "aoa_meas": f["aoa_measured"][:], "aoa_error": f["aoa_error"][:]})
# %% Indices where
zero_ind, _ = sig.find_peaks(-np.abs(df.aoa_ref))
zero_ind = [i for i in zero_ind if np.gradient(df.aoa_ref)[i] > 0]
period_len = np.mean(np.gradient(zero_ind)).astype("int")
nb_periods = (df.aoa_error.shape[0]//period_len)
# %%
df = df.iloc[:nb_periods*period_len]


def reshape_array(array): return np.reshape(
    array.values, (nb_periods, period_len))


aoa_error = reshape_array(df.aoa_error)
aoa_ref = reshape_array(df.aoa_ref)
aoa_meas = reshape_array(df.aoa_meas)
# %%
aoa_error_mean = np.mean(aoa_error, axis=0)
aoa_ref_mean = np.mean(aoa_ref, axis=0)
aoa_pos = np.column_stack(
    (np.linspace(0, 2*np.pi, len(aoa_error_mean)), aoa_error_mean))
dfpos = pd.DataFrame(aoa_pos, columns=["pos", "aoa"])

# %%
plt.plot(aoa_ref_mean, aoa_error_mean)
plt.xlabel("Angle of Attack [°]")
plt.ylabel("Mean AOA Error [°]")
