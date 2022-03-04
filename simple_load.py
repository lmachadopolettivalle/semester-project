from collections import namedtuple
import numpy as np

FILEPATH = "/cluster/home/lmachado/run_data/run_0/"

data = np.load(f"{FILEPATH}/shells.npy", mmap_mode='r')
info = np.load(f"{FILEPATH}/shell_info.npy")

print(len(data[0]))

print(info[0])

ShellInfo = namedtuple(ShellInfo, ["fits_filename", "notsure", "z_low", "z_high", "dontknow1", "dontknow2", "dontknow3"])
