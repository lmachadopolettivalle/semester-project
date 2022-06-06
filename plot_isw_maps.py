from class_common_settings import TCMB
import healpy as hp
from matplotlib import pyplot as plt
import numpy as np

ZMAX = 1.75
BOXSIZE = 900

FILENAME = f"alm_files/alm_zmax{ZMAX:.2f}_boxsize{BOXSIZE}.npy"

# Read Alm values computed using pyGenISW
with open(FILENAME, "rb") as f:
    alm_isw = np.load(f)

"""
# Clean up A_LM, removing low L values
cleanup = [0 if l < 20 else 1 for l in range(372)]
alm_isw = hp.almxfl(alm_isw, cleanup)
"""

# Compute ISW map
isw_map = hp.alm2map(alm_isw, nside=2048)
isw_map *= TCMB * 1e6 # Convert from relative units to microKelvin

hp.mollview(
    isw_map,
    title=f"zmax = {ZMAX}, boxsize = {BOXSIZE} Mpc/h", unit=r"$\Delta T_{ISW}$ [$\mu$K]",
    cmap=plt.get_cmap("RdBu_r"),
    min=-30,
    max=30,
)

#plt.savefig(f"images/ISWMap_{ZMAX}_{BOXSIZE}.pdf")

plt.show()
