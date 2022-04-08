import argparse
from collections import namedtuple
import numpy as np
import pyGenISW
from class_common_settings import h0, sigma_8, Omega_b, CLASS_COMMON_SETTINGS
import uuid

# Collect data for boxsize, runindex, zmax
parser = argparse.ArgumentParser()

parser.add_argument("--boxsize", type=int, required=True)
parser.add_argument("--runindex", type=int, required=True)

args = parser.parse_args()

# Read run data
DATAPATH = "/cluster/home/lmachado/run_data"
BOXSIZE = args.boxsize # Can be 900 or 2250
RUNINDEX = args.runindex # Index of simulation run

zmin = 0.0

FILEPATH = f"{DATAPATH}/box_{BOXSIZE}/run_{RUNINDEX}"
data = np.load(f"{FILEPATH}/shells.npy", mmap_mode='r')
info = np.load(f"{FILEPATH}/shell_info.npy")

ShellInfo = namedtuple("ShellInfo", ["fits_filename", "id", "z_low", "z_high", "comoving_radius_z_low", "comoving_radius_z_high", "comoving_radius_z_avg"])
info = [ShellInfo(*row) for row in info]

# Obtain redshift values for each slice
zedge_min = np.array([row.z_low for row in info])
zedge_max = np.array([row.z_high for row in info])

# Configure pyGenISW
# Parameters obtained from params.yml
# Matter content: contains CDM + Baryons + Neutrinos
zmin_lookup         = 0.
zmax_lookup         = 10.
zbin_num            = 10000
zbin_mode           = "log"
GISW = pyGenISW.isw.SphericalBesselISW()
GISW.cosmo(omega_m=CLASS_COMMON_SETTINGS["Omega_m"], omega_l=1 - CLASS_COMMON_SETTINGS["Omega_m"], h0=h0, omega_b=Omega_b, ns=CLASS_COMMON_SETTINGS["n_s"], As=CLASS_COMMON_SETTINGS["A_s"], sigma8=sigma_8)
GISW.calc_table(zmin=zmin_lookup, zmax=zmax_lookup, zbin_num=zbin_num, zbin_mode=zbin_mode)

# Convert redshift slices into Spherical Harmonic Alm values
for i in range(0, len(zedge_min)):
    # Map of the density for redshift slice i
    counts = data[i]

    test_mean = np.mean(counts)

    # Compute mean from expected number of particles
    # based on average density in entire box
    AVERAGE_DENSITY = 0.92 ** 3 # particles / ((Mpc/h)^3)
    MIN_R = float(GISW.get_rz(zedge_min[i]))
    MAX_R = float(GISW.get_rz(zedge_max[i]))
    SLICE_VOLUME = (4*np.pi/3) * (MAX_R**3 - MIN_R**3)
    mean_count = AVERAGE_DENSITY * SLICE_VOLUME / len(counts)
    # Note that above, we divide mean_count by the number of pixels, i.e. the length of the map
    # Note: the length of the map should match hp.nside2npix(nside) with nside=2048

    print("HEHE")

    print(test_mean)
    print(mean_count)
    print((test_mean - mean_count)/test_mean)

    # Construct zero mean overdensity array
    map_slice = (counts - mean_count) / mean_count
