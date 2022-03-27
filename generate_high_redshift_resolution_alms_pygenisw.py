import argparse
from astropy.io import fits
from collections import namedtuple
import healpy as hp
import numpy as np
import pyGenISW
from class_common_settings import h0, sigma_8, CLASS_COMMON_SETTINGS
import os

# Collect data for boxsize, runindex, zmax
parser = argparse.ArgumentParser()

parser.add_argument("--runindex", type=int, required=True)

args = parser.parse_args()

RUNINDEX = args.runindex # Index of simulation run
BOXSIZE = 900 # High redshift resolution data only exists for 900 Box

# Read run data
DATAPATH = f"/cluster/work/refregier/alexree/isw_on_lightcone/cosmo_grid_sims/benchmarks/redshift_resolution/run_{RUNINDEX}"

zmin = 0.0

# Loop through .fits files in DATAPATH,
# build info and data lists
def extract_zlow_zhigh_from_filename(filename):
    # Assumes filename in following format:
    # 'CosmoML-shell_z-high=0.5938637_z-low=0.5891474.fits
    zhigh = float(
        filename[
            filename.find("z-high=")+len("z-high=") : filename.find("_z-low")
        ]
    )
    zlow = float(
        filename[
            filename.find("z-low=")+len("z-low=") : filename.find(".fits")
        ]
    )
    
    return zlow, zhigh

filelist = os.listdir(DATAPATH)
filenames = sorted([f for f in filelist if f.endswith(".fits")], key=extract_zlow_zhigh_from_filename)

ShellInfo = namedtuple("ShellInfo", ["fits_filename", "id", "z_low", "z_high", "comoving_radius_z_low", "comoving_radius_z_high", "comoving_radius_z_avg"])

info = [
    ShellInfo(f, i, extract_zlow_zhigh_from_filename(f)[0], extract_zlow_zhigh_from_filename(f)[1], None, None, None) for i, f in enumerate(filenames)
]

# Helper function to read data from a .fits file
def data_from_file(filename):
    # TODO figure out correct way to return data
    return hp.read_map(filename)
    #with fits.open(filename) as f:
    #    return f[1].data

# Parameters obtained from params.yml
# Matter content: contains CDM + Baryons + Neutrinos
omega_m             = 0.26
omega_l             = 1 - omega_m
zmin_lookup         = 0.
zmax_lookup         = 10.
zbin_num            = 10000
zbin_mode           = "log"

GISW = pyGenISW.isw.SphericalBesselISW()
GISW.cosmo(omega_m=omega_m, omega_l=omega_l, h0=h0, ns=CLASS_COMMON_SETTINGS["n_s"], As=CLASS_COMMON_SETTINGS["A_s"], sigma8=sigma_8)
GISW.calc_table(zmin=zmin_lookup, zmax=zmax_lookup, zbin_num=zbin_num, zbin_mode=zbin_mode)

# Determine zmax as the redshift for which r(zmax) = BOXSIZE
zmax = float(GISW.get_zr(BOXSIZE))
print(f"zmax for boxsize {BOXSIZE} is {zmax}")

# Filter data and info to contain only redshifts up to zmax
info = [i for i in info if i.z_high < zmax]

# Obtain redshift values for each slice
zedge_min = np.array([row.z_low for row in info])
zedge_max = np.array([row.z_high for row in info])

# CMB temperature
TCMB = GISW.Tcmb

# Setup for the SBT
# Lbox Units: Mpc/h (according to pyGenISW paper, Section 2.1)
Lbox                = BOXSIZE
kmin                = 2 * np.pi / Lbox
# kmax selected to limit analysis to linear perturbations.
# Since non-linear perturbations happen for k > 0.1 h / Mpc
kmax                = 0.1
lmax                = None
nmax                = None
uselightcone        = True
boundary_conditions = "normal" # Either normal or derivative

GISW.setup(zmin, zmax, zedge_min, zedge_max, kmin=kmin, kmax=kmax,
           lmax=lmax, nmax=nmax, uselightcone=uselightcone,
           boundary_conditions=boundary_conditions,
           temp_path=f"temp_zmax{zmax}_boxsize{BOXSIZE}_runindex{RUNINDEX}/")

# Convert redshift slices into Spherical Harmonic Alm values
for i in range(0, len(zedge_min)):
    # Read data
    filename = info[i].fits_filename
    counts = data_from_file(f"{DATAPATH}/{filename}")

    # Map of the density for redshift slice i
    map_slice = counts

    # Clean up negatives in map slice
    # and enforce zero mean in overdensity array
    mean_count = np.mean(counts)
    map_slice = (counts - mean_count) / mean_count # Construct zero mean overdensity array
    mask_negatives = np.where(counts == 0.0)[0]

    map_slice[mask_negatives] = 0 # Resolve issue caused by missing particles

    # Compute alm from map_slice,
    # and store inside temp/ directory
    print(f"About to perform slice2alm in index i: {i}")
    GISW.slice2alm(map_slice, i)

    # Clean up data, to prevent running out of memory
    del counts

print("Done with for loop")

# Convert Spherical Harmonic Alm values to SBT coefficients
GISW.alm2sbt()

# Compute ISW
# Optionally, pass in a range of redshifts within which to
# compute ISW
alm_isw = GISW.sbt2isw_alm()

# Save alm_isw to file,
# to reduce computation time in the future
with open(f"alm_files/highredshiftresolutionalm_zmax{zmax:.2f}_boxsize{BOXSIZE}_runindex{RUNINDEX}.npy", "wb") as f:
    np.save(f, alm_isw)

# Clean up temp directory
GISW.clean_temp()
