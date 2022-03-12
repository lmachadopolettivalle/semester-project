import argparse
from collections import namedtuple
import numpy as np
import pyGenISW

# Collect data for boxsize, runindex, zmax
parser = argparse.ArgumentParser()

parser.add_argument("--boxsize", type=int, required=True)
parser.add_argument("--runindex", type=int, required=True)
parser.add_argument("--zmax", type=float, required=True)

args = parser.parse_args()

# Read run data
DATAPATH = "/cluster/home/lmachado/run_data"
BOXSIZE = args.boxsize # Can be 900 or 2250
RUNINDEX = args.runindex # Index of simulation run

zmin                = 0.0
zmax                = args.zmax

FILEPATH = f"{DATAPATH}/box_{BOXSIZE}/run_{RUNINDEX}"
data = np.load(f"{FILEPATH}/shells.npy", mmap_mode='r')
info = np.load(f"{FILEPATH}/shell_info.npy")

ShellInfo = namedtuple("ShellInfo", ["fits_filename", "id", "z_low", "z_high", "comoving_radius_z_low", "comoving_radius_z_high", "comoving_radius_z_avg"])
info = [ShellInfo(*row) for row in info]

# Obtain redshift values for each slice
zedge_min = np.array([row.z_low for row in info])
zedge_max = np.array([row.z_high for row in info])

# Parameters obtained from params.yml
# Matter content: contains CDM + Baryons + Neutrinos
omega_m             = 0.26
omega_l             = 1 - omega_m
h0                  = 0.6736
zmin_lookup         = 0.
zmax_lookup         = 10.
zbin_num            = 10000
zbin_mode           = "log"

GISW = pyGenISW.isw.SphericalBesselISW()
GISW.cosmo(omega_m=omega_m, omega_l=omega_l, h0=h0)
GISW.calc_table(zmin=zmin_lookup, zmax=zmax_lookup, zbin_num=zbin_num, zbin_mode=zbin_mode)

TCMB = GISW.Tcmb

# Setup for the SBT
# Lbox Units: Mpc/h (according to pyGenISW paper, Section 2.1)
Lbox                = BOXSIZE
kmin                = 2.*np.pi/Lbox
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
    # Map of the density for redshift slice i
    counts = data[i]
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

print("Done with for loop")

# Convert Spherical Harmonic Alm values to SBT coefficients
GISW.alm2sbt()

# Compute ISW
# Optionally, pass in a range of redshifts within which to
# compute ISW
alm_isw = GISW.sbt2isw_alm()

# Save alm_isw to file,
# to reduce computation time in the future
with open(f"alm_files/alm_zmax{zmax}_boxsize{BOXSIZE}_runindex{RUNINDEX}.npy", "wb") as f:
    np.save(f, alm_isw)
