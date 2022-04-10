import argparse
from classy import Class
from collections import namedtuple
import numpy as np
import pyGenISW
from class_common_settings import h0, sigma_8, Omega_b, CLASS_COMMON_SETTINGS
import uuid

# Create CLASS object to compute comoving distances
M = Class()
M.set(CLASS_COMMON_SETTINGS)
M.compute()
background = M.get_background()
def get_r_from_z(z):
    # Note comov. dist is in Mpc, so must multiply by h0 to make into Mpc/h
    return float(h0 * np.interp(z, background["z"][::-1], background["comov. dist."][::-1]))

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

# Parameters obtained from params.yml
# Matter content: contains CDM + Baryons + Neutrinos
zmin_lookup         = 0.
zmax_lookup         = 10.
zbin_num            = 10000
zbin_mode           = "log"

GISW = pyGenISW.isw.SphericalBesselISW()
GISW.cosmo(omega_m=CLASS_COMMON_SETTINGS["Omega_m"], omega_l=1 - CLASS_COMMON_SETTINGS["Omega_m"], h0=h0, omega_b=Omega_b, ns=CLASS_COMMON_SETTINGS["n_s"], As=CLASS_COMMON_SETTINGS["A_s"], sigma8=sigma_8)
GISW.calc_table(zmin=zmin_lookup, zmax=zmax_lookup, zbin_num=zbin_num, zbin_mode=zbin_mode)

# Determine zmax as the redshift for which r(zmax) = BOXSIZE
zmax = float(GISW.get_zr(BOXSIZE)) # TODO use CLASS and interpolation instead of pyGenISW
zmax_SBT = zmax + 0.2
print(f"zmax for boxsize {BOXSIZE} is {zmax}")

# Filter data and info to contain only redshifts up to zmax

# Obtain redshift values for each slice
zedge_min = np.array([row.z_low for row in info])
zedge_max = np.array([row.z_high for row in info])

# CMB temperature
TCMB = GISW.Tcmb

# Setup for the SBT
# Lbox Units: Mpc/h (according to pyGenISW paper, Section 2.1)
Lbox                = BOXSIZE
kmin                = 2.*np.pi/Lbox
# kmax selected to limit analysis to linear perturbations.
# Since non-linear perturbations happen for k > 0.1 h / Mpc
kmax                = 0.3
lmax                = None
nmax                = None
uselightcone        = True
boundary_conditions = "normal" # Either normal or derivative

GISW.setup(zmin, zmax_SBT, zedge_min, zedge_max, kmin=kmin, kmax=kmax,
           lmax=lmax, nmax=nmax, uselightcone=uselightcone,
           boundary_conditions=boundary_conditions,
           temp_path=f"temp_zmax{zmax}_boxsize{BOXSIZE}_runindex{RUNINDEX}_{uuid.uuid4()}/")

# Convert redshift slices into Spherical Harmonic Alm values
for i in range(0, len(zedge_min)):
    # Map of the density for redshift slice i
    counts = data[i]

    # Compute mean from expected number of particles
    # based on average density in entire box
    AVERAGE_DENSITY = 0.92 ** 3 # particles / ((Mpc/h)^3)
    MIN_R = get_r_from_z(zedge_min[i])
    MAX_R = get_r_from_z(zedge_max[i])
    SLICE_VOLUME = (4*np.pi/3) * (MAX_R**3 - MIN_R**3)
    mean_count = AVERAGE_DENSITY * SLICE_VOLUME / len(counts)
    # Note that above, we divide mean_count by the number of pixels, i.e. the length of the map
    # Note: the length of the map should match hp.nside2npix(nside) with nside=2048

    # Double check mean count
    npmean = np.mean(counts)
    relative_diff_mean_count = (npmean - mean_count) / npmean
    print(f"{relative_diff_mean_count:.3f}")

    # Construct zero mean overdensity array
    map_slice = (counts - mean_count) / mean_count

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
alm_isw = GISW.sbt2isw_alm(zmin=0.0, zmax=zmax)

# Save alm_isw to file,
# to reduce computation time in the future
with open(f"alm_files/alm_zmax{zmax:.2f}_boxsize{BOXSIZE}_runindex{RUNINDEX}.npy", "wb") as f:
    np.save(f, alm_isw)
