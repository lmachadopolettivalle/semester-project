from collections import namedtuple
from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import pyGenISW

# Read run data
FILEPATH = "/cluster/home/lmachado/run_data/run_0/"
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

# Setup for the SBT
zmin                = 0.
zmax                = 1.4
# Lbox Units: Mpc/h (according to pyGenISW paper, Section 2.1)
Lbox                = 900
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
           boundary_conditions=boundary_conditions)

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

# Optionally, store SBT coefficients
sbt_fname_prefix = "testnamesbt" # Prefix for output file
GISW.save_sbt(prefix=sbt_fname_prefix)

# Compute ISW
# Optionally, pass in a range of redshifts within which to
# compute ISW
alm_isw = GISW.sbt2isw_alm()

# Convert Alm values to healpix map
nside = 256
map_isw = hp.alm2map(alm_isw, nside) * GISW.Tcmb

# After generating map, create power spectrum
cl = hp.anafast(map_isw, lmax=1000)
cl *= (1e6)**2 # anafast returns Kelvin^2, but want to plot in microKelvin^2
ell = np.arange(len(cl))

plt.figure(figsize=(10, 5))
plt.plot(ell, cl, label="pyGenISW")
plt.xlabel("$\ell$")
plt.ylabel(r"$C_{\ell} (\mu K^2)$")
plt.xlim([2,1000])
plt.yscale("log")
plt.grid()

# import necessary modules
from classy import Class
from math import pi

#############################################
#
# Cosmological parameters and other CLASS parameters
common_settings = {# LambdaCDM parameters
    'h':h0,
    'omega_b':0.0493*h0**2,
    'omega_cdm':0.209*h0**2,
    'A_s':3.0589e-09,
    'n_s':0.9649,
    'tau_reio':0.05430842 ,
    # output and precision parameters
    'output':'tCl,pCl,lCl',
    'lensing':'yes',
    'l_max_scalars':5000
}
#
M = Class()
#
###############
#
# call CLASS for the total Cl's and then for each contribution
#
###############
#
M.set(common_settings)
M.compute()
cl_tot = M.raw_cl(1000)
M.empty()           # reset input


M.set(common_settings)
M.set({'temperature contributions':'lisw'})
M.compute()
cl_lisw = M.raw_cl(1000)
M.empty()

#
ell = cl_tot['ell']
plt.plot(ell, (1e6)**2 * cl_lisw["tt"], label="Late ISW from CLASS")
#
plt.legend()

plt.savefig('angular_power_spectrum.pdf')

plt.show()
