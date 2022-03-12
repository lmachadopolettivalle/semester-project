from collections import namedtuple
from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import pyGenISW
from classy import Class

# Read run data
DATAPATH = "/cluster/home/lmachado/run_data"
BOXSIZE = 2250 # Can be 900 or 2250
RUNINDEX = 0 # Index of simulation run
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
zmin                = 0.
zmax                = 2.
# Lbox Units: Mpc/h (according to pyGenISW paper, Section 2.1)
Lbox                = BOXSIZE
kmin                = 2.*np.pi/Lbox
# kmax selected to limit analysis to linear perturbations.
# Since non-linear perturbations happen for k > 0.1 h / Mpc
# However, this kmax depends on the redshift range under consideration.
# For instance, for z up to 2, kmax can be 0.02 instead of the 0.1 value for z = 0.
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

cl = hp.alm2cl(alm_isw)
cl *= (1e6)**2 # alm2cl returns Kelvin^2, but want to plot in microKelvin^2
ell = np.arange(len(cl))

plt.plot(ell, cl, label="pyGenISW")
plt.xlabel("$\ell$")
plt.ylabel(r"$C_{\ell} (\mu K^2)$")
plt.xlim([0, 200])
plt.ylim([1e-6, 1e3])
plt.yscale("log")
plt.grid()

#########
# CLASS #
#########
common_settings = {
    'h':h0,
    'omega_b':0.0493*h0**2,
    'omega_cdm':0.209*h0**2,
    'A_s':3.0589e-09,
    'n_s':0.9649,
    'T_cmb': TCMB,
    'output':'tCl,pCl,lCl',
    'lensing':'yes',
    'temperature contributions': 'lisw',
    #'early_late_isw_redshift': zedge_max[-1]
}

M = Class()
M.set(common_settings)
M.compute()
cl_lisw = M.raw_cl(1000)
M.empty()


# Units: CLASS outputs in strange units.
# Need to multiply by (Tcmb*1e6)^2 to convert into microK^2
# According to https://github.com/lesgourg/class_public/issues/304#issuecomment-671790592
# and https://github.com/lesgourg/class_public/issues/322#issuecomment-613941965
class_unit_factor = (TCMB * 1e6)**2
ell = cl_lisw['ell']
plt.plot(ell, class_unit_factor * cl_lisw["tt"], label="Late ISW from CLASS")

plt.legend()

plt.savefig('angular_power_spectrum.pdf')

plt.show()
