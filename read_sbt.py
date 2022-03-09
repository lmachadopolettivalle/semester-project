from collections import namedtuple
from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import pyGenISW
from classy import Class

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

TCMB = GISW.Tcmb

# Setup for the SBT
zmin                = 0.
zmax                = 4.
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

data = np.load("testnamesbt_sbt_zmin_0.0_zmax_3.5_lmax_498_nmax_159_normal.npz", allow_pickle=True)
GISW.kln_grid = data['kln_grid']
GISW.kln_grid_masked = data['kln_grid_masked']
GISW.l_grid_masked = data['l_grid_masked']
GISW.Nln_grid_masked = data['Nln_grid_masked']
GISW.delta_lmn = data['delta_lmn']

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
plt.xlim([2, 200])
plt.ylim([1e-6, 1e3])
plt.yscale("log")
plt.grid()

plt.legend()

plt.savefig('read_sbt_plot.pdf')

plt.show()
