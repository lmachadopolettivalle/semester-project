from collections import namedtuple
from matplotlib import pyplot as plt
import numpy as np
import healpy as hp
import pyGenISW

# Read run data
FILEPATH = "/cluster/home/lmachado/run_data/run_0/"
data = np.load(f"{FILEPATH}/shells.npy", mmap_mode='r')
info = np.load(f"{FILEPATH}/shell_info.npy")

ShellInfo = namedtuple("ShellInfo", ["fits_filename", "notsure", "z_low", "z_high", "dontknow1", "dontknow2", "dontknow3"])
info = [ShellInfo(*row) for row in info]

print(info)

# Obtain redshift values for each slice
zedge_min = np.array([row.z_low for row in info])
zedge_max = np.array([row.z_high for row in info])

# setup basic cosmology using the TheoryCL package and creating look up tables
# for the linear growth functions

omega_m             = 0.25              # matter density
omega_l             = 1 - omega_m       # lambda density
h0                  = 0.7               # Hubble constant (H0/100)
zmin_lookup         = 0.                # minimum redshift for lookup table
zmax_lookup         = 10.               # maximum redshift for lookup table
zbin_num            = 10000             # number of points in the lookup table
zbin_mode           = 'log'             # either 'linear' or 'log' binnings in z (log means log of 1+z)

GISW = pyGenISW.isw.SphericalBesselISW()
GISW.cosmo(omega_m=omega_m, omega_l=omega_l, h0=h0)
GISW.calc_table(zmin=zmin_lookup, zmax=zmax_lookup, zbin_num=zbin_num, zbin_mode=zbin_mode)

# spherical Bessel Transform (SBT) setup
zmin                = 0.                # minimum redshift for the SBT
zmax                = 4.                # maximum redshift for the SBT
#zedge_min           =                   # the minimum of each redshift slice
#zedge_max           =                   # the maximum of each redshift slice
Lbox                = 3072.             # size of the simulation box
kmin                = 2.*np.pi/Lbox     # minimum k
kmax                = 0.1               # maximum k
lmax                = None              # if you want to specify the maximum l for the SBT transform
nmax                = None              # if you want to specify the maximum n for the SBT transform
uselightcone        = True              # set to True if the
boundary_conditions = 'normal'          # boundary conditions for the SBT, either 'normal' or 'derivative'

GISW.setup(zmin, zmax, zedge_min, zedge_max, kmin=kmin, kmax=kmax,
           lmax=lmax, nmax=nmax, uselightcone=uselightcone,
           boundary_conditions=boundary_conditions)

# converts each redshift slice into spherical harmonic alms
for i in range(0, len(zedge_min)):
    """
    map_particle_count = hp.read_map(cosmo_path + 'CosmoML-shell_z-high=' + f'{zs_max[i]}' + '_z-low=' + f'{zs_min[i]}' + '.fits') 

    print('mean part count', np.mean(map_particle_count))

    map_slice = (map_particle_count/np.mean(map_particle_count)) - np.ones_like(map_particle_count) #very imoprtant: convert to zero mean overdenisty field (as required by PyGenISW)

    negatives = np.where(map_slice == -1.)[0]
    print(negatives)
    map_slice[negatives] =0 #fix this up so we dont have erroneous negative values- this would come from where we have 0 particles in the map (a problem for lower redshift slices) 

    GISW.slice2alm(map_slice, i) #this funtion saves to some 'temp path' defined in the source code- maybe re-work this and see where being saved 
    """
    # map of the density for redshift slice i corresponding to zedge_min[i]
    counts = data[i]

    # Clean up negatives in map slice
    # and enforce zero mean in overdensity array
    mean_count = np.mean(counts)
    map_slice = (counts - mean_count) / mean_count # Construct zero mean overdensity array
    mask_negatives = np.where(counts == 0.0)[0]
    map_slice[mask_negatives] = 0 # Resolve issue caused by missing particles

    # Compute alm from map_slice,
    # and store inside temp/ directory
    print(f" about to perform slice2alm in index i: {i}")
    GISW.slice2alm(map_slice, i)

print("Done with for loop")

# converts spherical harmonic alms for the slices to the SBT coefficients
GISW.alm2sbt()

# you can store the SBT coefficients to avoid recomputing this again by running:
sbt_fname_prefix = "testnamesbt"# name of prefix for SBT file
GISW.save_sbt(prefix=sbt_fname_prefix)

# create ISW for contributions between zmin_isw and zmax_isw
alm_isw = GISW.sbt2isw_alm()

# convert alm to healpix map
nside = 256
map_isw = hp.alm2map(alm_isw, nside) * GISW.Tcmb

# After generating map, create power spectrum
cl = hp.anafast(map_isw, lmax=100)
ell = np.arange(len(cl))

plt.figure(figsize=(10, 5))
plt.plot(ell, ell * (ell + 1) * cl)
plt.xlabel("$\ell$")
plt.ylabel("$\ell(\ell+1)C_{\ell}$")
plt.grid()
plt.savefig("angular_power_spectrum.png")
plt.show()
