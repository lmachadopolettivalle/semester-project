import numpy as np
import healpy as hp
import pyGenISW

# Read run data
FILEPATH = "/cluster/home/lmachado/run_data/run_0/"
data = np.load(f"{FILEPATH}/shells.npy", mmap_mode='r')
info = np.load(f"{FILEPATH}/shell_info.npy")

# Obtain redshift values for each slice


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
zmax                = 2.                # maximum redshift for the SBT
zedge_min           =                   # the minimum of each redshift slice
zedge_max           =                   # the maximum of each redshift slice
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
    # TODO
    map_particle_count = hp.read_map(cosmo_path + 'CosmoML-shell_z-high=' + f'{zs_max[i]}' + '_z-low=' + f'{zs_min[i]}' + '.fits') 

    print('mean part count', np.mean(map_particle_count))

    map_slice = (map_particle_count/np.mean(map_particle_count)) - np.ones_like(map_particle_count) #very imoprtant: convert to zero mean overdenisty field (as required by PyGenISW)

    negatives = np.where(map_slice == -1.)[0]
    print(negatives)
    map_slice[negatives] =0 #fix this up so we dont have erroneous negative values- this would come from where we have 0 particles in the map (a problem for lower redshift slices) 

    GISW.slice2alm(map_slice, i) #this funtion saves to some 'temp path' defined in the source code- maybe re-work this and see where being saved 



   map_slice = # map of the density for redshift slice i corresponding to zedges_min[i]
   GISW.slice2alm(map_slice, i)

# converts spherical harmonic alms for the slices to the SBT coefficients
GISW.alm2sbt()

# you can store the SBT coefficients to avoid recomputing this again by running:
sbt_fname_prefix = # name of prefix for SBT file
GISW.sbt_save(prefix=sbt_fname_prefix)

# create ISW for contributions between zmin_isw and zmax_isw
alm_isw = GISW.sbt2isw_alm(zmin=zmin_isw, zmax=zmax_isw)

# convert alm to healpix map
nside = 256
map_isw = hp.alm2map(alm_isw, nside)*GISW.Tcmb
