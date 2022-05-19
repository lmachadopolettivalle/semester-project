import numpy as np
import sbt_class
import yaml 

# Inputs:
# 'data': maps for each redshift bin. np.array of shape (NUMBER_OF_REDSHIFT_BINS, NUMBER_OF_PIXELS)
# 'info': list of tuples (z_low, z_high) for each redshift bin. np.array of shape (NUMBER_OF_REDSHIFT_BINS, 2)
# 'zmax': maximum redshift
# 'cosmology.yml': filepath to cosmology parameters
# 'output_filepath': file to save resulting ALMs

# Steps:
# Read cosmology
# Create and configure SBT class
# Run through for loop and final functions.
# Output file with ALMs and return the ALMs.

def read_cosmology(filename):
    with open(filename) as f:
        cosmology = yaml.safe_load(f)
    return cosmology


def get_SBT_class_instance(cosmology):
    zmin_lookup         = 0.
    zmax_lookup         = 10.
    zbin_num            = 10000
    zbin_mode           = "log"

    SBT = sbt_class.SphericalBesselISW(cosmology)
    SBT.calc_table(zmin=zmin_lookup, zmax=zmax_lookup, zbin_num=zbin_num, zbin_mode=zbin_mode)

    return SBT

def get_recommended_zmax(boxsize, SBT):
    # zmax can be optionally specified by the user.
    # However, if zmax is not passed in, the pipeline will choose the recommended value of z_2boxsize,
    # with z_2boxsize such that r(z_2boxsize) = 2 * boxsize,
    # and r(z) is the comoving distance (in Mpc/h) from z=0 up to z.
    # The factor of 2 is motivated by experimentation
    # with boxsizes 900 Mpc/h and 2250 Mpc/h.
    # In both boxsizes, going up to larger zmax (i.e. beyond this factor of 2 in comoving distance)
    # leads to large, non-physical features in the resulting ISW maps from healpy.
    return SBT.get_zr(2 * boxsize)


def get_zero_mean_overdensity_from_map(map_slice):
    mean_count = np.mean(map_slice)
    return (map_slice - mean_count) / mean_count


def isw_pipeline(boxsize, maps, z_bins, cosmology_filepath, zmax=None, output_filepath=None):
    cosmology = read_cosmology(cosmology_filepath)

    zmin = 0.0
    zedge_min_list = np.array([row[0] for row in z_bins])
    zedge_max_list = np.array([row[1] for row in z_bins])

    print("Creating SBT class")
    SBT = get_SBT_class_instance(cosmology)

    if zmax is None:
        zmax = get_recommended_zmax(boxsize, SBT)
        print(f"zmax is None, using recommended zmax = {zmax:.2f}")

    print("Configuring SBT class")
    SBT.setup(zmin, zmax, zedge_min_list, zedge_max_list, kmin=2*np.pi / boxsize, kmax=0.1)


    # Convert redshift slices into Spherical Harmonic Alm values
    print("Computing slice2alm")
    for i, map_slice in enumerate(maps):
        print(f"Map: {i}")
        overdensity_map_slice = get_zero_mean_overdensity_from_map(map_slice)
        SBT.slice2alm(overdensity_map_slice, i)

    print("Done with for loop")

    # Convert Spherical Harmonic Alm values to SBT coefficients
    SBT.alm2sbt()

    # Compute ISW
    # Optionally, pass in a range of redshifts within which to
    # compute ISW
    alm_isw = SBT.sbt2isw_alm(zmin=zmin, zmax=zmax)

    if output_filepath is None:
        output_filepath = f"alm_files/alm_zmax{zmax:.2f}_boxsize{boxsize}.npy"

    # Save alm_isw to file
    with open(output_filepath, "wb") as f:     
        np.save(f, alm_isw)

    return alm_isw




if __name__ == "__main__":
    BOXSIZE = 900
    RUNINDEX = 0
    ZMAX = None

    cosmology_filepath = "cosmology.yml"

    if BOXSIZE == 900:
        FILEPATH = f"/cluster/work/refregier/alexree/isw_on_lightcone/cosmo_grid_sims/benchmarks/fiducial_bench/run_{RUNINDEX}/"
    else:
        FILEPATH = f"/cluster/work/refregier/alexree/isw_on_lightcone/cosmo_grid_sims/benchmarks/box_size/run_{RUNINDEX}/"
    
    maps = np.load(f"{FILEPATH}/shells.npy", mmap_mode='r')
    
    info = np.load(f"{FILEPATH}/shell_info.npy")
    z_bins = np.array([
        (row[2], row[3])
        for row in info
    ])

    isw_pipeline(BOXSIZE, maps, z_bins, cosmology_filepath, zmax=ZMAX, output_filepath=None)
