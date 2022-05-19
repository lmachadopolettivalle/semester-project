import argparse
import numpy as np
from pipeline import isw_pipeline
import os 


def setup(args):
    # The setup function gets executed first before esub starts (useful to create directories for example)

    print('RUNNING SETUP')

    parser = argparse.ArgumentParser()

    parser.add_argument("--boxsize", type=int, required=True)
    parser.add_argument("--runindex", type=int, required=True)
    parser.add_argument("--zmax", type=float, default=None)

    args = parser.parse_args(args)


    BOXSIZE = args.boxsize # Can be 900 or 2250
    RUNINDEX = args.runindex # Index of simulation run
    ZMAX = args.zmax


    return BOXSIZE, RUNINDEX, ZMAX
 

def main(indices, args): 

    num_cores_of_job = int(os.environ['LSB_MAX_NUM_PROCESSORS'])

    os.environ["OMP_NUM_THREADS"] = str(num_cores_of_job) #use ~8 cores for big speed gain 

    BOXSIZE, RUNINDEX, ZMAX = setup(args)

    # Read run data
    DATAPATH = "/cluster/work/refregier/alexree/isw_on_lightcone/cosmo_grid_sims/benchmarks"
    RUN_PATHS = {
        900: "fiducial_bench",
        2250: "box_size",
    }
    FILEPATH = f"{DATAPATH}/{RUN_PATHS[BOXSIZE]}/run_{RUNINDEX}"
    
    maps = np.load(f"{FILEPATH}/shells.npy", mmap_mode='r')
    
    info = np.load(f"{FILEPATH}/shell_info.npy")
    z_bins = np.array([
        (row[2], row[3])
        for row in info
    ])
    
    cosmology_filepath = "cosmology.yml"
    
    alms = isw_pipeline(BOXSIZE, maps, z_bins, cosmology_filepath, zmax=ZMAX, output_filepath=None)

    yield indices
