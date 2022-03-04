'''
This is a very rough sketch of the code to geenrate an ISW map realsation from a goroup of shells 
I was working on a group of shells that had their redshift ranges in the filename, this is not the case for the compressed shells- that information is now in the 'shell info' file
so these parts of the code will have to be changed
I have intentionally left out imports as the eventual idea of this project is to make an ISW generating algorithm that has no dependencies on external code that we can import intop the next release of UFalcon 
so we should start thinking about digging out the important algorithms in PyGenISW and making our own version in a python file 
'''


zmin                = 0.                # minimum redshift for the SBT
zmax                = 2.5               # maximum redshift for the SBT
zedge_min           = zs_min                  # the list of the lower z bin vals for each slice - caution here these are ot ordered 
zedge_max           = zs_max                  # the list of the upper z bin vals for each slice 
Lbox                = 2250.            # size of the simulation box- cehck these units pls - AR 05/01/22: noticed a lack of power on large scales so set this very high and see if situation improves 
kmin                = 2.*np.pi/Lbox     # minimum k
kmax                = 0.1               # maximum k
lmax                = None              # if you want to specify the maximum l for the SBT transform
nmax                = None              # if you want to specify the maximum n for the SBT transform
uselightcone        = True              # set to True if the
boundary_conditions = 'normal'          # boundary conditions for the SBT, either 'normal' or 'derivative'

GISW.setup(zmin, zmax, zedge_min, zedge_max, kmin=kmin, kmax=kmax,
           lmax=lmax, nmax=nmax, uselightcone=uselightcone,
           boundary_conditions=boundary_conditions)

# # converts each redshift slice into spherical harmonic alms

# cycle

part_per_side = 2080
exp_density = part_per_side**3/Lbox**3

print('expected denisty:', exp_density)
for i in range(len(zs_min)): 
    # load the shells
    #ensure the indexing is correct 
    map_particle_count = hp.read_map(cosmo_path + 'CosmoML-shell_z-high=' + f'{zs_max[i]}' + '_z-low=' + f'{zs_min[i]}' + '.fits') 

    print('mean part count', np.mean(map_particle_count))

    map_slice = (map_particle_count/np.mean(map_particle_count)) - np.ones_like(map_particle_count) #very imoprtant: convert to zero mean overdenisty field (as required by PyGenISW)

    negatives = np.where(map_slice == -1.)[0]
    print(negatives)
    map_slice[negatives] =0 #fix this up so we dont have erroneous negative values- this would come from where we have 0 particles in the map (a problem for lower redshift slices) 

    GISW.slice2alm(map_slice, i) #this funtion saves to some 'temp path' defined in the source code- maybe re-work this and see where being saved 

# converts spherical harmonic alms for the slices to the SBT coefficients
GISW.alm2sbt()

# # you can store the SBT coefficients to avoid recomputing this again by running:
# sbt_fname_prefix = # name of prefix for SBT file
# GISW.sbt_save(prefix=sbt_fname_prefix)

# create ISW for contributions between zmin_isw and zmax_isw
map_isw = GISW.sbt2isw_map(zmin=zmin, zmax=zmax) 
