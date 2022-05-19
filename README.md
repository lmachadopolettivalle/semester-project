# ETH Semester Project

Spring 2022 Semester Project in Cosmology Group at ETH

## Overleaf Report

https://www.overleaf.com/project/6218bd0131771933cb77b756

## Setup to run code

### Install requirements
`module load gcc/6.3.0`

`module load python/3.7.4`

`module load hdf5`

`git clone https://github.com/knaidoo29/TheoryCL`

`python -m pip install -r requirements.txt`

```
git clone git@cosmo-gitlab.phys.ethz.ch:cosmo_public/esub-epipe.git
cd esub-epipe
python setup.py install
```

```
git clone git@cosmo-gitlab.phys.ethz.ch:cosmo/UFalcon.git
cd UFalcon
git checkout update_utils
python setup.py install
```

### Running a pipeline using epipe
1. Create a pipeline file (e.g. `pipeline.yml`)
1. Submit jobs with `epipe` (e.g. `epipe pipeline.yml`)

### Visualization
Once you have run the pipeline to generate ALMs and store them within the `alm_files` directory, you can use the several `plot*.py` scripts to plot different validation figures.

- `plot_class_zmax_impact.py`: CLASS spectra for different upper redshift cuts `zmax`.
- `plot_class_theorycl_kmax.py`: Compare CLASS and TheoryCL spectra for different `kmax` and `zmax` choices. Showcase that with a fixed `kmax`, increasing `zmax` improves the agreement between CLASS and TheoryCL at smaller scales (larger l).
- `plot_isw_maps.py`: Compute and display the ISW map from a pre-generated file of `alms`.
- `plot_spectra_comparison.py`: Script used to compare spectra between CLASS, TheoryCL, and the SBT algorithm. Can be executed using `epipe` and the `plot_pipeline.yml`. It is also part of the main `pipeline.yml`.
