from classy import Class
from class_common_settings import CLASS_COMMON_SETTINGS, h0, TCMB, sigma_8, Omega_b
import healpy as hp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import TheoryCL

# Types of runs
RUN_TYPES = {
    "FIDUCIAL": "FIDUCIAL",
    "HIGH PARTICLE COUNT": "HIGH PARTICLE COUNT",
    "HIGH REDSHIFT RESOLUTION": "HIGH REDSHIFT RESOLUTION",
}

# List of colors
COLORS = {
    900: {
        "FIDUCIAL": "blue",
        "HIGH PARTICLE COUNT": "red",
        "HIGH REDSHIFT RESOLUTION": "green",
    },
    2250: {
        "FIDUCIAL": "orange",
    },
}

ZMAX = {
    900: 0.32,
    2250: 0.93,
}

# Maximum L for which to compute C_L
MAXIMUM_L = 200

# Configure shared plot
# Top = C_L vs l
# Bottom: C_L / C_L(CLASS) vs l
fig, ax = plt.subplots(1, 1)
divider = make_axes_locatable(ax)
ax2 = divider.append_axes("bottom", size="36%", pad="10%")
ax.figure.add_axes(ax2)


# List all available Alm files
ALM_FILES_DIRECTORY = "alm_files"
filenames = sorted(os.listdir(ALM_FILES_DIRECTORY))
# E.g. alm_zmax2.00_boxsize2250_runindex0.npy

# Setup for the SBT
zmin = 0.0

#########
# CLASS #
#########
print("Starting CLASS computation")

class_cl_dict = {}

# Units: CLASS outputs in strange units.
# Need to multiply by (Tcmb*1e6)^2 to convert into microK^2
# According to https://github.com/lesgourg/class_public/issues/304#issuecomment-671790592
# and https://github.com/lesgourg/class_public/issues/322#issuecomment-613941965
class_unit_factor = (TCMB * 1e6)**2

for boxsize, zmax in ZMAX.items():
    M = Class()
    CLASS_COMMON_SETTINGS["early_late_isw_redshift"] = zmax
    M.set(CLASS_COMMON_SETTINGS)
    M.compute()
    cl_lisw = M.raw_cl(MAXIMUM_L)
    M.empty()

    class_ell = cl_lisw['ell']
    class_cl = class_unit_factor * cl_lisw["tt"]
    class_cl_dict[boxsize] = class_cl
    ax.plot(class_ell, class_cl, label=f"CLASS, zmax={zmax}", c=COLORS[boxsize][RUN_TYPES["FIDUCIAL"]], ls="-")

print("Done with CLASS computation")


### TheoryCL
"""
print("Starting TheoryCL plot")

SCL = TheoryCL.SourceCL(TheoryCL.CosmoLinearGrowth())

# Set cosmology
SCL.cosmo(omega_m=CLASS_COMMON_SETTINGS["Omega_m"], omega_l=CLASS_COMMON_SETTINGS["Omega_m"], h0=h0, omega_b=Omega_b, ns=CLASS_COMMON_SETTINGS["n_s"], As=CLASS_COMMON_SETTINGS["A_s"], sigma8=sigma_8)

# Creates a table of the following linear growth functions for later interpolation:
# - r : comoving distance
# - H : Hubble parameter
# - D : linear growth rate
# - f : dlnD/dlna via approximation.
SCL.calc_table(zmin=0., zmax=10., zbin_num=10000, zbin_mode='log')

# Calculates the linear power spectra using CAMB and create callable interpolator.
SCL.calc_pk()

lmax              = 200           # maximum l mode to compute CLs.
zmin              = 0.            # minimum redshift integrals along the line-of-sight are computed to.
zmax              = 5.            # maximum redshift to which integrals along the line-of-sight are computed to.
rbin_num          = 1000          # number of bins along radial coordinates for integration.
rbin_mode         = 'linear'      # linear or log binning schemes.
kmin              = None          # minimum k for integrals, if None defaults to minimum value pre-calculated by CAMB.
kmax              = 1.            # maximum k for integrals
kbin_num          = 1000          # number of bins in Fourier coordinates for integration.
kbin_mode         = 'log'         # linear or log binning schemes.
switch2limber     = 200           # beyond this l we only compute the CLs using the Limber/flat-sky approximation.

SCL.setup(lmax, zmin=zmin, zmax=zmax, rbin_num=rbin_num, rbin_mode=rbin_mode,
          kmin=kmin, kmax=kmax, kbin_num=kbin_num, kbin_mode=kbin_mode,
          switch2limber=switch2limber)

# Define sources, for example the ISW and matter distribution between redshift 0 and 1.4.

zmin, zmax = 0.0, 0.93
SCL.set_source_ISW(zmin, zmax)
SCL.set_source_gal_tophat(zmin, zmax, 1.) # the 1. is the linear bias

SCL.get_CL() # Units: Kelvin^2

ax.plot(SCL.L_full, (1e6)**2 * SCL.CLs_full[:, 0], color="black", linestyle="--", linewidth=2., label="TheoryCL")
print("Done with TheoryCL")
"""
###

### pyGenISW

# Loop through pyGenISW Alm files
for filename in filenames:
    # Parse boxsize, zmax, runindex
    # E.g. alm_zmax2.0_boxsize2250_runindex0.npy
    # Note: this parsing works well because even the high resolution and
    # high particle count runs are stored with matching name patterns.
    parts = filename.split("_")
    zmax = float(parts[1][len("zmax"):])
    boxsize = int(parts[2][len("boxsize"):])
    runindex = int(parts[3].split(".")[0][len("runindex"):])

    # Only keep 900, zmax=0.32, and 2250, zmax=0.93
    if zmax > 0.94:
        continue

    # Determine type of run
    if "high" not in parts[0]:
        run_type = RUN_TYPES["FIDUCIAL"]
    elif "particle" in parts[0]:
        run_type = RUN_TYPES["HIGH PARTICLE COUNT"]
    elif "redshift" in parts[0]:
        run_type = RUN_TYPES["HIGH REDSHIFT RESOLUTION"]

    # Read Alm values computed using pyGenISW
    with open(f"{ALM_FILES_DIRECTORY}/{filename}", "rb") as f:
        alm_isw = np.load(f)

    cl = hp.alm2cl(alm_isw)
    cl *= (1e6)**2 # alm2cl returns Kelvin^2, but want to plot in microKelvin^2
    ell = np.arange(len(cl))

    ell = ell[:MAXIMUM_L + 1]
    cl = cl[:MAXIMUM_L + 1]

    ax.plot(
        ell,
        cl,
        label=f"Boxsize={boxsize} Mpc/h, zmax={zmax:.2f}, {run_type.title()}",
        ls="--",
        c=COLORS[boxsize][run_type],
    )

    # Compute cosmic variance for residual plots
    cosmic_variance = np.abs(class_cl_dict[boxsize]) * np.sqrt(2 / (2*class_ell + 1))

    # Compute fractional difference
    # Note that cl may have fewer than MAXIMUM_L elements, in which case we need to pad the difference array
    fractional_diff = (cl - class_cl_dict[boxsize][:len(cl)])/ cosmic_variance[:len(cl)]

    fractional_diff = np.pad(fractional_diff, (0, len(cosmic_variance) - len(fractional_diff)), "constant")

    ax2.plot(
        class_ell,
        fractional_diff,
        ls="--",
        c=COLORS[boxsize][run_type],
    )

# Add cosmic variance to residual plot for reference
ax2.fill_between(class_ell, [1]*len(class_ell), [-1]*len(class_ell), color="black", alpha=0.3)

# Plot settings
ax2.set_xlabel("$\ell$")

ax2.axhline(0, color="black")

ax.set_ylabel(r"$C_{\ell} (\mu K^2)$")
ax2.set_ylabel("Fractional change\n(units of cosmic variance)")

ax.set_xlim([0, 200])
ax2.set_xlim([0, 200])
ax.set_ylim([1e-9, 1e3])
ax2.set_ylim([-10, 10])

ax.set_yscale("log")

ax.grid()
ax2.grid()

ax.legend()

fig.suptitle("L-ISW Spectrum Comparison")

#plt.savefig("images/angular_power_spectrum_impact_boxsize.pdf")

print("Done")
plt.show()
