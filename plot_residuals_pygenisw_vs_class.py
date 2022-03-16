from classy import Class
from class_common_settings import CLASS_COMMON_SETTINGS, h0, TCMB
import healpy as hp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os

# List of colors
COLORS = {
    900: "blue",
    2250: "orange",
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
    ax.plot(class_ell, class_cl, label=f"CLASS, zmax={zmax}", c=COLORS[boxsize], ls="-")

print("Done with CLASS computation")


# Compute cosmic variance for residual plots
cosmic_variance = np.sqrt(2 / (2*class_ell + 1))

# Loop through pyGenISW Alm files
for filename in filenames:
    # Parse boxsize, zmax, runindex
    # E.g. alm_zmax2.0_boxsize2250_runindex0.npy
    parts = filename.split("_")
    zmax = float(parts[1][len("zmax"):])
    boxsize = int(parts[2][len("boxsize"):])
    runindex = int(parts[3].split(".")[0][len("runindex"):])

    # Only keep 900, zmax=0.32, and 2250, zmax=0.93
    if zmax > 0.94:
        continue

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
        label=f"Boxsize={boxsize} Mpc/h, zmax={zmax:.2f}",
        ls="--",
        c=COLORS[boxsize],
    )

    # Compute fractional difference
    fractional_diff = (cl / cosmic_variance[:len(cl)]) - 1

    ax2.plot(
        ell,
        fractional_diff,
        ls="--",
        c=COLORS[boxsize],
    )

# Add cosmic variance to residual plot for reference
ax2.fill_between(class_ell, [1]*len(class_ell), [-1]*len(class_ell), color="black", alpha=0.3)

# Plot settings
ax2.set_xlabel("$\ell$")

ax2.axhline(0, color="black")

ax.set_ylabel(r"$C_{\ell} (\mu K^2)$")
ax2.set_ylabel("Fractional change")

ax.set_xlim([0, 200])
ax2.set_xlim([0, 200])
ax.set_ylim([1e-6, 1e3])
ax2.set_ylim([-2, 2])

ax.set_yscale("log")

ax.grid()
ax2.grid()

ax.legend()

fig.suptitle("L-ISW Spectrum, impact of Box Size")

#plt.savefig("images/angular_power_spectrum_impact_boxsize.pdf")

print("Done")
plt.show()
