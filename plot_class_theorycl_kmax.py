from classy import Class
from matplotlib import pyplot as plt
import numpy as np
from class_common_settings import *
import TheoryCL

class_unit_factor = (TCMB * 1e6)**2

M = Class()

# Test different early/late cutoffs
redshift_cutoffs = [0.32, 0.93]
BLUE = "#004488"
LIGHT_BLUE = "#003377"
DARK_BLUE = "#115599"
YELLOW = "#ddaa33"
LIGHT_YELLOW = "#cc9922"
DARK_YELLOW = "#eebb44"

Z_COLORS = {
    0.32: BLUE,
    0.93: YELLOW,
}

fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.4))

#########
# CLASS #
#########
for cutoff in redshift_cutoffs:
    CLASS_COMMON_SETTINGS['early_late_isw_redshift'] = cutoff

    M.set(CLASS_COMMON_SETTINGS)
    M.compute()
    cl_lisw = M.raw_cl(200)
    M.empty()

    ax.plot(
        cl_lisw['ell'],
        class_unit_factor * cl_lisw["tt"],
        label=f"CLASS, zmax: {cutoff:.2f}",
        color=Z_COLORS[cutoff],
    )


############
# TheoryCL #
############
kmax_values = [0.1, 0.5]
K_COLORS = {
    0.32: {
        0.1: LIGHT_BLUE,
        0.5: DARK_BLUE,
    },
    0.93: {
        0.1: LIGHT_YELLOW,
        0.5: DARK_YELLOW,
    },
}
K_MARKERS = {
    0.1: "o",
    0.5: "x",
}

for zmax in redshift_cutoffs:
    for kmax in kmax_values:
        print("Starting TheoryCL plot")

        SCL = TheoryCL.SourceCL()

        # Set cosmology
        SCL.cosmo(omega_m=CLASS_COMMON_SETTINGS["Omega_m"], omega_l=1-CLASS_COMMON_SETTINGS["Omega_m"], h0=h0, omega_b=Omega_b, ns=CLASS_COMMON_SETTINGS["n_s"], As=CLASS_COMMON_SETTINGS["A_s"], sigma8=sigma_8)

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
        zmax_los          = 5.            # maximum redshift to which integrals along the line-of-sight are computed to.
        rbin_num          = 1000          # number of bins along radial coordinates for integration.
        rbin_mode         = 'linear'      # linear or log binning schemes.
        kmin              = 1e-4          # minimum k for integrals, if None defaults to minimum value pre-calculated by CAMB.
        kmax              = kmax          # maximum k for integrals
        kbin_num          = 1000          # number of bins in Fourier coordinates for integration.
        kbin_mode         = 'log'         # linear or log binning schemes.
        switch2limber     = 2             # beyond this l we only compute the CLs using the Limber/flat-sky approximation.


        SCL.setup(lmax, zmin=zmin, zmax=zmax_los, rbin_num=rbin_num, rbin_mode=rbin_mode,
                kmin=kmin, kmax=kmax, kbin_num=kbin_num, kbin_mode=kbin_mode,
                switch2limber=switch2limber)

        # Define sources, for example the ISW and matter distribution between two redshifts
        SCL.set_source_ISW(zmin, zmax)

        SCL.get_CL() # Unitless

        ax.plot(
            SCL.L,
            class_unit_factor * SCL.CLs[:, 0],
            linestyle="--",
            linewidth=2.,
            label=f"TheoryCL, zmax={zmax:.2f}, kmax={kmax:.1f}",
            color=K_COLORS[zmax][kmax],
            marker=K_MARKERS[kmax],
            markevery=15
        )
        del SCL
        print("Done with TheoryCL")

print("Done with for loop")

ax.set_xlabel("$\ell$", fontsize=12)
plt.ylabel(r"$C_{\ell}$ [$\mu K^2$]", fontsize=12)
plt.title("CLASS vs. TheoryCL, Dependence on kmax", fontsize=12)
plt.xlim([2, 200])
plt.ylim([1e-7, 1e3])
plt.yscale("log")
plt.grid()

plt.legend()

plt.savefig("images/class_theorycl_kmax.pdf")

plt.show()

