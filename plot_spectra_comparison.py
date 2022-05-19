from classy import Class
from class_common_settings import CLASS_COMMON_SETTINGS, h0, TCMB, sigma_8, Omega_b
import healpy as hp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import TheoryCL

def main(indices, args):
    
    num_cores_of_job = int(os.environ['LSB_MAX_NUM_PROCESSORS'])

    os.environ["OMP_NUM_THREADS"] = str(num_cores_of_job) #use ~4 cores  

    REDSHIFTS = [0.32, 0.75, 0.93, 1.75, 3.5]
    BOXSIZES = [900, 2250]

    # List of colors
    COLORS = {
        0.32: "blue",
        0.75: "red",
        0.93: "orange",
        1.75: "green",
        3.5: "black",
    }

    # Maximum L for which to compute C_L
    MAXIMUM_L = 200

    # Configure shared plot
    # Top = C_L vs l
    # Bottom: C_L / C_L(CLASS) vs l
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.4))
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size="36%", pad="10%")
    ax.figure.add_axes(ax2)


    # List all available Alm files
    ALM_FILES_DIRECTORY = "alm_files"
    filenames = sorted(os.listdir(ALM_FILES_DIRECTORY))
    # E.g. alm_zmax0.93_boxsize2250.npy

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

    for zmax in REDSHIFTS:
        M = Class()
        CLASS_COMMON_SETTINGS["early_late_isw_redshift"] = zmax
        M.set(CLASS_COMMON_SETTINGS)
        M.compute()
        cl_lisw = M.raw_cl(MAXIMUM_L)
        M.empty()

        class_ell = cl_lisw['ell']
        class_cl = class_unit_factor * cl_lisw["tt"]
        class_cl_dict[zmax] = class_cl
        ax.plot(class_ell, class_cl, label=f"CLASS, zmax={zmax:.2f}", c=COLORS[zmax], ls="-")


    print("Done with CLASS computation")


    ############
    # TheoryCL #
    ############

    theorycl_ell_dict = {}
    theorycl_dict = {}

    for zmax in REDSHIFTS:
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

        lmax              = MAXIMUM_L     # maximum l mode to compute CLs.
        zmin              = 0.            # minimum redshift integrals along the line-of-sight are computed to.
        zmax_los          = 5.            # maximum redshift to which integrals along the line-of-sight are computed to.
        rbin_num          = 1000          # number of bins along radial coordinates for integration.
        rbin_mode         = 'linear'      # linear or log binning schemes.
        kmin              = 1e-4          # minimum k for integrals, if None defaults to minimum value pre-calculated by CAMB.
        kmax              = 0.1           # maximum k for integrals
        kbin_num          = 1000          # number of bins in Fourier coordinates for integration.
        kbin_mode         = 'log'         # linear or log binning schemes.
        switch2limber     = 2             # beyond this l we only compute the CLs using the Limber/flat-sky approximation.


        SCL.setup(lmax, zmin=zmin, zmax=zmax_los, rbin_num=rbin_num, rbin_mode=rbin_mode,
                kmin=kmin, kmax=kmax, kbin_num=kbin_num, kbin_mode=kbin_mode,
                switch2limber=switch2limber)

        # Define sources, for example the ISW and matter distribution between two redshifts
        SCL.set_source_ISW(zmin, zmax)

        SCL.get_CL() # Unitless

        theorycl_ell_dict[zmax] = SCL.L
        theorycl_dict[zmax] = class_unit_factor * SCL.CLs[:, 0]

        ax.plot(theorycl_ell_dict[zmax], theorycl_dict[zmax], color=COLORS[zmax], linestyle="--", linewidth=2., label=f"TheoryCL, zmax={zmax:.2f}")
        del SCL
        print("Done with TheoryCL")

        ###

    ############
    # pyGenISW #
    ############

    # Loop through pyGenISW Alm files
    for filename in filenames:
        # Parse boxsize, zmax
        # E.g. alm_zmax2.0_boxsize2250.npy
        if "alm" not in filename:
            continue
        parts = filename.split("_")
        zmax = float(parts[1][len("zmax"):])
        boxsize = int(parts[2][len("boxsize"):])

        # Read Alm values computed using pyGenISW
        with open(f"{ALM_FILES_DIRECTORY}/{filename}", "rb") as f:
            alm_isw = np.load(f)

        cl = hp.alm2cl(alm_isw)
        cl *= class_unit_factor # pyGenISW returns unitless ALMs. So must multiply by (1e6*TCMB)**2
        ell = np.arange(len(cl))

        ell = ell[:MAXIMUM_L + 1]
        cl = cl[:MAXIMUM_L + 1]

        ax.plot(
            ell,
            cl,
            label=f"Boxsize={boxsize} Mpc/h, zmax={zmax:.2f}",
            ls="--",
        )

        # Compute cosmic variance for residual plots
        cosmic_variance = np.abs(theorycl_dict[zmax]) * np.sqrt(2 / (2*theorycl_ell_dict[zmax] + 1))

        min_length = min(
            len(cl),
            len(theorycl_dict[zmax]),
        )

        # Compute fractional difference
        # Note that cl may have fewer than MAXIMUM_L elements, in which case we need to pad the difference array
        fractional_diff = (cl[:min_length] - theorycl_dict[zmax][:min_length]) / cosmic_variance[:min_length]

        #fractional_diff = np.pad(fractional_diff, (0, len(cosmic_variance) - len(fractional_diff)), "constant")

        ax2.plot(
                theorycl_ell_dict[zmax][:min_length],
                fractional_diff[:min_length],
            ls="--",
        )

    # Add cosmic variance to residual plot for reference
    ax2.fill_between(theorycl_ell_dict[zmax], [1]*len(theorycl_ell_dict[zmax]), [-1]*len(theorycl_ell_dict[zmax]), color="black", alpha=0.3)

    # Plot settings
    ax2.set_xlabel("$\ell$", fontsize=12)

    ax2.axhline(0, color="black")

    ax.set_ylabel(r"$C_{\ell}$ [$\mu K^2$]", fontsize=12)
    ax2.set_ylabel(r"$\sigma$ ($C_{\ell}$)", fontsize=12)

    ax.set_xlim([2, MAXIMUM_L])
    ax2.set_xlim([2, MAXIMUM_L])
    ax.set_ylim([1e-7, 1e3])
    ax2.set_ylim([-7.5, 7.5])

    ax.set_yscale("log")

    ax.grid()
    ax2.grid()

    ax.legend()

    ax.set_title("Late ISW Spectrum Comparison", fontsize=12)

    plt.savefig("images/HEHEHEpower_spectrum_sbt.pdf")

    print("Done")
    #plt.show()

    yield indices

