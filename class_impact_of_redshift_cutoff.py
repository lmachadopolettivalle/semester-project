from classy import Class
from matplotlib import pyplot as plt
import numpy as np
import pyGenISW
GISW = pyGenISW.isw.SphericalBesselISW()
TCMB = GISW.Tcmb

h0 = 0.6736

class_unit_factor = (TCMB * 1e6)**2

common_settings = {
    'h':h0,
    'omega_b':0.0493*h0**2,
    #'omega_cdm':0.209*h0**2,
    'omega_m': 0.26*h0**2,
    'A_s':3.0589e-09,
    'n_s':0.9649,
    'tau_reio':0.05430842,
    'T_cmb': TCMB,
    # output and precision parameters
    'output':'tCl,pCl,lCl',
    'lensing':'yes',
    'l_max_scalars':5000,
    'temperature contributions': 'lisw',
}

M = Class()

# Test different early/late cutoffs
redshift_cutoffs = np.logspace(np.log10(1), np.log10(50), 10)

for cutoff in redshift_cutoffs:
    print(cutoff)
    common_settings['early_late_isw_redshift'] = cutoff

    M.set(common_settings)
    M.compute()
    cl_lisw = M.raw_cl(200)
    M.empty()

    plt.plot(cl_lisw['ell'], class_unit_factor * cl_lisw["tt"], label=f"Cutoff: {cutoff:.3f}")

print("Done with for loop")

plt.xlabel("$\ell$")
plt.ylabel(r"$C_{\ell} (\mu K^2)$")
plt.xlim([2, 200])
plt.ylim([1e-6, 1e3])
plt.yscale("log")
plt.grid()

plt.legend()

plt.savefig('class_redshift_cutoff.pdf')

plt.show()
