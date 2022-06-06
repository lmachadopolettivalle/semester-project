from classy import Class
from matplotlib import pyplot as plt
import numpy as np
from class_common_settings import *

N = 10

plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Blues_r(np.linspace(0.1, 0.8, N)))

class_unit_factor = (TCMB * 1e6)**2

M = Class()

# Test different early/late cutoffs
redshift_cutoffs = np.logspace(np.log10(1), np.log10(50), N)

fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.4))

for cutoff in redshift_cutoffs:
    print(cutoff)
    CLASS_COMMON_SETTINGS['early_late_isw_redshift'] = cutoff

    M.set(CLASS_COMMON_SETTINGS)
    M.compute()
    cl_lisw = M.raw_cl(200)
    M.empty()

    ax.plot(cl_lisw['ell'], class_unit_factor * cl_lisw["tt"], label=f"zmax: {cutoff:.1f}")

print("Done with for loop")

ax.set_xlabel("$\ell$", fontsize=12)
plt.ylabel(r"$C_{\ell}$ [$\mu K^2$]", fontsize=12)
plt.title("CLASS Dependence on zmax", fontsize=12)
plt.xlim([2, 200])
plt.ylim([1e-7, 1e3])
plt.yscale("log")
plt.grid()

plt.legend()

plt.savefig("images/class_redshift_cutoff.pdf")

plt.show()

