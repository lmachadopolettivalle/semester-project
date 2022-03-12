from classy import Class
from matplotlib import pyplot as plt
import numpy as np
from class_common_settings import CLASS_COMMON_SETTINGS, TCMB

class_unit_factor = (TCMB * 1e6)**2

M = Class()

# Test different early/late cutoffs
redshift_cutoffs = np.logspace(np.log10(1), np.log10(50), 10)

for cutoff in redshift_cutoffs:
    print(cutoff)
    CLASS_COMMON_SETTINGS['early_late_isw_redshift'] = cutoff

    M.set(CLASS_COMMON_SETTINGS)
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

plt.savefig("images/class_redshift_cutoff.pdf")

plt.show()
