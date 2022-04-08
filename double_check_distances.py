from classy import Class
from class_common_settings import CLASS_COMMON_SETTINGS, h0, TCMB, sigma_8, Omega_b
import numpy as np
import pyGenISW

#########
# CLASS #
#########
print("Starting CLASS computation")
M = Class()
M.set(CLASS_COMMON_SETTINGS)
M.compute()

background = M.get_background()

def get_r_from_z(z):
    # Note comov. dist is in Mpc, so must multiply by h0 to make into Mpc/h
    return h0 * np.interp(z, background["z"][::-1], background["comov. dist."][::-1])

print("CLASS")
print(get_r_from_z(0.32))
print(get_r_from_z(0.93))
print(get_r_from_z(1.4))

#######################
# TheoryCL / pyGenISW #
#######################
# Parameters obtained from params.yml
# Matter content: contains CDM + Baryons + Neutrinos
zmin_lookup         = 0.
zmax_lookup         = 10.
zbin_num            = 10000
zbin_mode           = "log"

GISW = pyGenISW.isw.SphericalBesselISW()
GISW.cosmo(omega_m=CLASS_COMMON_SETTINGS["Omega_m"], omega_l=1 - CLASS_COMMON_SETTINGS["Omega_m"], h0=h0, omega_b=Omega_b, ns=CLASS_COMMON_SETTINGS["n_s"], As=CLASS_COMMON_SETTINGS["A_s"], sigma8=sigma_8)
GISW.calc_table(zmin=zmin_lookup, zmax=zmax_lookup, zbin_num=zbin_num, zbin_mode=zbin_mode)

print("TheoryCL / pyGenISW")
print(float(GISW.get_rz(0.32)))
print(float(GISW.get_rz(0.93)))
print(float(GISW.get_rz(1.4)))
