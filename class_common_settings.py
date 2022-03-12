import pyGenISW

h0 = 0.6736

# CMB Temperature from pyGenISW
GISW = pyGenISW.isw.SphericalBesselISW()
TCMB = GISW.Tcmb

CLASS_COMMON_SETTINGS = {
    'h':h0,
    'omega_b':0.0493*h0**2,
    'omega_cdm':0.209*h0**2,
    'A_s':3.0589e-09,
    'n_s':0.9649,
    'T_cmb': TCMB,
    'output':'tCl,pCl,lCl',
    'lensing':'yes',
    'temperature contributions': 'lisw',
    #'early_late_isw_redshift': zedge_max[-1]
}
