from classy import Class
import numpy as np
from pipeline import read_cosmology

def get_CLASS_config(cosmology):
    return {
        "h": cosmology.get("H0", 70) / 100,
        "N_ncdm": cosmology.get("N_ncdm", 1),
        "Omega_m": cosmology.get("Omega_m", 0.26),
        "m_ncdm": cosmology.get("m_ncdm", 0.06),
        "N_ur": cosmology.get("N_ur", 2.0308),
        "A_s": cosmology.get("A_s", 2.1304e-9),
        "n_s": cosmology.get("n_s", 0.9649),
        "T_cmb": cosmology.get("TCMB", 2.73),
    }

def get_CLASS_instance(cosmology):
    class_config = get_CLASS_config(cosmology)
    M = Class()
    M.set(class_config)
    M.compute()
    background = M.get_background()

    return background

COSMOLOGY = read_cosmology("cosmology.yml")

CLASS_COMMON_SETTINGS = get_CLASS_config(COSMOLOGY)
CLASS_COMMON_SETTINGS.update({
    'output':'tCl,pCl,lCl',
    'lensing':'yes',
    'temperature contributions': 'lisw',
})

h0 = COSMOLOGY.get("H0", 70) / 100
sigma_8 = COSMOLOGY.get("sigma_8", 0.8)
Omega_b = COSMOLOGY.get("Omega_baryon", 0.05)
TCMB = COSMOLOGY.get("TCMB", 2.73)
