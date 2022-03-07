# import necessary modules
from classy import Class
from math import pi

#############################################
#
# Cosmological parameters and other CLASS parameters
h = 0.6736
#
common_settings = {# LambdaCDM parameters
                   'h':h,
                   'omega_b':0.0493*h**2,
                   'omega_cdm':0.209*h**2,
                   'A_s':2.100549e-09,
                   'n_s':0.9660499,
                   'tau_reio':0.05430842 ,
                   # output and precision parameters
                   'output':'tCl,pCl,lCl',
                   'lensing':'yes',
                   'l_max_scalars':5000}
#
M = Class()
#
###############
#
# call CLASS for the total Cl's and then for each contribution
#
###############
#
M.set(common_settings)
M.compute()
cl_tot = M.raw_cl(1000)
M.empty()           # reset input


M.set(common_settings)
M.set({'temperature contributions':'lisw'})
M.compute()
cl_lisw = M.raw_cl(1000)
M.empty()

# modules and settings for the plot
#
# uncomment to get plots displayed in notebook
import matplotlib
import matplotlib.pyplot as plt
# esthetic definitions for the plots
font = {'size'   : 16, 'family':'STIXGeneral'}
axislabelfontsize='large'
matplotlib.rc('font', **font)
matplotlib.mathtext.rcParams['legend.fontsize']='medium'
plt.rcParams["figure.figsize"] = [8.0,6.0]

#################
#
# start plotting
#
#################
#
plt.xlim([2,1000])
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_l^{TT} \,\,\, [\times 10^{10}]$")
plt.grid()
#
ell = cl_tot['ell']
factor = 1.e10
#plt.semilogx(ell,factor*cl_lisw['tt'],'y-',label=r'$\mathrm{late-ISW}$')
plt.plot(ell, factor*cl_lisw["tt"], label=r'$\mathrm{late-ISW}$')
plt.yscale("log")
#
plt.legend(loc='right',bbox_to_anchor=(1.4, 0.5))

plt.savefig('cltt_terms.pdf',bbox_inches='tight')

plt.show()
