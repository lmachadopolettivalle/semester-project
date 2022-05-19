import numpy as np
from scipy import integrate, interpolate
import healpy as hp
import subprocess
import os
from astropy.cosmology import FlatLambdaCDM
from astropy import units
import bessel
from UFalcon import utils
from scipy.interpolate import interp1d
import uuid

class SphericalBesselISW():

    """Class for computing the ISW using spherical Bessel Transforms from maps
    of the density contrast given in redshift slices.
    """

    def __init__(self, cosmo_params):
        """Initialises the class.

        Parameters
        ----------
        CosmoLinearGrowth : class
            Parent class for calculating Cosmological linear growth functions.
        """
        self.Tcmb = 2.7255
        self.C = 3e8
        self.temp_path = None
        self.sbt_zmin = None
        self.sbt_zmax = None
        self.sbt_zedge_min = None
        self.sbt_zedge_max = None
        self.slice_in_range = None
        self.sbt_rmin = None
        self.sbt_rmax = None
        self.sbt_kmin = None
        self.sbt_kmax = None
        self.sbt_lmax = None
        self.sbt_nmax = None
        self.sbt_redge_min = None
        self.sbt_redge_max = None
        self.uselightcone = None
        self.temp_path = None
        self.boundary_conditions = None
        self.sim_dens = None

        self.cosmo = FlatLambdaCDM(H0=cosmo_params['H0'], Om0=cosmo_params['Omega_m'], Neff=cosmo_params['Neff'],Ob0=cosmo_params['Omega_baryon'], m_nu=cosmo_params['m_nu'], Tcmb0=cosmo_params['TCMB'])

        self.H0 = cosmo_params['H0']
        self.Om0 = cosmo_params['Omega_m']

    def calc_table(self, zmin=0., zmax=10., zbin_num=1000, zbin_mode='linear', alpha=0.55,
                   kind='cubic'):
        """Constructs table of cosmological linear functions to be interpolated for speed.

        Parameters
        ----------
        zmin : float
            Minimum redshift for tabulated values of the linear growth functions.
        zmax : float
            Maximum redshift for tabulated values of the linear growth functions.
        zbin_num : int
            Number of redshift values to compute the growth functions.
        zbin_mode : str
            Redshift binning, either linear or log of 1+z.
        alpha : float
            The power in the approximation to f(z) = Omega_m(z)**alpha
        kind : str
            The kind of interpolation used by the created interpolation functions as function of z and r.
        """
        # store some variables for table generation
        self.zmin = zmin # minimum redshift for table
        self.zmax = zmax # maximum redshift for table
        self.zbin_num = zbin_num # size of array
        self.zbin_mode = zbin_mode # linear or log
        self.f_alpha = alpha # for fz approximation
        # construct z array
        if zbin_mode == "linear":
            self.z_table = np.linspace(self.zmin, self.zmax, self.zbin_num)
        else:
            self.z_table = np.logspace(np.log10(zmin+1.), np.log10(zmax+1.), zbin_num) - 1.
        # constructs table of linear growth functions rz, Hz, Dz and fz
        self.rz_table = np.array([self.get_rz_no_interp(z_val) for z_val in self.z_table])
        self.Hz_table = np.array([self.get_Hz(z_val) for z_val in self.z_table])
        self.Dz_table = self.get_Dz(self.z_table)
        self.fz_table = self.get_fz_numerical(self.z_table[::-1], self.Dz_table[::-1])[::-1]

        # constructs callable interpolators for rz, Hz, Dz and fz
        self.rz_interpolator = interp1d(self.z_table, self.rz_table, kind=kind)
        self.Hz_interpolator = interp1d(self.z_table, self.Hz_table, kind=kind)
        self.Dz_interpolator = interp1d(self.z_table, self.Dz_table, kind=kind)
        self.fz_interpolator = interp1d(self.z_table, self.fz_table, kind=kind)
        # constructs callable interpolators for rz, Hz, Dz and fz as a function of r
        self.zr_interpolator = interp1d(self.rz_table, self.z_table, kind=kind)
        self.Hr_interpolator = interp1d(self.rz_table, self.Hz_table, kind=kind)
        self.Dr_interpolator = interp1d(self.rz_table, self.Dz_table, kind=kind)
        self.fr_interpolator = interp1d(self.rz_table, self.fz_table, kind=kind)


    def setup(self, zmin, zmax, zedge_min, zedge_max, kmin=None, kmax=0.1,
              lmax=None, nmax=None, uselightcone=True, temp_path='temp/',
              boundary_conditions='normal'):
        """Finds the slices that are required to compute the SBT coefficients from.

        Parameters
        ----------
        zmin : float
            Minimum redshift for spherical Bessel transform.
        zmax : float
            Maximum redshift for spherical Bessel transform.
        zedge_min : array
            Minimum redshift edge for each slice.
        zedge_max : array
            Maximum redshift edge for each slice.
        kmin : float
            Minium Fourier mode to consider.
        kmax : float
            Maximum Fourier mode to consider.
        lmax : int
            Maximum l mode to compute to, if None will be computed based on kmax.
        nmax : int
            Maximum n mode to comput to, if None will be computed based on kmax.
        uselightcone : bool
            True if density contrast maps are given as a lightcone and not all at
            redshift 0.
        boundary_conditions : str
            - normal : boundaries where spherical bessel function is zero.
            - derivative : boundaries where the derivative of the spherical Bessel
              function is zero.
        """
        if zedge_min.min() > zmin:
            print('zmin given,', zmin, 'is smaller than the zmin of the redshift slices. Converting zmin to zmin_edges.zmin().')
            self.sbt_zmin = zedge_min.min()
        else:
            self.sbt_zmin = zmin
        if zedge_max.max() < zmax:
            print('zmax given,', zmax, 'is larger than the zmax of the redshift slices. Converting zmax to zmax_edges.zmax().')
            self.sbt_zmax = zedge_max.max()
        else:
            # Add extra factor to zmax to improve boundary computations
            self.sbt_zmax = zmax + 0.2
        self.sbt_zedge_min = zedge_min
        self.sbt_zedge_max = zedge_max
        self.slice_in_range = np.where((self.sbt_zedge_min <= self.sbt_zmax))[0]

        self.sbt_rmin = self.get_rz(self.sbt_zmin)
        self.sbt_rmax = self.get_rz(self.sbt_zmax)

        self.sbt_kmin = kmin
        self.sbt_kmax = kmax
        if lmax is None:
            self.sbt_lmax = int(self.sbt_rmax*self.sbt_kmax) + 1
        else:
            self.sbt_lmax = lmax
        if nmax is None:
            self.sbt_nmax = int(self.sbt_rmax*self.sbt_kmax/np.pi) + 1
        else:
            self.sbt_nmax = nmax

        self.sbt_redge_min = self.get_rz(self.sbt_zedge_min)
        self.sbt_redge_max = self.get_rz(self.sbt_zedge_max)

        self.uselightcone = uselightcone
        self.temp_path = f"temp_{uuid.uuid4()}/"


        self.create_folder(self.temp_path)


        if boundary_conditions == 'normal' or boundary_conditions == 'derivative':
            self.boundary_conditions = boundary_conditions
        else:
            print("boundary_conditions can only be 'normal' or 'derivative', not", boundary_conditions)

#------------------------------------------------------------------------------------------------------------------------#
    #Now define the replacement functions from PyCosmo 
    #Todo add proper commenting with inputs and outputs

    #numerical derivative implementation in TheoryCL -- we can replace this but leave for now 
    def numerical_differentiate(self, x, f, equal_spacing=False, interpgrid=1000, kind='cubic'):
        """For unequally spaced data we interpolate onto an equal spaced 1d grid which
        we ten use the symmetric two-point derivative and the non-symmetric three point
        derivative estimator.
        Parameters
        ----------
        x : array
            X-axis.
        f : array
            Function values at x.
        equal_spacing : bool, optional
            Automatically assumes data is not equally spaced and will interpolate from it.
        interp1dgrid : int, optional
            Grid spacing for the interpolation grid, if equal spacing is False.
        kind : str, optional
            Interpolation kind.
        Returns
        -------
        df : array
            Numerical differentiation values for f evaluated at points x.
        Notes
        -----
        For non-boundary values:
        df   f(x + dx) - f(x - dx)
        -- = ---------------------
        dx            2dx
        For boundary values:
        df   - f(x + 2dx) + 4f(x + dx) - 3f(x)
        -- = ---------------------------------
        dx                  2dx
        """
        if equal_spacing == False:
            interpf = interp1d(x, f, kind=kind)
            x_equal = np.linspace(x.min(), x.max(), interpgrid)
            f_equal = interpf(x_equal)
        else:
            x_equal = np.copy(x)
            f_equal = np.copy(f)
        dx = x_equal[1] - x_equal[0]
        df_equal = np.zeros(len(x_equal))
        # boundary differentials
        df_equal[0] = (-f_equal[2] + 4*f_equal[1] - 3.*f_equal[0])/(2.*dx)
        df_equal[-1] = (f_equal[-3] - 4*f_equal[-2] + 3.*f_equal[-1])/(2.*dx)
        # non-boundary differentials
        df_equal[1:-1] = (f_equal[2:] - f_equal[:-2])/(2.*dx)
        if equal_spacing == False:
            interpdf = interp1d(x_equal, df_equal, kind=kind)
            df = interpdf(x)
        else:
            df = np.copy(df_equal)
        return df

    def create_folder(self, root, path=None):
        """Creates a folder with the name 'root' either in the current folder if path
        is None or a specified path.
        Parameters
        ----------
        root : str
            The name of the created folder.
        path : str, optional
            The name of the path of the created folder.
        """
        if path is None:
            if os.path.isdir(root) is False:
                subprocess.call('mkdir ' + root, shell=True)
        else:
            if os.path.isdir(path+root) is False:
                subprocess.call('mkdir ' + path + root, shell=True)

    def get_a(self,r): 
        'Get the scale factor a as a function of comoving distance r from an interpolation'
        #this function is quite hacky and could be refined... 

        return utils.a_of_r(r, self.cosmo)

    def get_rz(self, z): 
        return self.rz_interpolator(z)

    def get_rz_no_interp(self, z): 
        return (self.H0 / 100) * self.cosmo.comoving_distance(z).to(units.Mpc).value # comoving_distance returns Mpc. Multiplying by h gives Mpc/h.

    def get_zr(self, r): 
        #return redshift for a given comoving distance 
        return self.zr_interpolator(r)

    def get_zr_no_interp(self, r): 
        a = utils.a_of_r(r, self.cosmo)
 
        return 1/a -1 
    
    def get_Dr(self, r):
        return self.Dr_interpolator(r)

    def get_Dz(self, z):
        return utils.growth_z(z, self.cosmo)            

    def get_Hr(self, r):
        return self.Hr_interpolator(r)

    def get_Hz(self, z):
        return self.cosmo.H(z).value

    def get_fr(self, r):
        return self.fr_interpolator(r)

    def z2a(self, z):
        return 1./(1.+z)

    def get_fz_numerical(self, z, Dz, **kwargs):
        a = self.z2a(z)
        loga = np.log(a)
        logD = np.log(Dz)
        fz = self.numerical_differentiate(loga, logD, **kwargs)
        return fz

    def get_fz(self, z):
        return self.fz_interpolator(z)
    
#------------------------------------------------------------------------------------------------------------------------#



    def slice2alm(self, map_slice, index):
        """Given a density contrast map and its corresponding index (for its
        zedges minimum and maximum) slice2alm will convert the map to its
        spherical harmonics and save the files.

        Parameters
        ----------
        map_slice : array
            Healpix density contrast map.
        index : int
            Index of the slice for its zedges.
        """
        if index in self.slice_in_range:
            map_ = map_slice
            wl = hp.sphtfunc.pixwin(hp.get_nside(map_), lmax=self.sbt_lmax)
            alm = hp.map2alm(map_, lmax=self.sbt_lmax)
            alm = hp.almxfl(alm, 1./wl)
            condition = np.where(self.slice_in_range == index)[0]
            np.savetxt(self.temp_path+'map_alm_'+str(condition[0])+'.txt', np.dstack((alm.real, alm.imag))[0])
        else:
            print('Slice not in zmin and zmax range.')

    def alm2sbt(self):
        """Converts spherical harmonic coefficients in redshift slices to spherical
        Bessel coefficients. Stored as delta_lmn in units of (Mpc/h)^(1.5).
        """
        l = np.arange(self.sbt_lmax+1)[2:]
        n = np.arange(self.sbt_nmax+1)[1:]
        l_grid, n_grid = np.meshgrid(l, n, indexing='ij')
        self.l_grid = l_grid
        self.n_grid = n_grid
        qln_grid = np.zeros(np.shape(self.l_grid))
        print('Finding zeros for Bessel function up to n = '+str(self.sbt_nmax))
        for i in range(0, len(self.l_grid)):
            l_val = self.l_grid[i][0]
            if i < 10:
                if self.boundary_conditions == 'normal':
                    qln_grid[i] = bessel.get_qln(l_val, self.sbt_nmax, nstop=100)
                elif self.boundary_conditions == 'derivative':
                    qln_grid[i] = bessel.get_der_qln(l_val, self.sbt_nmax, nstop=100)
            else:
                if self.boundary_conditions == 'normal':
                    qln_grid[i] = bessel.get_qln(l_val, self.sbt_nmax, nstop=100,
                                                 zerolminus1=qln_grid[i-1],
                                                 zerolminus2=qln_grid[i-2])
                elif self.boundary_conditions == 'derivative':
                    qln_grid[i] = bessel.get_der_qln(l_val, self.sbt_nmax, nstop=100,
                                                     zerolminus1=qln_grid[i-1],
                                                     zerolminus2=qln_grid[i-2])

        self.kln_grid = qln_grid/self.sbt_rmax
        print('Constructing l and n value grid')
        if self.boundary_conditions == 'normal':
            self.Nln_grid = ((self.sbt_rmax**3.)/2.)*bessel.get_jl(self.kln_grid*self.sbt_rmax, self.l_grid+1)**2.
        elif self.boundary_conditions == 'derivative':
            self.Nln_grid = ((self.sbt_rmax**3.)/2.)*(1. - self.l_grid*(self.l_grid+1.)/((self.kln_grid*self.sbt_rmax)**2.))
            self.Nln_grid *= bessel.get_jl(self.kln_grid*self.sbt_rmax, self.l_grid)**2.
        if self.sbt_kmin is None and self.sbt_kmax is None:
            l_grid_masked = self.l_grid
            n_grid_masked = self.n_grid
            kln_grid_masked = self.kln_grid
            Nln_grid_masked = self.Nln_grid
        else:
            l_grid_masked = []
            n_grid_masked = []
            kln_grid_masked = []
            Nln_grid_masked = []
            for i in range(0, len(self.l_grid)):
                if self.sbt_kmin is None and self.sbt_kmax is None:
                    condition = np.arange(len(self.kln_grid[i]))
                elif self.sbt_kmin is None:
                    condition = np.where(self.kln_grid[i] <= self.sbt_kmax)[0]
                elif self.sbt_kmax is None:
                    condition = np.where(self.kln_grid[i] >= self.sbt_kmin)[0]
                else:
                    condition = np.where((self.kln_grid[i] >= self.sbt_kmin) & (self.kln_grid[i] <= self.sbt_kmax))[0]
                if len(condition) != 0:
                    l_grid_masked.append(self.l_grid[i, condition])
                    n_grid_masked.append(self.n_grid[i, condition])
                    kln_grid_masked.append(self.kln_grid[i, condition])
                    Nln_grid_masked.append(self.Nln_grid[i, condition])
            l_grid_masked = np.array(l_grid_masked, dtype=object)
            n_grid_masked = np.array(n_grid_masked, dtype=object)
            kln_grid_masked = np.array(kln_grid_masked, dtype=object)
            Nln_grid_masked = np.array(Nln_grid_masked, dtype=object)
        self.l_grid_masked = l_grid_masked
        self.n_grid_masked = n_grid_masked
        self.kln_grid_masked = kln_grid_masked
        self.Nln_grid_masked = Nln_grid_masked
        # New part
        print('Pre-compute spherical Bessel integrals')
        _interpolate_jl_int = []
        for i in range(0, len(self.l_grid_masked)):
            _xmin = 0.
            _xmax = (self.kln_grid_masked[i]*self.sbt_rmax).max() + 1.
            _x = np.linspace(_xmin, _xmax, 10000)
            _jl_int = np.zeros(len(_x))
            _jl_int[1:] = integrate.cumtrapz((_x**2.)*bessel.get_jl(_x, l_grid[i][0]), _x)
            _interpolate_jl_int.append(interpolate.interp1d(_x, _jl_int, kind='cubic', bounds_error=False, fill_value=0.))

        print('Computing spherical Bessel Transform from spherical harmonics')
        for which_slice in range(0, len(self.slice_in_range)):
            index = self.slice_in_range[which_slice]
            r_eff = (3./4.)*(self.sbt_redge_max[index]**4. - self.sbt_redge_min[index]**4.)/(self.sbt_redge_max[index]**3. - self.sbt_redge_min[index]**3.)
            Dz_eff = self.get_Dr(r_eff)
            Sln = np.zeros(np.shape(self.kln_grid))
            for i in range(0, len(l_grid)):
                if self.sbt_kmin is None and self.sbt_kmax is None:
                    condition = np.arange(len(self.kln_grid[i]))
                elif self.sbt_kmin is None:
                    condition = np.where(self.kln_grid[i] <= self.sbt_kmax)[0]
                elif self.sbt_kmax is None:
                    condition = np.where(self.kln_grid[i] >= self.sbt_kmin)[0]
                else:
                    condition = np.where((self.kln_grid[i] >= self.sbt_kmin) & (self.kln_grid[i] <= self.sbt_kmax))[0]
                if len(condition) != 0:
                    Sln[i, condition] += np.array([(1./(np.sqrt(self.Nln_grid_masked[i][j])*self.kln_grid_masked[i][j]**3.))*(_interpolate_jl_int[i](self.kln_grid_masked[i][j]*self.sbt_redge_max[index]) - _interpolate_jl_int[i](self.kln_grid_masked[i][j]*self.sbt_redge_min[index])) for j in range(0, len(self.l_grid_masked[i]))])
            data = np.loadtxt(self.temp_path + 'map_alm_'+str(which_slice)+'.txt', unpack=True)
            delta_lm_real = data[0]
            delta_lm_imag = data[1]
            delta_lm = delta_lm_real + 1j*delta_lm_imag
            if self.uselightcone == True:
                delta_lm /= Dz_eff
            if which_slice == 0:
                l_map, m_map = hp.Alm.getlm(hp.Alm.getlmax(len(delta_lm)))
                delta_lmn = np.zeros((self.sbt_nmax, len(delta_lm)), dtype='complex')
                conditions1 = []
                conditions2 = []
                for i in range(0, len(Sln[0])):
                    if self.sbt_kmin is None and self.sbt_kmax is None:
                        condition = np.arange(len(self.kln_grid[:, i]))
                    elif self.sbt_kmin is None:
                        condition = np.where(self.kln_grid[:, i] <= self.sbt_kmax)[0]
                    elif self.sbt_kmax is None:
                        condition = np.where(self.kln_grid[:, i] >= self.sbt_kmin)[0]
                    else:
                        condition = np.where((self.kln_grid[:, i] >= self.sbt_kmin) & (self.kln_grid[:, i] <= self.sbt_kmax))[0]
                    if len(condition) == 0:
                        lmax = 0
                    else:
                        lmax = self.l_grid[condition, i].max()
                    condition1 = np.where(self.l_grid[:, i] <= lmax)[0]
                    condition2 = np.where(l_map <= lmax)[0]
                    conditions1.append(condition1)
                    conditions2.append(condition2)
                conditions1 = np.array(conditions1, dtype=object)
                conditions2 = np.array(conditions2, dtype=object)
            for i in range(0, len(Sln[0])):
                _delta_lmn = np.zeros(len(delta_lm), dtype='complex')
                _delta_lmn[conditions2[i].astype('int')] = hp.almxfl(delta_lm[conditions2[i].astype('int')], np.concatenate([np.zeros(2), Sln[conditions1[i].astype('int'), i]]))
                delta_lmn[i] += _delta_lmn
            
        self.delta_lmn = delta_lmn


    def save_sbt(self, prefix=None):
        """Saves spherical Bessel transform coefficients.

        Parameters
        ----------
        prefix : str
            Prefix for file containing spherical Bessel transform.
        """
        if prefix is None:
            fname = 'sbt_zmin_'+str(self.sbt_zmin)+'_zmax_'+str(self.sbt_zmax)+'_lmax_'+str(self.sbt_lmax)+'_nmax_'+str(self.sbt_nmax)
        else:
            fname = prefix + '_sbt_zmin_'+str(self.sbt_zmin)+'_zmax_'+str(self.sbt_zmax)+'_lmax_'+str(self.sbt_lmax)+'_nmax_'+str(self.sbt_nmax)
        if self.boundary_conditions == 'normal':
            fname += '_normal.npz'
        elif self.boundary_conditions == 'derivative':
            fname += '_derivative.npz'
        np.savez(fname, kln_grid=self.kln_grid, kln_grid_masked=self.kln_grid_masked, l_grid_masked=self.l_grid_masked,
                 Nln_grid_masked=self.Nln_grid_masked, delta_lmn=self.delta_lmn)

    def sbt2isw_alm(self, zmin=None, zmax=None):
        """Returns the ISW spherical harmonics between zmin and zmax from the computed
        spherical Bessel Transform.

        Parameters
        ----------
        zmin : float
            Minimum redshift for ISW computation.
        zmax : float
            Maximum redshift for ISW computation.
        """
        if zmin is None:
            zmin = self.sbt_zmin
        if zmax is None:
            zmax = self.sbt_zmax
        r = np.linspace(self.get_rz(zmin), self.get_rz(zmax), 1000)

        
        Dz = self.get_Dr(r)
        Hz = self.get_Hr(r)
        fz = self.get_fr(r)


        DHF = Dz*Hz*(1.-fz)
        Iln = np.zeros(np.shape(self.kln_grid))
        for i in range(0, len(self.kln_grid)):
            if self.sbt_kmin is None and self.sbt_kmax is None:
                condition = np.arange(len(self.kln_grid[i]))
            elif self.sbt_kmin is None:
                condition = np.where(self.kln_grid[i] <= self.sbt_kmax)[0]
            elif self.sbt_kmax is None:
                condition = np.where(self.kln_grid[i] >= self.sbt_kmin)[0]
            else:
                condition = np.where((self.kln_grid[i] >= self.sbt_kmin) & (self.kln_grid[i] <= self.sbt_kmax))[0]
            if len(condition) != 0:
                Iln[i, condition] += np.array([(1./np.sqrt(self.Nln_grid_masked[i][j]))*integrate.simps(DHF*bessel.get_jl(self.kln_grid_masked[i][j]*r, self.l_grid_masked[i][j]), r) for j in range(0, len(self.l_grid_masked[i]))])

        alm_isw = np.zeros(len(self.delta_lmn[0]), dtype='complex')
        for i in range(0, len(self.delta_lmn)):
            alm_isw += hp.almxfl(self.delta_lmn[i], np.concatenate([np.zeros(2), Iln[:, i]/(self.kln_grid[:, i]**2.)]))

        alm_isw *= 3.*self.Om0*(self.H0**2.)/(self.C**3.)
        alm_isw *= 1e9/((self.H0*1e-2)**3.)

        return alm_isw

    def sbt2isw_map(self, zmin, zmax, nside=256):
        """Returns a healpix map of the ISW between zmin and zmax computed from
        the spherical Bessel Transform.

        Parameters
        ----------
        zmin : float
            Minimum redshift for ISW computation.
        zmax : float
            Maximum redshift for ISW computation.
        nside : int
            Nside for healpix map.
        """
        alm_isw = self.sbt2isw_alm(zmin, zmax)
        map_isw = hp.alm2map(alm_isw, nside)*self.Tcmb
        return map_isw

    def clean_temp(self):
        """Removes temporary spherical harmonic files."""
        if self.slice_in_range is not None:
            for i in range(0, len(self.slice_in_range)):
                subprocess.call('rm -r ' + self.temp_path, shell=True)

    def __del__(self):
        # When the instance is no longer needed,
        # remove the temp directory
        try:
            print("Removing temporary directory...")
            os.rmdir(self.temp_path)
            print("Successfully removed temporary directory.")
        except:
            print(f"Could not remove temporary directory ({self.temp_path})")
