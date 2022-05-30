import numpy as N  
from cora.signal import corr21cm
from drift.core import skymodel
from core import mpiutil


class kspace_cartesian():
    def __init__(self, kpar_start, kpar_end, kpar_dim, kperp_start, kperp_end, kperp_dim, kltrans):
        self.alpha_dim = kpar_dim * kperp_dim 
        self.kperp_dim = kperp_dim
        self.kpar_dim = kpar_dim
        self.telescope = kltrans.telescope
        self.beamtransfer = kltrans.beamtransfer
        self.kltrans = kltrans
        
        self.k_par_boundaries = N.linspace(kpar_start, kpar_end, kpar_dim + 1)
        self.k_par_centers = 0.5 * (self.k_par_boundaries[:-1] + self.k_par_boundaries[1:])
        
        self.k_perp_boundaries = N.linspace(kperp_start, kperp_end, kperp_dim + 1)
        self.k_perp_centers = 0.5 * (self.k_perp_boundaries[:-1] + self.k_perp_boundaries[1:])
        
        Aux1, Aux2 = N.broadcast_arrays(self.k_par_centers[:, N.newaxis], self.k_perp_centers)
        Aux = (Aux1 ** 2 + Aux2 ** 2) ** 0.5 
        self.k_pars = Aux1.flatten()
        self.k_perps = Aux2.flatten()
        self.k_centers = Aux.flatten()

        
    def make_response_matrix_sky(self, ind):
        """ make response matrix in sky basis
        """
        p_ind = self.make_binning_function(ind)
        return self.make_clzz(p_ind)

        
    def make_binning_function(self, band_ind):
        row_ind = int(band_ind / self.kperp_dim)
        col_ind = int(band_ind % self.kperp_dim)
        
        kpar_s = self.k_par_boundaries[row_ind]
        kpar_e = self.k_par_boundaries[row_ind+1]
        kperp_s = self.k_perp_boundaries[col_ind]
        kperp_e = self.k_perp_boundaries[col_ind+1]
        
        def band(k, mu):

            kpar = k * mu
            kperp = k * (1.0 - mu ** 2) ** 0.5

            parb = (kpar >= kpar_s) * (kpar <= kpar_e)
            perpb = (kperp >= kperp_s) * (kperp < kperp_e)

            return (parb * perpb).astype(N.float64)

        return band
    
    def make_binning_power(self):
        cr = corr21cm.Corr21cm()
        cr.ps_2d = False
        return cr.ps_vv(self.k_centers)
        
    def make_clzz(self, pk):
        """Make an angular powerspectrum from the input matter powerspectrum.
        -------
        clzz : [lmax+1, nfreq, nfreq]
        """
        crt = corr21cm.Corr21cm(ps=pk, redshift=1.5)
        crt.ps_2d = True

        clzz = skymodel.im21cm_model(
            self.telescope.lmax,
            self.telescope.frequencies,
            self.telescope.num_pol_sky,
            cr=crt,
            temponly=True,
        )
        
        return clzz         
    
    
class Covariances(kspace_cartesian):
    def fetch_response_matrix_list_sky(self):
        npol = self.telescope.num_pol_sky
        ldim = self.telescope.lmax + 1
        nfreq = self.telescope.nfreq
        self.resp_mat_shape = (npol, npol, ldim, nfreq, nfreq)

        aux_list = mpiutil.parallel_map(self.make_response_matrix_sky, list(range(self.alpha_dim)))
        
        self.para_ind_list=[]
        self.k_pars_used = []
        self.k_perps_used = []
        self.k_centers_used = []
        Resp_mat_list=[]
        for i in range(self.alpha_dim):
            if not N.all(aux_list[i]==0):  # Filter trivial bands
                self.para_ind_list.append(i)
                self.k_pars_used.append(self.k_pars[i])
                self.k_perps_used.append(self.k_perps[i])
                self.k_centers_used.append(self.k_centers[i])
                Resp_mat_list.append(aux_list[i])
        self.nonzero_alpha_dim=len(self.para_ind_list)
        return Resp_mat_list
    
        
    def make_response_matrix_kl_m(self, mi, response_matrix_list_sky, threshold = None):
        response_matrix_list_kl = []
        for i in range(self.nonzero_alpha_dim):
            mat = N.zeros(self.resp_mat_shape)
            mat[0,0,:,:,:] = response_matrix_list_sky[i]
            aux1 = self.kltrans.project_matrix_sky_to_kl(mi, mat, threshold)
            aux2 = (aux1 + aux1.conj().T)/2 # Make the quasi-Hermitian the exact Hermitian
            response_matrix_list_kl.append(aux2)
        return response_matrix_list_kl
    
   #make_covariance_kl_m(self.pvec, mi, Q_alpha_list, self.threshold)
        
    def make_covariance_kl_m(self, pvec, mi, response_mat_list_kl, threshold = None):
        
        cv_mat = self.make_noise_covariance_kl_m(mi, threshold)
        assert len(pvec)==self.nonzero_alpha_dim
        for i in range(self.nonzero_alpha_dim):
            cv_mat += pvec[i]*response_mat_list_kl[i]
        return cv_mat
    
    
    def make_foregrounds_covariance_sky(self):
        """ Construct foregrounds covariance in sky basis
        -------
        cv_fg  : N.ndarray[pol2, pol1, l, freq1, freq2]
        """

        KLclass = self.kltrans
        
            # If not polarised then zero out the polarised components of the array
        if KLclass.use_polarised:
            cv_fg = skymodel.foreground_model(
                        self.telescope.lmax,
                        self.telescope.frequencies,
                        self.telescope.num_pol_sky,
                        pol_length=KLclass.pol_length,
                        )
        else:
            cv_fg = skymodel.foreground_model(
                    self.telescope.lmax, self.telescope.frequencies, self.telescope.num_pol_sky, pol_frac=0.0
                )
        return cv_fg

    def make_instrumental_noise_telescope(self):
        """ Construct diagonal instrumental noise power in telescope basis

        """
        bl = N.arange(self.telescope.npairs)
        bl = N.concatenate((bl, bl))
        npower = self.telescope.noisepower(
                                           bl[N.newaxis, :], N.arange(self.telescope.nfreq)[:, N.newaxis]
                                           ).reshape(self.telescope.nfreq, self.beamtransfer.ntel)
        return npower



    def make_noise_covariance_kl_m(self, mi, threshold=None):
        """ Compute the noise covariances in the KL basis. The noise includes 
        both foregrounds and instrumental noise. This is for a single m-mode.
        Parameters
        ----------
        mi : integer
            The m-mode to calculate at.
        threshold: real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file.
        Returns
        -------
        cv_n_kl : N.ndarray[kl_len, kl_len]
            Noice covariance matrices.
        """
        cv_fg = self.make_foregrounds_covariance_sky()
        npower = self.make_instrumental_noise_telescope()
        # Project the foregrounds from the sky onto the telescope.
        cv_fg_svd = self.beamtransfer.project_matrix_sky_to_svd(mi, cv_fg)

        # Project into SVD basis and add into noise matrix
        cv_thermal_svd = self.beamtransfer.project_matrix_diagonal_telescope_to_svd(mi, npower)
    
        cv_totaln = cv_fg_svd + cv_thermal_svd

        # Project into KL basis
        cv_n_kl = self.kltrans.project_matrix_svd_to_kl(mi, cv_totaln, threshold)

        return (cv_n_kl + cv_n_kl.conj().T)/2
    
    #def generate_covariance(self, mi):
        
        #Q_alpha_list, CV = self.make_response_matrix_sky()
        
        
        
 
    