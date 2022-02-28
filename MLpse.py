import numpy as N  
import scipy.linalg
from cora.signal import corr21cm
from drift.core import skymodel
from caput import config, mpiutil


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
        
        def fun(i):
            Q_alpha = N.zeros((npol,npol,ldim,nfreq,nfreq))
            Q_alpha[0,0,:,:,:] = self.make_response_matrix_sky(i)
            return Q_alpha
        
        # Filter some trivial bands
        aux_list = mpiutil.parallel_map(fun, list(range(self.alpha_dim)))
        
        self.para_ind_list=[]
        Resp_mat_list=[]
        for i in range(self.alpha_dim):
            if not N.all(aux_list[i]==0):
                self.para_ind_list.append(i)
                Resp_mat_list.append(aux_list[i])
        self.nonzero_alpha_dim=len(self.para_ind_list)
        return Resp_mat_list
    
        
    def make_response_matrix_kl_m(self, mi, response_matrix_list_sky, threshold = None):
        response_matrix_list_kl = []
        for i in range(self.nonzero_alpha_dim):
            mat = response_matrix_list_sky[i]
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
                    self.telescope.lmax, self.telescope.frequencies, npol, pol_frac=0.0
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



    def make_noise_covariance_kl_m(self,mi,threshold=None):
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
        
        
    
    
    
class Likelihood:
    def __init__(self, data_kl, Covariance_class_obj, Threshold = None):
        self.data_kl = data_kl
        self.mat_list = Covariance_class_obj.fetch_response_matrix_list_sky()
        self.pvec = None
        self.threshold = Threshold
        self.CV = Covariance_class_obj
        self.dim = self.CV.nonzero_alpha_dim
        self.nontrivial_mmode_list = self.filter_m_modes()
        parameters = self.CV.make_binning_power()
        self.parameter_model_values = []
        for i in self.CV.para_ind_list:
            self.parameter_model_values.append(parameters[i])
            
        
    def __call__(self, pvec):
        if self.pvec is pvec:
            return
        else:
            self.pvec = pvec                
            fun = mpiutil.parallel_map(
                                       self.make_funs_mi, self.nontrivial_mmode_list
                                       )
            # Unpack into separate lists of the log-likelihood function, jacobian, and hessian
            return sum(list(fun))
            
    def filter_m_modes(self):
        m_list = []
        for mi in range(self.CV.telescope.mmax + 1):
            if self.CV.kltrans.modes_m(mi)[0] is None:
                if mpiutil.rank0:
                    print("The m={} mode is null.".format(mi))
            else:
                m_list.append(mi)
        return m_list
    
    def make_funs_mi(self, mi):
        
        Q_alpha_list = self.CV.make_response_matrix_kl_m(mi, self.mat_list, self.threshold)
        C = self.CV.make_covariance_kl_m(self.pvec, mi, Q_alpha_list, self.threshold)
        len = C.shape[0]
        Identity = N.identity(len)
        vis_kl = self.data_kl[mi, :len]
        v_column = N.matrix(vis_kl.reshape((-1,1)))
        Dmat = v_column @ v_column.H
        
        C_inv = scipy.linalg.inv(C)
        C_inv_D = C_inv @ Dmat
        
        # compute m-mode log-likelihood
        fun_mi = N.trace(N.log(C)) + N.trace(C_inv_D)

        return fun_mi.real
    
    def calculate_Errors(self):
        fun = mpiutil.parallel_map(self.Fisher_m, self.nontrivial_mmode_list)
        return scipy.linalg.inv(sum(list(fun)))
    
    def Fisher_m(self,mi):
        Q_alpha_list = self.CV.make_response_matrix_kl_m(mi, self.mat_list, self.threshold)
        C = self.CV.make_covariance_kl_m(self.parameter_model_values, mi, Q_alpha_list, self.threshold)
        #len = C.shape[0]
        C_inv = scipy.linalg.inv(C)
        hess_mi = N.empty((self.dim,self.dim), dtype='complex128')
        for i in range(self.dim):
            for j in range(i, self.dim):
                hess_mi[i,j] = hess_mi[j,i] = N.trace(C_inv @ Q_alpha_list[i] @ C_inv @ Q_alpha_list[j])
        return hess_mi.real
    
class Likelihood_with_J_only(Likelihood):
    def __call__(self, pvec):
        if self.pvec is pvec:
            return
        else:
            self.pvec = pvec                
            Result = mpiutil.parallel_map(
                self.make_funs_mi, self.nontrivial_mmode_list
            )
            # Unpack into separate lists of the log-likelihood function, jacobian, and hessian
            fun, jac = list(zip(*Result))
            self.fun = sum(list(fun))
            self.jac = sum(list(jac))
    
    def make_funs_mi(self, mi):
        
        Q_alpha_list = self.CV.make_response_matrix_kl_m(mi, self.mat_list, self.threshold)
        C = self.CV.make_covariance_kl_m(self.pvec, mi, Q_alpha_list, self.threshold)
        len = C.shape[0]
        Identity = N.identity(len)
        vis_kl = self.data_kl[mi, :len]
        v_column = N.matrix(vis_kl.reshape((-1,1)))
        Dmat = v_column @ v_column.H
        
        C_inv = scipy.linalg.inv(C)
        C_inv = (C_inv + C_inv.conj().T)/2
        C_inv_D = C_inv @ Dmat
        
        # compute m-mode log-likelihood
        fun_mi = N.trace(N.log(C)) + N.trace(C_inv_D)
        #fun_mi = N.trace(N.log(C)) + N.einsum('ij,ji->', C_inv, Dmat)
        
        # compute m-mode Jacobian
        aux = (Identity - C_inv_D) @ C_inv
        pd = []
        for i in range(self.dim):
            # pd.append(N.trace(C_inv @ Q_alpha[i] @ (1. - C_inv @ self.Dmat))) 
            # N.einsum('ij,ji->', Q_alpha_list[i], aux)
            #aux1 = N.einsum('ij,ji->', C_inv, Q_alpha_list[i]) - N.einsum('ij,jk,kl,li->', C_inv, Q_alpha_list[i] , C_inv, Dmat)
            pd.append(N.trace(Q_alpha_list[i] @ aux)) 
        jac_mi = N.array(pd).reshape((self.dim,))
                
        return fun_mi.real, jac_mi.real
    
class Likelihood_with_J_H(Likelihood):

    def __call__(self, pvec):
        if self.pvec is pvec:
            return
        else:
            self.pvec = pvec                
            Result = mpiutil.parallel_map(
                self.make_funs_mi, self.nontrivial_mmode_list
            )
            # Unpack into separate lists of the log-likelihood function, jacobian, and hessian
            fun, jac, hess = list(zip(*Result))
            self.fun = sum(list(fun))
            self.jac = sum(list(jac))
            self.hess = sum(list(hess))
    
    def make_funs_mi(self, mi):
        Q_alpha_list = self.CV.make_response_matrix_kl_m(mi, self.mat_list, self.threshold)
        C = self.CV.make_covariance_kl_m(self.pvec, mi, Q_alpha_list, self.threshold)
        len = C.shape[0]
        Identity = N.identity(len)
        vis_kl = self.data_kl[mi, :len]
        v_column = N.matrix(vis_kl.reshape((-1,1)))
        Dmat = v_column @ v_column.H
        
        C_inv = scipy.linalg.inv(C)
        C_inv_D = C_inv @ Dmat
        
        # compute m-mode log-likelihood
        fun_mi = N.trace(N.log(C)) + N.trace(C_inv_D)
        
        # compute m-mode Jacobian
        aux = (Identity - C_inv_D) @ C_inv
        pd = []
        for i in range(self.dim):
            # pd.append(N.trace(C_inv @ Q_alpha[i] @ (1. - C_inv @ self.Dmat))) 
            # To save computing source, it can be simplified as
            pd.append(N.trace(Q_alpha_list[i] @ aux)) 
        jac_mi = N.array(pd).reshape((self.dim,))
            
        # compute m-mode Hessian
        hess_mi = N.empty((self.dim,self.dim), dtype='complex128')
        aux = (C_inv_D - 0.5*Identity) @ C_inv
        for i in range(self.dim):
            for j in range(i, self.dim):
                    #aux[i,j] = N.trace(C_inv @ Q_alpha[i] @ C_inv @ Q_alpha[j] @ (C_inv @ self.D - 0.5))
                hess_mi[i,j] = hess_mi[j,i] = 2. * N.trace(Q_alpha_list[i] @ C_inv @ Q_alpha_list[j] @ aux)
                
        return fun_mi.real, jac_mi.real, hess_mi.real


        
        
        
 
    
