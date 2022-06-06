import numpy as N
import h5py
from cora.signal import corr21cm
from drift.core import skymodel
from core import mpiutil
from mpi4py import MPI


class kspace_cartesian:
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
        # aux_list = mpiutil.parallel_map(self.make_response_matrix_sky, list(range(self.alpha_dim)))
        self.para_ind_list = []
        self.k_pars_used = []
        self.k_perps_used = []
        self.k_centers_used = []
        Resp_mat_list = []
        for i in range(self.alpha_dim):
            aux_array = self.make_response_matrix_sky(i)
            if not N.all(aux_array==0):  # Filter trivial bands
                self.para_ind_list.append(i)
                self.k_pars_used.append(self.k_pars[i])
                self.k_perps_used.append(self.k_perps[i])
                self.k_centers_used.append(self.k_centers[i])
                Resp_mat_list.append(aux_array)
        self.nonzero_alpha_dim = len(self.para_ind_list)
        return Resp_mat_list
        
    def make_response_matrix_kl_m(self, mi, response_matrix_list_sky, threshold = None):
        response_matrix_list_kl = []
        for i in range(self.nonzero_alpha_dim):
            mat = N.zeros(self.resp_mat_shape)
            mat[0, 0, :, :, :] = response_matrix_list_sky[i]
            aux1 = self.kltrans.project_matrix_sky_to_kl(mi, mat, threshold)
            aux2 = (aux1 + aux1.conj().T)/2 # Make the quasi-Hermitian the exact Hermitian
            response_matrix_list_kl.append(aux2)
        return response_matrix_list_kl

    def make_covariance_kl_m(self, pvec, mi, response_mat_list_kl, threshold = None):
        cv_mat = self.make_noise_covariance_kl_m(mi, threshold)
        # assert len(pvec)==self.nonzero_alpha_dim
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

class Covariance_saveKL(Covariances):
    def __call__(self, filepath):
        npol = self.telescope.num_pol_sky
        ldim = self.telescope.lmax + 1
        nfreq = self.telescope.nfreq
        self.resp_mat_shape = (npol, npol, ldim, nfreq, nfreq)
        self.filter_m_modes() # Filter out trivial mmodes on KL basis
        self.filesavepath = filepath
        self.make_response_matrix()
        mpiutil.barrier()

    def make_covariance_kl_m(self, pvec, mi, threshold = None):
        cv_mat = self.make_noise_covariance_kl_m(mi, threshold)
        # assert len(pvec)==self.nonzero_alpha_dim
        for i in range(self.nonzero_alpha_dim):
            cv_mat += pvec[i]*self.load_Q_kl_mi_param(mi,self.para_ind_list[i])
        return cv_mat

    def load_Q_kl_list(self,mi):
        return [self.load_Q_kl_mi_param(mi, p) for p in self.para_ind_list]

    def load_Q_kl_mi_param(self,mi,param_ind):
        return h5py.File(self.filesavepath + str(param_ind) + ".hdf5",'r')[str(mi)][...].astype(N.csingle)

    def filter_m_modes(self):
        self.nontrivial_mmode_list = []
        for mi in range(self.telescope.mmax + 1):
            if self.kltrans.modes_m(mi)[0] is None:
                if mpiutil.rank0:
                    print("The m={} mode is null.".format(mi))
            else:
                self.nontrivial_mmode_list.append(mi)
        return

    def project_Q_sky_to_kl(self, mi, qsky):
        mat = N.zeros(self.resp_mat_shape)
        mat[0, 0, :, :, :] = qsky
        mproj = self.beamtransfer.project_matrix_sky_to_svd(mi, mat, temponly=True)
        result = self.kltrans.project_matrix_svd_to_kl(mi, mproj)
        return ((result + result.conj().T)/2).astype(N.csingle)

    def make_response_matrix(self):
        local_params = []
        #self.local_para_ind_list = []
        local_k_pars_used = []
        local_k_perps_used = []
        local_k_centers_used = []
        local_Resp_mat_list = []
        for i in mpiutil.partition_list_mpi(list(range(self.alpha_dim))):
            aux_array = self.make_response_matrix_sky(i)
            if not N.all(aux_array==0):
                #self.local_para_ind_list.append(i)
                local_params.append(i)
                local_k_pars_used.append(self.k_pars[i])
                local_k_perps_used.append(self.k_perps[i])
                local_k_centers_used.append(self.k_centers[i])
                local_Resp_mat_list.append(aux_array)

        local_size = N.array(len(local_params)).astype(N.int32)
        self.sendcounts = N.zeros(mpiutil.size, dtype=N.int32)
        self.displacements = N.zeros(mpiutil.size, dtype=N.int32)
        mpiutil._comm.Allgather([local_size, MPI.INT], [self.sendcounts, MPI.INT])
        self.nonzero_alpha_dim = N.sum(self.sendcounts)
        self.displacements[1:] = N.cumsum(self.sendcounts)[:-1]
        """
        for i in range(local_size):
            f = h5py.File(self.filesavepath + str(local_params[i])+'.hdf5', 'w')
            for mi in self.nontrivial_mmode_list:
                if mpiutil.rank0:
                    print(mi)
                f.create_dataset(str(mi), data=self.project_Q_sky_to_kl(mi, local_Resp_mat_list[i]))
            f.close()
        print('Process {} done!'.format(mpiutil.rank))
        mpiutil.barrier()
        """
        k_pars_used = N.empty(self.nonzero_alpha_dim)
        k_perps_used = N.empty(self.nonzero_alpha_dim)
        k_centers_used = N.empty(self.nonzero_alpha_dim)
        para_ind_list = N.zeros(self.nonzero_alpha_dim, dtype=N.int32)
        # Resp_mat_array = N.zeros((self.nonzero_alpha_dim, ldim, nfreq, nfreq), dtype=float)
        # aux_scale = ldim * nfreq * nfreq
        # aux_mpitype = MPI.DOUBLE.Create_contiguous(2)
        mpiutil._comm.Allgatherv([N.array(local_params).astype(N.int32), MPI.INT],
                                 [para_ind_list, self.sendcounts, self.displacements, MPI.INT])
        mpiutil._comm.Allgatherv([N.array(local_k_pars_used).astype(float), MPI.DOUBLE],
                                 [k_pars_used, self.sendcounts, self.displacements, MPI.DOUBLE])
        mpiutil._comm.Allgatherv([N.array(local_k_perps_used).astype(float), MPI.DOUBLE],
                                 [k_perps_used, self.sendcounts, self.displacements, MPI.DOUBLE])
        mpiutil._comm.Allgatherv([N.array(local_k_centers_used).astype(float), MPI.DOUBLE],
                                 [k_centers_used, self.sendcounts, self.displacements, MPI.DOUBLE])
        # mpiutil._comm.Allgatherv([N.array(local_Resp_mat_list).astype(float), MPI.DOUBLE],
        #                          [Resp_mat_array, sendcounts*aux_scale, displacements*aux_scale, MPI.DOUBLE])
        self.para_ind_list = para_ind_list
        self.k_pars_used = k_pars_used
        self.k_perps_used = k_perps_used
        self.k_centers_used = k_centers_used
        return


"""
    def save_Q_kl_m(self,mi):
        sendbuf = N.array([self.project_Q_sky_to_kl(mi, item)
                           for item in self.local_Resp_mat_list]).astype(complex)
        a, b=sendbuf.shape[-2:]
        recvbuf = N.zeros((self.nonzero_alpha_dim, a, b), dtype=complex)
        # large_dtype = MPI.COMPLEX16.Create_contiguous(a*b).Commit()
        mpiutil._comm.Allgatherv(sendbuf,
                                 [recvbuf, self.sendcounts*a*b, self.displacements*a*b, MPI.COMPLEX16])
        if mpiutil.rank0:
            if not N.all(recvbuf==0):
                with h5py.File(self.filesavepath, "w") as f:
                    f.create_dataset("{}".format(mi), data=recvbuf)
                self.nontrivial_mmode_list.append(mi)
        mpiutil.barrier()
        return
"""

