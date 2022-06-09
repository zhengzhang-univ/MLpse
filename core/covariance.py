import numpy as N
import h5py
from drift.core import skymodel
from core import mpiutil
from core.kspace import kspace_cartesian
from mpi4py import MPI
from core.mpiutil import myTiming
    
    
class Covariances(kspace_cartesian):
    def make_foregrounds_covariance_sky(self):
        """  Construct foregrounds covariance in sky basis
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
        """
        The noise includes both foregrounds and instrumental noise.
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


class Covariance_saveKL(Covariances):
    def __call__(self, filepath, saveKL=True):
        self.filter_m_modes() # Filter out trivial mmodes on KL basis
        self.filesavepath = filepath
        self.make_response_matrix()
        mpiutil.barrier()

    def make_response_matrix(self, saveKL=True):
        local_params = []
        local_k_pars_used = []
        local_k_perps_used = []
        local_k_centers_used = []
        local_Resp_mat_list = []
        for i in mpiutil.partition_list_mpi(list(range(self.alpha_dim))):
            aux_array = self.make_response_matrix_sky(i)
            if not N.all(aux_array==0):
                local_params.append(i)
                local_k_pars_used.append(self.k_pars[i])
                local_k_perps_used.append(self.k_perps[i])
                local_k_centers_used.append(self.k_centers[i])
                local_Resp_mat_list.append(aux_array)

        local_size = N.array(len(local_params)).astype(N.int32)
        sendcounts = N.zeros(mpiutil.size, dtype=N.int32)
        displacements = N.zeros(mpiutil.size, dtype=N.int32)
        mpiutil._comm.Allgather([local_size, MPI.INT], [sendcounts, MPI.INT])
        self.nonzero_alpha_dim = N.sum(sendcounts)
        displacements[1:] = N.cumsum(sendcounts)[:-1]

        self.k_pars_used = N.empty(self.nonzero_alpha_dim)
        self.k_perps_used = N.empty(self.nonzero_alpha_dim)
        self.k_centers_used = N.empty(self.nonzero_alpha_dim)
        self.para_ind_list = N.zeros(self.nonzero_alpha_dim, dtype=N.int32)

        mpiutil._comm.Allgatherv([N.array(local_params).astype(N.int32), MPI.INT],
                                 [self.para_ind_list, sendcounts, displacements, MPI.INT])
        mpiutil._comm.Allgatherv([N.array(local_k_pars_used).astype(float), MPI.DOUBLE],
                                 [self.k_pars_used, sendcounts, displacements, MPI.DOUBLE])
        mpiutil._comm.Allgatherv([N.array(local_k_perps_used).astype(float), MPI.DOUBLE],
                                 [self.k_perps_used, sendcounts, displacements, MPI.DOUBLE])
        mpiutil._comm.Allgatherv([N.array(local_k_centers_used).astype(float), MPI.DOUBLE],
                                 [self.k_centers_used, sendcounts, displacements, MPI.DOUBLE])
        if saveKL:
            self.save_response_matrix_KL(local_Resp_mat_list)
        else:
            return

    @myTiming
    def save_response_matrix_KL(self, local_Resp_mat_list):
        local_params = mpiutil.partition_list_mpi(self.para_ind_list)
        for i in range(len(local_params)):
            f = h5py.File(self.filesavepath + str(local_params[i])+'.hdf5', 'w')
            for mi in self.nontrivial_mmode_list:
                if mpiutil.rank0:
                    print(mi)
                f.create_dataset(str(mi), data=self.project_Q_sky_to_kl(mi, local_Resp_mat_list[i]))
            f.close()
        mpiutil.barrier()
        return

    def load_Q_kl_mi_param(self,mi,param_ind):
        return h5py.File(self.filesavepath + str(param_ind) + ".hdf5",'r')[str(mi)][...]

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

