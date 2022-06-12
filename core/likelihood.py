import copy
import numpy as N
import scipy.linalg
import h5py
from util import mpiutil
from util.util import myTiming_rank0, cache_last_n_classfunc
from core.covariance import fetch_triu, build_Hermitian_from_triu


class Likelihood:

    @myTiming_rank0
    def __init__(self, data_path, covariance_class_obj, threshold = None):
        self.threshold = threshold
        self.CV = covariance_class_obj
        self.dim = self.CV.nonzero_alpha_dim
        self.nontrivial_mmode_list = self.CV.nontrivial_mmode_list
        self.partition_modes_m()
        self.mmode_count = len(self.nontrivial_mmode_list)
        parameters = self.CV.make_binning_power()
        #global memorysize
        self.memorysize = 2 * len(self.local_ms)
        self.parameter_model_values = N.array([parameters[i] for i in self.CV.para_ind_list])
        self.pvec = N.zeros_like(self.parameter_model_values)
        self.local_cv_noise_kl = []
        self.local_data_kl_m = []
        self.local_Q_triu_kl_m = []
        fdata = h5py.File(data_path, 'r')
        for mi in self.local_ms:
            cvnoise = self.CV.make_noise_covariance_kl_m(mi, threshold)
            length = cvnoise.shape[0]
            self.local_cv_noise_kl.append(cvnoise)
            self.local_data_kl_m.append(fdata['vis'][mi][:length])
            self.local_Q_triu_kl_m.append(self.CV.load_Q_kl_mi_triu(mi))
        fdata.close()
        if mpiutil.rank0:
            print(" The likelihood class object has been initialized!")
        mpiutil.barrier()

    def __call__(self, pvec):
        if not N.allclose(self.pvec, pvec):
            self.pvec = pvec
            if len(self.local_ms) is not 0:
                Result = [self.make_fun_and_jac_mi(mi) for mi in self.local_ms]
                auxf, auxj = list(zip(*Result))
                send_f = N.array([sum(auxf)])
                send_j = sum(auxj)
            else:
                send_f = N.array([0.])
                send_j = N.zeros((self.dim,))
            recv_f = N.array([0.])
            recv_j = N.zeros((self.dim,))
            mpiutil._comm.Allreduce(send_f, recv_f)
            mpiutil._comm.Allreduce(send_j, recv_j)
            self.fun = recv_f[0] / self.mmode_count
            self.jac = recv_j / self.mmode_count
        else:
            return


    def partition_modes_m(self):
        size = mpiutil.size
        mlen = len(self.nontrivial_mmode_list)
        if size < mlen:
            self.local_ms = []
            nbatch = int(N.floor(mlen/size))
            sorted_indices = N.argsort(N.array(self.CV.kl_len))[::-1]  # from max to min
            for i in range(nbatch):
                if i % 2 == 0:
                    aux = sorted_indices[size*i:size*(i+1)]
                else:
                    aux = sorted_indices[size*i:size*(i+1)][::-1]
                m = self.nontrivial_mmode_list[aux[mpiutil.rank]]
                self.local_ms.append(m)
            if nbatch % 2 == 0:
                if mpiutil.rank < (mlen-nbatch*size):
                    aux = sorted_indices[size*nbatch:]
                    m = self.nontrivial_mmode_list[aux[mpiutil.rank]]
                    self.local_ms.append(m)
            else:
                if (size - mpiutil.rank - 1) < (mlen - nbatch * size):
                    aux = sorted_indices[size * nbatch:]
                    m = self.nontrivial_mmode_list[aux[size - mpiutil.rank - 1]]
                    self.local_ms.append(m)
        else:
            self.local_ms = mpiutil.partition_list_mpi(self.nontrivial_mmode_list)
        return

    @myTiming_rank0
    def make_fun_and_jac_mi(self, mi):
        C = self.make_covariance_kl_m(self.pvec, mi)
        local_mindex = self.local_ms.index(mi)
        C_inv = scipy.linalg.inv(C)
        C_inv_D = C_inv @ self.local_data_kl_m[local_mindex] @ self.local_data_kl_m[local_mindex].H
        # compute m-mode log-likelihood
        fun_mi = N.linalg.slogdet(C)[1] + N.trace(C_inv_D)
        # compute m-mode Jacobian
        aux = (N.identity(C.shape[0]) - C_inv_D) @ C_inv
        jac_mi = N.array([N.trace(self.CV.load_Q_kl_mi_param(mi, self.CV.para_ind_list[i]) @ aux)
                          for i in range(self.dim)]).reshape((self.dim,))
        print("**make_fun_and_jac_mi: mi = {}, KL length = {}".format(mi, self.CV.kl_len[self.nontrivial_mmode_list.index(mi)]))
        return fun_mi.real, jac_mi.real

    @myTiming_rank0
    def make_covariance_kl_m(self, pvec, mi):
        cv_mat = copy.deepcopy(self.local_cv_noise_kl[self.local_ms.index(mi)])
        for i in range(self.dim):
            cv_mat += pvec[i]*self.CV.load_Q_kl_mi_param(mi, self.CV.para_ind_list[i])
        return cv_mat

    @myTiming_rank0
    def make_covariance_kl_m_in_memory(self, pvec, mi):
        m=self.local_ms.index(mi)
        cv_mat = self.local_cv_noise_kl[m] + \
                 build_Hermitian_from_triu(N.einsum("ij,j->i", self.local_Q_triu_kl_m[m], pvec))
        return cv_mat.astype(N.csingle)

    @myTiming_rank0
    def make_function_m(self, pvec, mi):
        local_mindex = self.local_ms.index(mi)
        C = self.make_covariance_kl_m_in_memory(pvec, mi)
        C_inv = scipy.linalg.inv(C).astype(N.csingle)
        C_inv_D = C_inv @ self.local_data_kl_m[local_mindex] @ self.local_data_kl_m[local_mindex].H
        result = N.linalg.slogdet(C)[1] + N.trace(C_inv_D)
        return result.real

    @myTiming_rank0
    def make_jacobian_m(self, pvec, mi):
        local_mindex = self.local_ms.index(mi)
        C = self.make_covariance_kl_m_in_memory(pvec, mi)
        C_inv = scipy.linalg.inv(C).astype(N.csingle)
        aux = C_inv @ self.local_data_kl_m[local_mindex]
        # aux = (N.identity(C.shape[0]) - C_inv_D) @ C_inv
        aux = C_inv - aux @ aux.conj().T
        aux = fetch_triu(aux.T) * 2
        print("aux shape: {}".format(aux.shape))
        size = C.shape[0]
        assert aux.shape[0] == int(size*(size+1)/2)
        #assert aux.shape[0] == self.local_Q_triu_kl_m[local_mindex].shape[0]
        count = 0
        for i in range(size):
            aux[count] *= 0.5
            count += size - i
        #result = N.sum(self.local_Q_triu_kl_m[local_mindex] * aux[:, N.newaxis], axis=0)
        result = N.einsum("ij, i -> j", self.local_Q_triu_kl_m[local_mindex], aux)
        return result.real.reshape((self.dim,))
        #def trace_product(x):
        #    return N.sum(build_Hermitian_from_triu(x) * aux.conj().T)
        #result = N.apply_along_axis(trace_product, axis=0, arr=self.local_Q_triu_kl_m[local_mindex])
        # result = N.array([N.trace(self.CV.load_Q_kl_mi_param(mi, self.CV.para_ind_list[i]) @ aux)
        #                   for i in range(self.dim)]).reshape((self.dim,))
        #return result.reshape((self.dim,)).real

    @myTiming_rank0
    def log_likelihood_func(self, pvec):
        if len(self.local_ms) is not 0:
            Result = [self.make_function_m(pvec, mi) for mi in self.local_ms]
            send_f = N.array([sum(Result)])
        else:
            send_f = N.array([0.])
        mpiutil.barrier()
        recv_f = N.array([0.])
        mpiutil._comm.Allreduce(send_f, recv_f)
        fun = recv_f[0] / self.mmode_count
        return fun

    @myTiming_rank0
    def jacobian(self, pvec):
        if len(self.local_ms) is not 0:
            Result = [self.make_jacobian_m(pvec, mi) for mi in self.local_ms]
            send_j = sum(Result)
        else:
            send_j = N.zeros((self.dim,))
        mpiutil.barrier()
        recv_j = N.zeros((self.dim,))
        mpiutil._comm.Allreduce(send_j, recv_j)
        jac = recv_j / self.mmode_count
        return jac


""" 
    def calculate_Errors(self):
        fun = mpiutil.parallel_map(self.Fisher_m, self.nontrivial_mmode_list, method="alt")
        return scipy.linalg.inv(sum(list(fun)))
    
    def Fisher_m(self, mi):
        local_mindex = self.local_ms.index(mi)
        Q_alpha_list = self.local_Q_alpha_m[local_mindex]
        C = self.CV.make_covariance_kl_m(self.parameter_model_values, mi, Q_alpha_list, self.threshold)
        # len = C.shape[0]
        C_inv = scipy.linalg.inv(C)
        hess_mi = N.empty((self.dim, self.dim), dtype='complex128')
        for i in range(self.dim):
            for j in range(i, self.dim):
                hess_mi[i, j] = hess_mi[j, i] = N.trace(C_inv @ Q_alpha_list[i] @ C_inv @ Q_alpha_list[j])
        return hess_mi.real
"""
    


"""
        # compute m-mode Hessian
        hess_mi = N.empty((self.dim, self.dim), dtype='complex128')
        aux = (C_inv_D - 0.5*Identity) @ C_inv
        for i in range(self.dim):
            for j in range(i, self.dim):
                hess_mi[i, j] = hess_mi[j, i] = 2. * N.trace(Q_alpha_list[i] @ C_inv @ Q_alpha_list[j] @ aux)

"""