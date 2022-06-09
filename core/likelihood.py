import copy
import numpy as N
import scipy.linalg
import h5py
from core import mpiutil
from core.mpiutil import myTiming


class Likelihood:
    @myTiming
    def __init__(self, data_path, covariance_class_obj, threshold = None):
        self.pvec = None
        self.threshold = threshold
        self.CV = covariance_class_obj
        self.dim = self.CV.nonzero_alpha_dim
        self.nontrivial_mmode_list = self.CV.nontrivial_mmode_list
        self.local_ms = mpiutil.partition_list_mpi(self.nontrivial_mmode_list, method="alt")
        self.mmode_count = len(self.nontrivial_mmode_list)
        parameters = self.CV.make_binning_power()
        self.parameter_model_values = [parameters[i] for i in self.CV.para_ind_list]
        self.local_cv_noise_kl = []
        self.local_data_kl_m = []
        fdata = h5py.File(data_path, 'r')
        for mi in self.local_ms:
            cvnoise = self.CV.make_noise_covariance_kl_m(mi, threshold)
            length = cvnoise.shape[0]
            self.local_cv_noise_kl.append(cvnoise)
            self.local_data_kl_m.append(N.matrix(fdata['vis'][mi][:length].reshape((-1, 1))))
        fdata.close()
        if mpiutil.rank0:
            print(" The likelihood class object has been initialized!")
        mpiutil.barrier()

    @myTiming
    def __call__(self, pvec):
        if self.pvec is not pvec:
            self.pvec = pvec
            if len(self.local_ms) is not 0:
                Result = [self.make_funs_and_jacs_mi(mi) for mi in self.local_ms]
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

    @myTiming
    def make_funs_and_jacs_mi(self, mi):
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
        return fun_mi.real, jac_mi.real

    @myTiming
    def make_covariance_kl_m(self, pvec, mi):
        cv_mat = copy.deepcopy(self.local_cv_noise_kl[self.local_ms.index(mi)])
        for i in range(self.dim):
            cv_mat += pvec[i]*self.CV.load_Q_kl_mi_param(mi, self.CV.para_ind_list[i])
        return cv_mat

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