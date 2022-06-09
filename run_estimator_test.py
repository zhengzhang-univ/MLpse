from util.Fetch_info import Parameters_collection
from core.covariance import Covariance_saveKL
from core.likelihood import Likelihood
from util.util import myTiming
from scipy.optimize import minimize
import numpy as N
from util import mpiutil
import time
import h5py
from numpy import linalg as LA
import copy
import functools

# Inputs:
configfile = "/data/zzhang/Viraj/drift_prod_hirax_survey_49elem_7point_64bands/config.yaml"
data_path ='/data/zzhang/Viraj/draco_out/klmode_group_0.h5'
kpar_start, kpar_end, kpar_dim, kperp_start, kperp_end, kperp_dim = 0, 0.30, 31, 0, 0.10, 10
kltrans_name = 'dk_5thresh_fg_1000thresh'
Scaling = True
Regularized = True
outputname = "MLPSE_Viraj_test"
Response_matrices_filename = "/data/zzhang/Viraj/tmp/"


# Fetch info about the telescope, SVD, KL filters, parameters of observation, etc.
pipeline_info = Parameters_collection.from_config(configfile)
CV = Covariance_saveKL(kpar_start, kpar_end, kpar_dim,
                       kperp_start, kperp_end, kperp_dim,
                       pipeline_info[kltrans_name])
CV(Response_matrices_filename, saveKL=False)

test = Likelihood(data_path, CV)
del CV, pipeline_info

p_th = copy.deepcopy(test.parameter_model_values)
p0 = p_th/2.

if Scaling:
    p0 = N.log(2*p0)
    if not Regularized:
        def log_likelihood(xvec):
            # xvec should be N.array object.
            pvec = (N.exp(xvec) + N.exp(-xvec))*.5 - 1.
            test(pvec)
            return test.fun    
        def Jacobian(xvec):
            pvec = (N.exp(xvec) + N.exp(-xvec))*.5 - 1.
            derpvec = (N.exp(xvec) - N.exp(-xvec))*.5
            test(pvec)
            result = test.jac*derpvec
            return result
    else:
        @myTiming
        @functools.lru_cache(maxsize=3)
        def log_likelihood(xvec):
            pvec = (N.exp(xvec) + N.exp(-xvec))*.5 - 1
            test(pvec)
            return test.fun + LA.norm(xvec)
        @myTiming
        @functools.lru_cache(maxsize=3)
        def Jacobian(xvec):
            pvec = (N.exp(xvec) + N.exp(-xvec))*.5 - 1
            derpvec = (N.exp(xvec) - N.exp(-xvec))*.5
            test(pvec)
            return test.jac*derpvec + xvec/LA.norm(xvec)
elif not Regularized:
    def log_likelihood(pvec):
        test(pvec)
        return test.fun
    def Jacobian(pvec):
        test(pvec)
        return test.jac
else:
    def log_likelihood(pvec):
        test(pvec)
        return test.fun + LA.norm(pvec)
    def Jacobian(pvec):
        test(pvec)
        return test.jac + pvec/LA.norm(pvec)

log_likelihood(p0)
Jacobian(p0)