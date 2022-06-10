from util.Fetch_info import Parameters_collection
from core.covariance import Covariance_saveKL
from core.likelihood import Likelihood
from util.util import *
from scipy.optimize import minimize
import numpy as N
from util import mpiutil
import time
import h5py
import copy

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

@regularized_scalar(Regularized)
@scaled_scalar(Scaling)
def log_likelihood(pvec):
    return test.log_likelihood_func(pvec)

@regularized_vector(Regularized)
@scaled_vector(Scaling)
def Jacobian(pvec):
    return test.jacobian(pvec)

if Scaling:
    p0 = N.log(2 * p0)


st = time.time()
res = minimize(log_likelihood, p0, method='BFGS', jac= Jacobian, tol=1e-3,
               options={'gtol': 1e-4, 'disp': True, 'maxiter':200, 'return_all':True}) # rex.x is the result.
et = time.time()



if mpiutil.rank0:
    if Scaling:
        result = (N.exp(res.x) + N.exp(-res.x))*.5 - 1
    else:
        result = res.x
    print("***** Time elapsed for the minimization: %f ***** \n" % (et - st))
    print("Succeed or not? {}\n".format(res.success))
    print("PS results: {}".format(result))
    print("{}\n".format(res.message))
    print("Number of iteration {}\n".format(res.nit))
    with h5py.File(outputname+".hdf5", "w") as f:
        f.create_dataset("first guess", data=p0)
        f.create_dataset("theory", data=test.parameter_model_values)
        f.create_dataset("ps_result", data=result)
        f.create_dataset("k_parallel", data=test.CV.k_pars_used)
        f.create_dataset("k_perp", data=test.CV.k_perps_used)
        f.create_dataset("k", data=test.CV.k_centers_used)
        #f.create_dataset("Hess_at_solution", data=)
        #f.create_dataset("nfev",data=res.nfev)
        #f.create_dataset("njev",data=res.njev)
        #f.create_dataset("status",data=res.status)
        #f.create_dataset("fun",data=res.fun)
        #f.create_dataset("jac",data=res.jac)
