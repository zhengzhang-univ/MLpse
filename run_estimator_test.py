from util.Fetch_info import Parameters_collection
from core.covariance import Covariance_saveKL
from core.likelihood import Likelihood
from util.util import *
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

@myTiming_rank0
@regularized_scalar(Regularized)
@scaled_scalar(Scaling)
def log_likelihood(pvec):
    return test.log_likelihood_func(pvec)

@myTiming_rank0
@regularized_vector(Regularized)
@scaled_vector(Scaling)
def Jacobian(pvec):
    return test.jacobian(pvec)

if Scaling:
    p0 = N.log(2 * p0)

Jacobian(p0)