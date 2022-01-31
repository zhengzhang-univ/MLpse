from Fetch_info import Parameters_collection
from MLpse import *
from scipy.optimize import minimize
import numpy as N
from caput import mpiutil
import time
import h5py

# Path to the parameter file.
configfile = "/data/zzhang/sim1/bt_matrices/config.yaml" 

# Fetch info about the telescope, SVD, KL filters, parameters of observation, etc.
# This gives a class object.
pipeline_info = Parameters_collection.from_config(configfile) 


CV = Covariances(0,0.3,2,0,0.15,2,pipeline_info['dk_1thresh_fg_3thresh'])

# Here the instance of the data array is in the sky basis. It will be changed into KL basis in the program.
# More realistic data, as the input of the program, should be the m-mode visibilities.

fakedata_sky = 100.* N.ones(shape=(16,4,201)) # shape: (frequency, polarisation, lmax+1)

test = Likelihood(fakedata_sky, CV)

# Use the calculations with theoretical models as the first guess of the minimization.
p0 = test.parameter_firstguess_list
    
def log_likelihood(pvec):
        test(pvec)
        return test.fun
    
def Jacobian(pvec):
        test(pvec)
        return test.jac
    
def Hessian(pvec):
        test(pvec)
        return test.hess

st = time.time()
res = minimize(log_likelihood, p0, method='Newton-CG', jac= Jacobian, hess=Hessian)
# rex.x is the result.
et = time.time()

if mpiutil.rank0:
    print("***** Time elapsed for the minimization: %f *****" % (et - st))
    with h5py.File("MLPSE.hdf5", "w") as f:
        f.create_dataset("power spectrum", data=res.x)
        f.create_dataset("k centers", data=CV.k_centers)
        

