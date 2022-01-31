from Fetch_info import Parameters_collection
from MLpse import *
from scipy.optimize import minimize
import numpy as N
from caput import mpiutil
import time

configfile = "/data/zzhang/sim1/bt_matrices/config.yaml"

pipeline_info = Parameters_collection.from_config(configfile) # Fetch info about the telescope and former steps


CV = Covariances(0,0.3,2,0,0.15,2,pipeline_info['dk_1thresh_fg_3thresh'])

fakedata_sky = 100.* N.ones(shape=(16,4,201)) # shape: (frequency, polarisation, lmax+1)

test = Likelihood(fakedata_sky, CV)

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
    print("***** likelihood maximization  time: %f" % (et - st))
    with h5py.File("MLPSE.hdf5", "w") as f:
        f.create_dataset("power spectrum", data=res.x)
        f.create_dataset("k centers", data=CV.k_centers)
        

