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

# Fetch KL-basis visibilities
data_path ='/data/zzhang/draco_out/klmode_group_0.h5'
fdata = h5py.File(data_path,'r')
vis=fdata['vis'][...]
fdata.close()

# In the step of minimization, one will have to choose whether to use Newton methods or not.
# Newtons methods make use of both Jocobian and Hessian, which is conceptually faster than 
# methods using Jacobian only. However, if then number of parameters is so large that it costs
# a lot to compute and inverse Hessian matrics, one should choose to use non-Newton methods, 
# e.g., 'L-BFGS' method.

with_Hessian = True

if with_Hessian:
    test = Likelihood_with_hess(vis, CV)
    Opt_Method = 'Newton-CG'
    def Hessian(pvec):
        test(pvec)
        return test.hess
else:
    test = Likelihood(vis, CV)
    Opt_Method = 'L-BFGS-B'
    Hessian = None

def log_likelihood(pvec):
        test(pvec)
        return test.fun
    
def Jacobian(pvec):
        test(pvec)
        return test.jac

# Calculate theoretical predictions:
p_th = test.parameter_model_values
# Give first guess for optimisation:
p0 = p_th
    

st = time.time()
res = minimize(log_likelihood, p0, method=Opt_Method, jac= Jacobian, hess=Hessian) # rex.x is the result.
et = time.time()

if mpiutil.rank0:
    print("***** Time elapsed for the minimization: %f *****" % (et - st))
    Aux1, Aux2 = N.broadcast_arrays(CV.k_par_centers[:, N.newaxis], CV.k_perp_centers)
    with h5py.File("MLPSE"+Opt_Method+str(len(res.x))+".hdf5", "w") as f:
        f.create_dataset("first guess", data=p0)
        f.create_dataset("theory", data=p_th)
        f.create_dataset("log likelihood",data=test.fun)
        f.create_dataset("power spectrum", data=res.x)
        f.create_dataset("k parallel", data=Aux1.flatten())
        f.create_dataset("k perp", data=Aux2.flatten())
        f.create_dataset("k centers", data=CV.k_centers)
        

