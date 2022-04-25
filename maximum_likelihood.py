from Fetch_info import Parameters_collection
from MLpse import *
from scipy.optimize import minimize
import numpy as N
from caput import mpiutil
import time
import h5py
from numpy import linalg as LA
import copy

# Inputs:
configfile = "/data/zzhang/sim1/bt_matrices/config.yaml" 
data_path ='/data/zzhang/draco_out/klmode_group_0.h5'
kpar_start, kpar_end, kpar_dim, kperp_start, kperp_end, kperp_dim = 0,0.3,2,0,0.15,2
kltrans_name = 'dk_1thresh_fg_3thresh'
Scaling = True
Regularized = True
outputname = "MLPSE_final_test_2"


# Fetch info about the telescope, SVD, KL filters, parameters of observation, etc.
# This gives a class object.
pipeline_info = Parameters_collection.from_config(configfile) 
CV = Covariances(kpar_start, kpar_end, kpar_dim, kperp_start, kperp_end, kperp_dim, pipeline_info[kltrans_name])

# Fetch KL-basis visibilities
fdata = h5py.File(data_path,'r')
vis=fdata['vis'][...]
fdata.close()


test = Likelihood_with_J_only(vis, CV)
    
p_th =copy.deepcopy( test.parameter_model_values)
p_th = N.array(p_th)
p0 = p_th

if Scaling:
    p0 = N.log(2*p_th)
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
        def log_likelihood(xvec):
            pvec = (N.exp(xvec) + N.exp(-xvec))*.5 - 1
            test(pvec)
            return test.fun + LA.norm(xvec)
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


st = time.time()
res = minimize(log_likelihood, p0, method='BFGS', jac= Jacobian, tol=1e-3, options={'gtol': 1e-4, 'disp': True, 'maxiter':300, 'return_all':True}) # rex.x is the result.
et = time.time()



if mpiutil.rank0:
    if Scaling:
        result =(N.exp(res.x) + N.exp(-res.x))*.5 - 1
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
