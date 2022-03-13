from Fetch_info import Parameters_collection
from MLpse import *
from scipy.optimize import minimize
import numpy as N
from caput import mpiutil
import time
import h5py
from numpy import linalg as LA

# Path to the parameter file.
configfile = "/data/zzhang/sim1/bt_matrices/config.yaml" 

# Fetch info about the telescope, SVD, KL filters, parameters of observation, etc.
# This gives a class object.
pipeline_info = Parameters_collection.from_config(configfile) 


CV = Covariances(0,0.3,3,0,0.15,6,pipeline_info['dk_1thresh_fg_3thresh'])

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


#avg = sum(p_th)/len(p_th)
#scaling_coef = p_th/avg






Scaling = True
NewtonMethods = False
Regularized = True

if NewtonMethods:
    test = Likelihood_with_J_H(vis, CV)
else:
    test = Likelihood_with_J_only(vis, CV)
    
p_th = test.parameter_model_values
p_th = N.array(p_th)

Hessian = None

if Scaling:
    if not Regularized:
        def log_likelihood(xvec):
            # xvec should be N.array object.
            #pvec = N.exp(xvec)*p_th
            pvec = N.exp(xvec)
            test(pvec)
            return test.fun    
        def Jacobian(xvec):
            #pvec = N.exp(xvec)*p_th
            pvec = N.exp(xvec)
            test(pvec)
            result = test.jac*pvec
            return result
        if NewtonMethods:
            def Hessian(xvec):
                pvec = N.exp(xvec)
                test(pvec)
                return N.diag(pvec**2)@test.hess
    else:
        def log_likelihood(xvec):
            pvec = N.exp(xvec)
            test(pvec)
            return test.fun + LA.norm(xvec)
        def Jacobian(xvec):
            pvec = N.exp(xvec)
            test(pvec)
            return test.jac*pvec + xvec/LA.norm(xvec)
        if NewtonMethods:
            def hess_of_norm(pvec):
                norm = LA.norm(pvec)
                return N.identity(len(pvec))/norm - N.outer(pvec,pvec)/(norm**3.)
            def Hessian(xvec):
                pvec = N.exp(xvec)
                test(pvec)
                result = N.diag(pvec**2)@test.hess + hess_of_norm(xvec)
                return result
elif not Regularized:
    def log_likelihood(pvec):
        test(pvec)
        return test.fun
    def Jacobian(pvec):
        test(pvec)
        return test.jac
    if NewtonMethods:
        def Hessian(pvec):
            test(pvec)
            return test.hess
else:
    def log_likelihood(pvec):
        test(pvec)
        return test.fun + LA.norm(pvec)
    def Jacobian(pvec):
        test(pvec)
        return test.jac + pvec/LA.norm(pvec)
    if NewtonMethods:
        def hess_of_norm(pvec):
            norm = LA.norm(pvec)
            return N.identity(len(pvec))/norm - N.outer(pvec,pvec)/(norm**3.)
        def Hessian(pvec):
            test(pvec)
            result = test.hess + hess_of_norm(pvec)
            return result

# Give first guess for optimisation:
# N.zeros(test.dim)
p0 = N.log(p_th)
#p0 = p_th
Opt_Method = 'BFGS'

st = time.time()
res = minimize(log_likelihood, p0, method=Opt_Method, jac= Jacobian, tol=1e-3, options={'gtol': 1e-4, 'disp': True, 'maxiter':300, 'return_all':True}) # rex.x is the result.
et = time.time()


#print("x values: {}".format(res.x))


if mpiutil.rank0:
    print("***** Time elapsed for the minimization: %f ***** \n" % (et - st))
    print("Succeed or not? {}\n".format(res.success))
    print("x values: {}".format(res.x))
    print("{}\n".format(res.message))
    print("Number of iteration {}\n".format(res.nit))

    Aux1, Aux2 = N.broadcast_arrays(CV.k_par_centers[:, N.newaxis], CV.k_perp_centers)
    with h5py.File("MLPSE_"+Opt_Method+str(len(res.x))+".hdf5", "w") as f:
        f.create_dataset("first guess", data=p0)
        f.create_dataset("theory", data=p_th)
        f.create_dataset("paramters", data=res.x)
        f.create_dataset("k parallel", data=Aux1.flatten())
        f.create_dataset("k perp", data=Aux2.flatten())
        f.create_dataset("k centers", data=CV.k_centers)
        #f.create_dataset("nfev",data=res.nfev)
        #f.create_dataset("njev",data=res.njev)
        #f.create_dataset("status",data=res.status)
        f.create_dataset("fun",data=res.fun)
        f.create_dataset("jac",data=res.jac)
