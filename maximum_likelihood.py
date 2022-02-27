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


#avg = sum(p_th)/len(p_th)
#scaling_coef = p_th/avg



#with_Hessian = False

#if with_Hessian:
#    test = Likelihood_with_J_H(vis, CV)
#    Opt_Method = 'trust-exact'
#    def Hessian(pvec):
#        aux = [a*b for a,b in zip(pvec, scaling_coef)]
#        test(aux)
#        return test.hess
#else:
test = Likelihood_with_J_only(vis, CV)
p_th = test.parameter_model_values
Opt_Method = 'BFGS'
Hessian = None
Fisher = test.calculate_Errors()
scaling_coef = []
Fisher_factor = N.trace(Fisher)/Fisher.shape[0]
for i in range(Fisher.shape[0]):
    scaling_coef.append(Fisher[i,i]/Fisher_factor)
    
    
    

def log_likelihood(pvec):
    aux1 = N.exp(pvec) 
    aux = [a*b for a,b in zip(aux1, scaling_coef)]
    test(aux)
    return test.fun
    
def Jacobian(pvec):
    #aux = [a*b for a,b in zip(pvec, scaling_coef)]    
    aux1 = N.exp(pvec)
    aux = [a*b for a,b in zip(aux1, scaling_coef)]
    test(aux)
    #return test.jac
    result = [a*b for a, b in zip(test.jac,aux)]
    Res_scaling = [a*b for a, b in zip(result,scaling_coef)]
    return N.array(Res_scaling)


# Give first guess for optimisation:
p0 = [N.log(a/b) for a,b in zip(p_th, scaling_coef)]

st = time.time()
res = minimize(log_likelihood, p0, method=Opt_Method, jac= Jacobian, tol=1e-2,options={'gtol': 1e-3, 'disp': True, 'maxiter':200, 'return_all':True}) # rex.x is the result.
et = time.time()

print("x values: {}".format(res.x))


if mpiutil.rank0:
    print("***** Time elapsed for the minimization: %f ***** \n" % (et - st))
    print("Succeed or not? {}\n".format(res.success))
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
        #f.create_dataset("allvecs",data=res.allvecs)
        f.create_dataset("fun history",data=fun_history)
        f.create_dataset("jac history",data=jac_history)
        f.create_dataset("pvec f history",data=pvec_fun_history)
        f.create_dataset("pvec j history",data=pvec_jac_history)
