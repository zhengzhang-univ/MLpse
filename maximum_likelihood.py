from Fetch_info import Parameters_collection
from MLpse import *
from scipy.optimize import minimize


configfile = "/data/zzhang/mpi_test/config.yaml"

pipeline_info = Parameters_collection.from_config(configfile) # Fetch info about the telescope and former steps


CV = Covariances(0,0.3,2,0,0.15,2,pipeline_info['dk_0thresh_fg_3thresh'])


test = Likelihood(data, threshold, CV)
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
    
res = minimize(log_likelihood, p0, jac= Jacobian, hess=Hessian)

# rex.x is the result.