from Fetch_info import Parameters_collection
from MLpse import *
from scipy.optimize import minimize
import numpy as N

configfile = "/data/zzhang/sim1/bt_matrices/config.yaml"

pipeline_info = Parameters_collection.from_config(configfile) # Fetch info about the telescope and former steps


CV = Covariances(0,0.3,2,0,0.15,2,pipeline_info['dk_1thresh_fg_3thresh'])

fakedata_sky = N.ones(shape=(16,4,201)) # shape: (frequency, polarisation, lmax+1)

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
    
res = minimize(log_likelihood, p0, method='Newton-CG', jac= Jacobian, hess=Hessian)

textfile = open("mlpse.txt", "w")
for element in list(res.x):
    textfile.write(element + "\n")
textfile.close()
# rex.x is the result.
