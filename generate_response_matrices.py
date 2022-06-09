from core.Fetch_info import Parameters_collection
from core.covariance import *

# Inputs:
configfile = "/data/zzhang/Viraj/drift_prod_hirax_survey_49elem_7point_64bands/config.yaml"
data_path = '/data/zzhang/Viraj/draco_out/klmode_group_0.h5'
kpar_start, kpar_end, kpar_dim, kperp_start, kperp_end, kperp_dim = 0, 0.30, 31, 0, 0.10, 10
kltrans_name = 'dk_5thresh_fg_1000thresh'
Scaling = True
Regularized = True
outputname = "MLPSE_Viraj_test"
Response_matrices_filename = "data/zzhang/Viraj/ResponseMatricesKL.hdf5"


# Fetch info about the telescope, SVD, KL filters, parameters of observation, etc.
# This gives a class object.
pipeline_info = Parameters_collection.from_config(configfile)
CV = Covariance_saveKL(kpar_start, kpar_end, kpar_dim, kperp_start, kperp_end, kperp_dim,
                          pipeline_info[kltrans_name])
CV(Response_matrices_filename)



