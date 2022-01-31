# MLpse

MLpse is a package designed for the Maximum Likelihood Estimation of Power Spectrum of the transit radio data with the m-mode formalism.

To do this, it depends on various related packages including:
- [driftscan](https://github.com/hirax-array/driftscan): for modelling the telescope and generating the products, for example, SVD and KL 
transformation matrices, required for simulation and analysis.
- [draco](https://github.com/hirax-array/draco/tree/master/draco): for simulating the time stream data from maps of the sky and transform 
it to m-mode data
  - [cora](https://github.com/hirax-array/cora): for modelling and simulating the radio sky
  - [driftscan](https://github.com/hirax-array/driftscan): as introduced above.
- [caput](https://github.com/hirax-array/caput): provides infrastructure for building these packages, especially MPI utilities.

## Descriptions of the package:
MLpse consists of three files:
1. "maximum_likelihood.py": This is the major script. It calls modules defined in the other two files. Run this to get an ".hdf5" file which records 
the values of power spectrum parameters and corresponding k bands.
2. "MLpse.py": core source code with 3 classes:
  - *kspace_cartesian*: for parameterizing and binning the k space.
  
  - *Covariances*: for dealing with the covariance matrices and projecting them to KL basis.
  
  - *Likelihood*: for calculating the per-mmode log-likelihood function and the associated Jacobians and Hessian matrics.
  
3. "Fetch_info.py": this is the module file adapted from "[drfitscan/core/manager.py](https://github.com/hirax-array/driftscan/blob/master/drift/core/manager.py)" for reading and translating the parameter file used in driftscan

## Typical workflow of MLpse with simulated data using the m-mode analysis pipeline:
1. Model the telescope and generate Beam transfer matrics, SVD and Karhunen-Loève (KL) transform matrices, etc. The settings 
of the telescope and SVD/KL filters are specified in the parameter file. Run the parameter file with command:
```
drift-makeproducts run product_params.yaml
```
or if you run with mpi
``` 
srun --mpi=pmi2 drift-makeproducts run /home/zzhang/parameter_files/prod_params.yaml
```

2. Model and simulate the radio sky, both the cosmological signal and the foregrounds. Examples are
```
cora-makesky --pol full --nside 512 --freq-mode edge --freq 600 650 16 --filename cora_sim_21cm_nside_512.h5 21cm
cora-makesky --pol full --nside 512 --freq-mode edge --freq 600 650 16 --filename cora_sim_galaxy_nside_512.h5 galaxy
```

3. Simulate a sidereal visibility stream from the products we generated. And generate the m-mode data out of it.
```
caput-pipeline run config_draco.yaml
```

4. Modify "maximum_likelihood.py" to acheive the maximum likelihood power spectrum using simulated data.
```
python maximum_likelihood.py
```





