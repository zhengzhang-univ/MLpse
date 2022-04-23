# MLPSE

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
MLpse consists of the following files:
0. "MLPSE_documentation".
1. "maximum_likelihood.py": This is the major script. It calls modules defined in the other two files. Run this to get an ".hdf5" file which records 
the values of power spectrum parameters and corresponding k bands.
2. "MLpse.py": core source code with several classes:
3. "Fetch_info.py": this is the module file adapted from "[drfitscan/core/manager.py](https://github.com/hirax-array/driftscan/blob/master/drift/core/manager.py)" for reading and translating the parameter file used in driftscan

## Typical workflow of MLpse with simulated data using the m-mode analysis pipeline:
1. Model the telescope and generate Beam transfer matrics, SVD and Karhunen-Loève (KL) transform matrices, etc. The settings 
of the telescope and SVD/KL filters are specified in the parameter file. 

2. Model and simulate the radio sky, both the cosmological signal and the foregrounds. 

3. Simulate a sidereal visibility stream, and generate the visibilities in KL basis, from the products generated above. 
An example for the parameter file (not necessarily correct!):
```
...
...
...
        -   type:       draco.analysis.fgfilter.KLModeProject
            requires:   pm
            in:         svdmodes
            out:        klmodes
            params:
                klname:    dk_1thresh_fg_100thresh
                save:   Yes
                output_root: '/data/zzhang/draco_out/klmode_'
```

4. Run "maximum_likelihood.py" to acheive the maximum likelihood power spectrum using simulated data.

An example for the sbatch file:
```
#!/bin/sh
  
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=10 # number of MPI processes
#SBATCH --cpus-per-task=2 # number of OpenMP processes
#SBATCH --mem=64000 # memory per node
#SBATCH --time=48:00:00
#SBATCH --job-name=mustwork


source activate hirax_sims

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/zzhang/MLpse
srun --mpi=pmi2 python maximum_likelihood.py
```





