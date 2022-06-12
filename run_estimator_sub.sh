#!/bin/sh

#SBATCH --nodes=20
#SBATCH --ntasks-per-node=5 # number of MPI processes
#SBATCH --cpus-per-task=4 # number of OpenMP processes
#SBATCH --mem=64000 # memory per node
#SBATCH --time=48:00:00
#SBATCH --job-name=please


source activate hirax_sims

#cd /data/zzhang/M_mode_Fisher 

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#srun --mpi=pmi2 drift-makeproducts run /home/zzhang/parameter_files/prod_params.yaml
#srun python script_a_try.py
#cd /data/zzhang/
#srun --mpi=pmi2 cora-makesky --pol full --nside 512 --freq-mode edge --freq 600 650 16 --filename cora_sim_21cm_nside_512.h5 21cm
#srun --mpi=pmi2 cora-makesky --pol full --nside 512 --freq-mode edge --freq 600 650 16 --filename cora_sim_galaxy_nside_512.h5 galaxy
cd /home/zzhang/MLpse
# srun --mpi=pmi2 python generate_response_matrices.py
srun --mpi=pmi2 python -u run_estimator.py > MLPSE.out
