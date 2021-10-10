#!/bin/bash

#SBATCH --job-name=4_PSO_2018_NYC_20210616
#SBATCH --output=4_PSO_2018_NYC_20210616.out
#SBATCH --ntasks=21
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --mail-type=END
#SBATCH --mail-user=yilunx@uchicago.edu
#SBATCH --error=4_PSO_2018_NYC_20210616.err

# Load Python and MPI modules
module load python/anaconda-2019.03
module unload intelmpi
module load intelmpi/2018.2.199+intel-18.0

mpirun -n 40 python3 4_PSO_2018_NYC_20210616.py