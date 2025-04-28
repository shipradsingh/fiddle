#!/bin/bash

#SBATCH --account=ucb633_asc1
#SBATCH --partition=amilan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    # Reduced from 64 to match Amilan specs
#SBATCH --mem=128G            # Adjusted memory request
#SBATCH --time=24:00:00
#SBATCH --job-name=fiddle_cpu
#SBATCH --error=fiddle_cpu_%j.err
#SBATCH --output=fiddle_cpu_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shipra.singh@colorado.edu

# Set environment variables for CPU optimization
export MKL_NUM_THREADS=32      # Match cpus-per-task
export OMP_NUM_THREADS=32      # Match cpus-per-task
export NUMEXPR_NUM_THREADS=32  # Match cpus-per-task
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Set matplotlib config directory
export MPLCONFIGDIR=/scratch/alpine/shsi2591/matplotlib_cache
mkdir -p /scratch/alpine/shsi2591/matplotlib_cache

# Change to scratch directory
cd /scratch/alpine/shsi2591/fiddle

# Conda setup
__conda_setup="$('/projects/shsi2591/software/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate fiddle

# Run your scripts
python src/train.py
python src/test.py
