#!/bin/bash
#SBATCH --job-name=lfp_Lynch_Ephron_ACCg_BLA_ACCg_01152019-bla
#SPATCH --partition=psych_day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=3:00:00
#SBATCH --output=job_scripts/lfp_Lynch_Ephron_ACCg_BLA_ACCg_01152019-bla.out
#SBATCH --error=job_scripts/lfp_Lynch_Ephron_ACCg_BLA_ACCg_01152019-bla.err

# Load necessary modules
module load miniconda
conda activate lfp_cluster
# Your commands to execute

python extract_lfp_one_file.py /gpfs/milgram/project/chang/pg496/data_dir/social_gaze/social_gaze_raw_mat/Lynch_Ephron_ACCg_BLA_ACCg_01152019-bla.mat
