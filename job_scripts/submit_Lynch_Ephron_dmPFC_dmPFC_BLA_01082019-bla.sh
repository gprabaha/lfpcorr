#!/bin/bash
#SBATCH --job-name=Lynch_Ephron_dmPFC_dmPFC_BLA_01082019-bla_rfft_week
#SPATCH --partition=psych_week
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=job_scripts/rfft_week_Lynch_Ephron_dmPFC_dmPFC_BLA_01082019-bla.out
#SBATCH --error=job_scripts/rfft_week_Lynch_Ephron_dmPFC_dmPFC_BLA_01082019-bla.err

# Load necessary modules
module load miniconda
conda activate lfp_cluster
# Your commands to execute

python windowed_fft_one_file.py /gpfs/milgram/project/chang/pg496/data_dir/social_gaze/social_gaze_raw_mat/Lynch_Ephron_dmPFC_dmPFC_BLA_01082019-bla.mat
