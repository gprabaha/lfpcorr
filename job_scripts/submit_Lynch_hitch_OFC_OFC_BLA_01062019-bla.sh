#!/bin/bash
#SBATCH --job-name=Lynch_hitch_OFC_OFC_BLA_01062019-bla_rfft_32G
#SPATCH --partition=psych_day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=1:00:00
#SBATCH --output=job_scripts/rfft_32G_Lynch_hitch_OFC_OFC_BLA_01062019-bla.out
#SBATCH --error=job_scripts/rfft_32G_Lynch_hitch_OFC_OFC_BLA_01062019-bla.err

# Load necessary modules
module load miniconda
conda activate lfp_cluster
# Your commands to execute

python windowed_fft_one_file.py /gpfs/milgram/project/chang/pg496/data_dir/social_gaze/social_gaze_raw_mat/Lynch_hitch_OFC_OFC_BLA_01062019-bla.mat
