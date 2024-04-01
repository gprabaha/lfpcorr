#!/bin/bash
#SBATCH --job-name=Kuro_Lynch_OFC_Stim_09162018-ofc_rfft_32G
#SPATCH --partition=psych_day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=1:00:00
#SBATCH --output=job_scripts/rfft_32G_Kuro_Lynch_OFC_Stim_09162018-ofc.out
#SBATCH --error=job_scripts/rfft_32G_Kuro_Lynch_OFC_Stim_09162018-ofc.err

# Load necessary modules
module load miniconda
conda activate lfp_cluster
# Your commands to execute

python windowed_fft_one_file.py /gpfs/milgram/project/chang/pg496/data_dir/social_gaze/social_gaze_raw_mat/Kuro_Lynch_OFC_Stim_09162018-ofc.mat
