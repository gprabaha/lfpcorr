#!/bin/bash
#SBATCH --job-name=Kuro_Hitch_dmPFC_BLA_OFC_10042018-dmpfc_rfft_32G
#SPATCH --partition=psych_day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=1:00:00
#SBATCH --output=job_scripts/rfft_32G_Kuro_Hitch_dmPFC_BLA_OFC_10042018-dmpfc.out
#SBATCH --error=job_scripts/rfft_32G_Kuro_Hitch_dmPFC_BLA_OFC_10042018-dmpfc.err

# Load necessary modules
module load miniconda
conda activate lfp_cluster
# Your commands to execute

python windowed_fft_one_file.py /gpfs/milgram/project/chang/pg496/data_dir/social_gaze/social_gaze_raw_mat/Kuro_Hitch_dmPFC_BLA_OFC_10042018-dmpfc.mat
