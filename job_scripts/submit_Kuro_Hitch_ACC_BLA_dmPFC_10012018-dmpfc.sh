#!/bin/bash
#SBATCH --job-name=Kuro_Hitch_ACC_BLA_dmPFC_10012018-dmpfc_rfft_week
#SPATCH --partition=psych_week
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=job_scripts/rfft_week_Kuro_Hitch_ACC_BLA_dmPFC_10012018-dmpfc.out
#SBATCH --error=job_scripts/rfft_week_Kuro_Hitch_ACC_BLA_dmPFC_10012018-dmpfc.err

# Load necessary modules
module load miniconda
conda activate lfp_cluster
# Your commands to execute

python windowed_fft_one_file.py /gpfs/milgram/project/chang/pg496/data_dir/social_gaze/social_gaze_raw_mat/Kuro_Hitch_ACC_BLA_dmPFC_10012018-dmpfc.mat
