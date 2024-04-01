#!/bin/bash
#SBATCH --job-name=lfp_Kuro_Hitch_dmPFC_BLA_OFC_10042018-dmpfc
#SPATCH --partition=psych_day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=3:00:00
#SBATCH --output=job_scripts/lfp_Kuro_Hitch_dmPFC_BLA_OFC_10042018-dmpfc.out
#SBATCH --error=job_scripts/lfp_Kuro_Hitch_dmPFC_BLA_OFC_10042018-dmpfc.err

# Load necessary modules
module load miniconda
conda activate lfp_cluster
# Your commands to execute

python extract_lfp_one_file.py /gpfs/milgram/project/chang/pg496/data_dir/social_gaze/social_gaze_raw_mat/Kuro_Hitch_dmPFC_BLA_OFC_10042018-dmpfc.mat
