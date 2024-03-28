#!/bin/bash
#SBATCH --job-name=rfft_kaiser_sg_Kuro_Lynch_BLA_OFC_08292018-ofc
#SPATCH --partition=psych_day
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --time=1:00:00
#SBATCH --output=job_scripts/rfft_kaiser_sg_Kuro_Lynch_BLA_OFC_08292018-ofc.out  
#SBATCH --error=job_scripts/rfft_kaiser_sg_Kuro_Lynch_BLA_OFC_08292018-ofc.err          

# Load necessary modules
module load miniconda
conda activate lfp_cluster

# Your commands to execute
python windowed_fft_one_file.py /gpfs/milgram/project/chang/pg496/social_gaze_raw_mat/Kuro_Lynch_BLA_OFC_08292018-ofc.mat