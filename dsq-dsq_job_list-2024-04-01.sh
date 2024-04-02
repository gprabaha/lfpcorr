#!/bin/bash
#SBATCH --output dsq-dsq_job_list-%A_%2a-%N.out
#SBATCH --array 0-65
#SBATCH --job-name dsq-dsq_job_list
#SBATCH --partition psych_day --mem=100g --cpus-per-task=8 -t 2:00:00

# DO NOT EDIT LINE BELOW
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/milgram/pi/chang/pg496/repositories/lfpcorr/job_scripts/dsq_job_list.txt --status-dir /gpfs/milgram/pi/chang/pg496/repositories/lfpcorr

