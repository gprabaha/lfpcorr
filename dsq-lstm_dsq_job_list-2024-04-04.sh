#!/bin/bash
#SBATCH --output dsq-lstm_dsq_job_list-%A_%3a-%N.out
#SBATCH --array 0-665
#SBATCH --job-name dsq-lstm_dsq_job_list
#SBATCH --partition psych_gpu --mem-per-cpu=20g --cpus-per-task=1 --gpus=1 -t 2:00:00

# DO NOT EDIT LINE BELOW
/gpfs/milgram/apps/hpc.rhel7/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/milgram/pi/chang/pg496/repositories/lfpcorr/lstm_dsq_job_list.txt --status-dir /gpfs/milgram/pi/chang/pg496/repositories/lfpcorr

