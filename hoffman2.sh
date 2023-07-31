#!/bin/bash
## The number of CPU cores as needed:
#$ -pe shared 1
#$ -cwd
## h_rt = run time limit
#$ -l gpu,V100,h_rt=5:00:00,h_data=4G
## specifies that the standard error stream of the job is merged into the standard output stream
#$ -j y
## sets the path to where the standard output stream of the job will be written
#$ -o $SCRATCH/joblogs/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -e $SCRATCH/joblogs/$JOB_NAME.$JOB_ID.$TASK_ID.err
#$ -S /bin/bash
## Email address to notify (it says I shouldn't change this)
#$ -M $USER@mail
## Notify when: b=begin, e=end, a=abort
#$ -m bea
. /u/local/Modules/default/init/modules.sh
module load cuda/11.8 
module load python/3.9.6

pip3 install torch

python3 LSTM/training.py
