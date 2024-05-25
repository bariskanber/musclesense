#!/bin/bash
#SBATCH --job-name=mmseg_ll     
#SBATCH --time=3-0            
#SBATCH --mem=64000           
#SBATCH --cpus-per-task=4   
#SBATCH --gpus=1 

echo "Running on $(hostname)"

cd /home/bkanber/musclesenseworkbench

./miniconda3/bin/python mmseg_ll.py -al calf -inputdir train "$@"

