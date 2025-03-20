#!/bin/bash
#
#SBATCH --job-name="llmtest"
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem-per-gpu=4G
#SBATCH --account=Education-EEMCS-Courses-CS4570

module load 2024r1
module load openmpi
module load py-pip

cd /scratch/akuruvilla/team-13
source ../llmenv/bin/activate
echo "Hello World"
python llm.py  >  /scratch/akuruvilla/debug.txt