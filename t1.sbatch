#!/bin/bash
#
#SBATCH --job-name="llmtest"
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu-a100
#SBATCH --mem-per-gpu=32G
#SBATCH --account=Education-EEMCS-Courses-CS4570

module load 2023r1
module load python
module load py-pip
module load cuda/11.6

python -m pip install --user --upgrade pip
module unload py-pip

# Navigate to your working directory
cd /scratch/akuruvilla/team-13
#source /scratch/akuruvilla/llmenv/bin/activate

#pip install -r requirements.txt
python -m pip install --user --upgrade pip transformers datasets huggingface_hub protobuf torch torchvision torchaudio  --extra-index-url https://download.pytorch.org/whl/cu118

#pip freeze >> /scratch/akuruvilla/team-13/debug.txt


python llmdelft2.py >> /scratch/akuruvilla/team-13/debug.txt
