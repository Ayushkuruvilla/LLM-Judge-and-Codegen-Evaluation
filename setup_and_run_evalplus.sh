#!/bin/bash

# Exit on any error
set -e

echo "Updating package lists..."
apt update

echo "Installing required packages..."
apt install -y git python3 python3-pip python3-venv

echo "Creating symbolic link for python3 to python..."
ln -s /usr/bin/python3 /usr/bin/python || true  # Prevent error if symlink already exists

echo "Creating a Python virtual environment..."
python -m venv ~/venv
source ~/venv/bin/activate

echo "Upgrading pip and installing evalplus..."
pip install --upgrade pip
pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"

# Comment: Login to Hugging Face CLI to authenticate for accessing restricted models
#echo "Prompting for Hugging Face login..."
# Command: huggingface-cli login


# Export Hugging Face token (replace <your_huggingface_token> with your token or ensure the user logs in)
export HF_TOKEN=<your_huggingface_token>

echo "Running evalplus.codegen with vllm backend..."
evalplus.codegen --model "mistralai/Mistral-7B-Instruct-v0.3" --greedy --root /results/humaneval --dataset humaneval --backend vllm || {
  echo "vllm backend failed, attempting hf backend..."
  
  # Comment: Alternative command to use the hf backend instead of vllm
  # Command: evalplus.codegen --model "mistralai/Mistral-7B-Instruct-v0.3" --greedy --root /results/humaneval --dataset humaneval --backend hf

  evalplus.codegen --model "mistralai/Mistral-7B-Instruct-v0.3" --greedy --root /results/humaneval --dataset humaneval --backend hf
}

echo "Script completed successfully."
