module load 2023r1
module load python
module load py-pip
echo "INstalling evalplus"
pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"


#export HF_TOKEN=sample_token
#echo login to huggingface using 
# Command: huggingface-cli login

echo "Running codegen"
evalplus.codegen --model "mistralai/Mistral-7B-Instruct-v0.3" --dataset humaneval --backend hf --greedy
