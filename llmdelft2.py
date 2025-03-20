import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch


# =========================
# LLM Judge Pipeline
# =========================
class LLMJudgePipeline:
    def __init__(self, model_name="gpt2", rating_prompt=None):
        """
        Initialize the LLM Judge Pipeline.
        Uses a Hugging Face model as the default evaluator.
        """
        self.model_name = model_name
        self.device = self._set_device()
        self.llm = self._load_model()
        self.humanevalplus_dataset = self._load_humanevalplus()
        self.rating_prompt = rating_prompt or (
            "Please rate the Sample Solution from 1 to 10 based on correctness and clarity, "
            "and provide a brief explanation for the rating."
        )

    def _set_device(self):
        """Set the device to GPU if available, otherwise CPU."""
        if torch.cuda.is_available():
            print("✅ GPU is available. Using GPU:", torch.cuda.get_device_name(0))
            return 0  # Device ID for the first GPU
        else:
            print("❌ GPU is not available. Using CPU.")
            return -1  # Use CPU

    def _load_model(self):
        """Load the Hugging Face model on the GPU if available."""
        print(f"Loading Hugging Face model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Load the model pipeline with the correct device
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=self.device  # Use GPU or CPU based on availability
        )

    def _load_humanevalplus(self):
        """Load the HumanEval+ dataset using the 'test' split."""
        print("Loading the HumanEval+ dataset...")
        test_dataset = load_dataset("evalplus/humanevalplus", split="test")
        return test_dataset

    def get_task_data(self, task_id):
        """Retrieve prompt and canonical solution for a given task_id from HumanEval+."""
        for example in self.humanevalplus_dataset:
            if example["task_id"] == task_id:
                return example["prompt"], example["canonical_solution"]
        raise ValueError(f"No data found for task_id: {task_id}")

    def judge_solution(self, prompt, canonical_solution, sample_solution):
        """Evaluate the sample solution using the Hugging Face model."""
        # Create the evaluation prompt
        evaluation_prompt = (
            f"{self.rating_prompt}\n"
            f"Task: {prompt}\n"
            f"Generated solution: {sample_solution}\n"
            f"Please keep your response concise and strictly under 500 tokens."
        )

        # Generate the evaluation response
        response = self.llm(evaluation_prompt, max_length=2500, do_sample=True, temperature=0.3, top_k=50, top_p=0.9, repetition_penalty=1.2)
        generated_text =  response[0]["generated_text"]
        clean_response = generated_text.replace(evaluation_prompt, "").strip()
        return clean_response

    def process_jsonl(self, jsonl_file, output_file):
        """Process a sample.jsonl file, evaluate each sample solution, and append the result to the output file."""
        with open(jsonl_file, "r") as input_file, open(output_file, "a") as output_file:
            count = 0
            for line in input_file:
                print(f"Processing task #{count}")
                count += 1

                data = json.loads(line)
                task_id = data["task_id"]
                sample_solution = data["solution"]

                # Retrieve prompt and canonical solution from HumanEval+
                prompt, canonical_solution = self.get_task_data(task_id)

                # Evaluate
                evaluation = self.judge_solution(prompt, canonical_solution, sample_solution)

                result = {"task_id": task_id, "evaluation": evaluation}

                # Append the result to the output file
                output_file.write(json.dumps(result) + "\n")

                print(result)

    def save_results_to_jsonl(self, results, output_file):
        """Save the evaluation results to a .jsonl file."""
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")


# =========================
# Usage Example
# =========================
if __name__ == "__main__":
    # Initialize the pipeline with a Hugging Face model
    detailed_prompt = (
    """
    You will be given a coding task and an implementation. Your task is to attribute a score for each relevant characteristic of the implementation. The score must accurately and fairly represent the quality of the code.
    Points in the score are awarded per category. The total score equals the sum of the weighted scores for each category.

    For correctness (40% weight, 0 points in this category results in 0 points for every other category directly):
    - 1 point in correctness: The code does not address the task at hand
    - 2 points in correctness: There are many major issues which prevent the code from fulfilling the required task.
    - 3 points in correctness: there are a few major issues / many minor issues which prevent the code from fulfilling the required tasks.
    - 4 points in correctness: There are a few minor corrections that must be completed before the code can fulfil the required task
    - 5 points in correctness: The code correctly implements the specified task and runs without any issue.

    For structure (25% weight):
    - 1 point in structure: the structure of the code is terrible, almost impossible for someone unfamiliar with the code to understand.
    - 2 points in structure: the structure of the code is poor, can require a lot of effort for someone unfamiliar with the code to understand.
    - 3 points in structure: the structure of the code is acceptable, can be understood with some effort by someone unfamiliar with the code.
    - 4 point in structure: the structure of the code is good, can be understood with a little effort by someone unfamiliar with the code.
    - 5 points in structure: the code is well-structured, someone unfamiliar with the code can understand it fully at a glance.

    For legibility (20% weight):
    - 1 point in legibility: the variable names are meaningless, the code is incomprehensible without viewing the documentation.
    - 2 points in legibility: the variable names are very unclear or overly long, the workings of the code can be puzzled together with a lot of help from the documentation.
    - 3 points in legibility: the variable names are somewhat clear, the workings of the code can be understood with some help from the documentation.
    - 4 points in legibility: the variable names are very clear, the workings of the code can be understood with occasional guidance from the documentation.
    - 5 points in legibility: the variable names are succinct and clear, the workings of the code can be plainly understood without viewing the documentation.

    For documentation (15% weight).
    - 1 point in documentation: the code comments are totally missing or are wholly inadequate and unhelpful.
    - 2 points in documentation: the code comments provide little relevant information for a basic partial understanding of the code.
    - 3 points in documentation: the code comments provide some information needed for a basic overall understanding of the code.
    - 4 points in documentation: the code comments provide sufficient information needed for a thorough overall understanding of the code.
    - 5 points in documentation: the code comments provide an abundance of information that grants an insightful and thorough understanding of the code.

    Provide the output in the same format as the following examples, replacing values as needed:
    '''
    Example 1
    The provided implementation scores as follows:
    - correctness: 4 out of 5 points.
    - structure: 3 out of 5 points.
    - legibility: 2 out of 5 points.
    - documentation: 3 out of 5 points.
    The total score is the sum of these numbers multiplied by the weight of each category: 4 * 0.4 + 3 * 0.25 + 2 * 0.2 + 3 * 0.15 = 3.2
    {"Score": 3.2}

    Example 2
    The provided implementation scores as follows:
    - correctness: 0 out of 5 points.
    - structure: 0 out of 5 points.
    - legibility: 0 out of 5 points.
    - documentation: 0 out of 5 points.
    The total score is the sum of these numbers multiplied by the weight of each category: 0 * 0.5 + 0 * 0.2 + 0 * 0.1 + 0 * 0.1 = 0
    {"Score": 0}
    '''      
    """)

    model_name = "meta-llama/Llama-3.3-70B-Instruct"  # Replace with your Hugging Face model
    pipeline = LLMJudgePipeline(model_name=model_name, rating_prompt=detailed_prompt)

    input_file = "Data/gemini-1.5-pro_google_temp_0.0.jsonl"


    # Save the results to a new .jsonl file
    output_file = "LLama70B.jsonl"
    pipeline.process_jsonl(input_file, output_file)
    print(f"Results saved to {output_file}")


