import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from openai import Client

# =========================
# LLM Judge Pipeline
# =========================
class LLMJudgePipeline:
    def __init__(self, model_name="gpt2", model_type="huggingface", api_key=None, rating_prompt=None):
        """
        Initialize the LLM Judge Pipeline.
        Supports both Hugging Face and OpenAI models.
        """
        self.model_name = model_name
        self.model_type = model_type
        self.api_key = api_key
        self.device = 0 if torch.cuda.is_available() else -1

        if model_type == "huggingface":
            self.llm = self._load_huggingface_model()
        elif model_type == "openai":
            self.client = self._initialize_openai_client()

        # Use default rating prompt if none is provided
        self.rating_prompt = rating_prompt or (
            "Please rate the Sample Solutions from 1 to 10 based on correctness and clarity, "
            "and provide a brief explanation for the rating. Select the better solution."
        )

        # Load the HumanEval+ dataset
        self.humanevalplus_dataset = self._load_humanevalplus()

    def _load_huggingface_model(self):
        """Load the Hugging Face model on the GPU if available."""
        print(f"Loading Hugging Face model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=self.device
        )

    def _initialize_openai_client(self):
        """Initialize the OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required for OpenAI models.")
        return Client(api_key=self.api_key)

    def _generate_openai_response(self, prompt, max_tokens=500):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Error during code generation:", e)
            return None

    def judge_solution(self, prompt, canonical_solution, solution_1, solution_2):
        """Evaluate the two sample solutions and select the better one."""
        evaluation_prompt = (
            f"{self.rating_prompt}\n"
            f"Task: {prompt}\n"
            f"Solution 1: {solution_1}\n"
            f"Solution 2: {solution_2}\n"
            f"Please evaluate both solutions and identify the better one."
        )

        if self.model_type == "huggingface":
            response = self.llm(evaluation_prompt, max_length=500, do_sample=True, temperature=0.7)
            return response[0]["generated_text"].strip()
        elif self.model_type == "openai":
            return self._generate_openai_response(evaluation_prompt, max_tokens=1000)

    def process_jsonl(self, jsonl_file_1, jsonl_file_2, output_file):
        """Process two JSONL files, evaluate each pair of solutions, and append the result to the output file."""
        with open(jsonl_file_1, "r") as file_1, open(jsonl_file_2, "r") as file_2, open(output_file, "a") as out_file:
            for line_1, line_2 in zip(file_1, file_2):
                data_1 = json.loads(line_1)
                data_2 = json.loads(line_2)

                task_id = data_1["task_id"]
                solution_1 = data_1["solution"]
                solution_2 = data_2["solution"]

                # Retrieve prompt and canonical solution from HumanEval+
                prompt, canonical_solution = self.get_task_data(task_id)

                # Evaluate solutions
                evaluation = self.judge_solution(prompt, canonical_solution, solution_1, solution_2)

                result = {
                    "task_id": task_id,
                    "evaluation": evaluation
                }

                # Append the result to the output file
                out_file.write(json.dumps(result) + "\n")
                print(result)

    def get_task_data(self, task_id):
        """Retrieve prompt and canonical solution for a given task_id from HumanEval+."""
        for example in self.humanevalplus_dataset:
            if example["task_id"] == task_id:
                return example["prompt"], example["canonical_solution"]
        raise ValueError(f"No data found for task_id: {task_id}")

    def _load_humanevalplus(self):
        """Load the HumanEval+ dataset using the 'test' split."""
        print("Loading the HumanEval+ dataset...")
        test_dataset = load_dataset("evalplus/humanevalplus", split="test")
        return test_dataset


if __name__ == "__main__":
    # Add OpenAI key
    openai_api_key = "sample-key"

    # Choose model: either "huggingface" or "openai"
    model_name = "gpt-4o"
    model_type = "openai"

    detailed_prompt = (
        """
    You will be given a coding task and an implementation. Your task is to attribute a score for each relevant characteristic of the implementation. The score must accurately and fairly represent the quality of the code.
    Points in the score are awarded per category. The total score equals the sum of the weighted scores for each category.
    
    For correctness (40% weight, 1 point in this category results in 0 points overall directly):
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
    - correctness: 1 out of 5 points.
    - structure: 0 out of 5 points.
    - legibility: 0 out of 5 points.
    - documentation: 0 out of 5 points.
    The total score is the sum of these numbers multiplied by the weight of each category: 0 * 0.5 + 0 * 0.2 + 0 * 0.1 + 0 * 0.1 = 0
    {"Score": 0}
    '''
        
    """)

    pipeline = LLMJudgePipeline(
        model_name=model_name,
        model_type=model_type,
        api_key=openai_api_key,
        rating_prompt=detailed_prompt,
    )

    # Evaluate the two JSONL files
    input_file_1 = "Data/gpt-3.5-turbo_openai_temp_0.0.jsonl"
    input_file_2 = "Data/gemini-1.5-pro_google_temp_0.0.jsonl"
    output_file = "Evaluation/llm_comparison_results_gpt-gemini.jsonl"

    pipeline.process_jsonl(input_file_1, input_file_2, output_file)

    print(f"Results are being appended to {output_file}")