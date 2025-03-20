# Team 13: LLM Judge and Codegen Evaluation 

This project evaluates **code generation models** using the **EvalPlus** framework and assesses **LLM-based judges** for the quality of generated code.

---

## 📂 Folder Structure

- **/Data**  
  This folder contains the **code generation results** for various models run with **EvalPlus**. It also includes a Python script (`display_solution.py`) to format and display the generated code outputs.

- **/Evaluation**  
  This folder contains the **human evaluation results** as well as the **LLM-as-a-Judge model evaluation results**. Additionally, it includes the **pass@1 results** for the code generation models and an **agreement_rates** Python file to compare the human judge and the LLM-as-a-Judge.

- **/Prompts**  
  This folder contains different prompts used for evaluating various LLMs, including **pairwise comparison**, **code commenting using GPT-3.5 Turbo**, and **score generation as an LLM judge**.

- **/autometrics**  
  This folder contains the results from `compute_autometrics.ipynb`, including **CodeBERT scores**, **BLEU**, and **ROUGE** results for evaluating the code quality.
---

## 🔧 Running the Codegen Evaluation

To run the **code generation evaluation**:

1. Modify the `setup_and_run_evalplus.sh` script or setup_evalplus.bat script to change the model or backend required to run **EvalPlus** on your **local**.
2. Refer to the official **[EvalPlus Documentation](https://pypi.org/project/evalplus/)** for details on setting different LLM tokens if needed.
3. On **DelftBlue**, use the `setup_delftblue_evalplus.sh` script to set up and run the evaluation.

---

## 🔧 Running the Judge Evaluation

For **judging the code** using **LLM-based models**:

- Use **`llm_interface.py`** to run the evaluation with **OpenAI-based models**.
- Use **`llm_interface_comparison.py`** to run the **pairwise comparison evaluation** with OpenAI-based models.
- On **DelftBlue**, use **`llmdelft2`** along with the **`t1.sbatch`** script to submit the evaluation job:
  ```bash
  sbatch t1.sbatch
  
## 🔧 Running the Automatic Metrics Evaluation

For **judging the code** using **automatic metrics**:

- Use **`compute_autometrics.ipynb`** to run the automatic metrics computation with **available model outputs**.

## 📊 Models Evaluated
## Code Generation Models(List of models run with evalplus)
- gemini-1.5-pro : results in Data folder
- Mistral:  results in Data folder
- gpt-3.5 : Results in Data folder

## LLM-as-a-Judge Models(List of models run)
- gpt-4o : Results in Evaluation folder
- Code-Llama: Results in Evaluation folder