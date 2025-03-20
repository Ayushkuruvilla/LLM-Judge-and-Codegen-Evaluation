import pandas as pd
import json


if __name__ == "__main__":

    # Load the JSONL and csv files
    jsonl_file_path = 'gpt4o_evaluation_gemini_1.jsonl'
    llm_scores = [json.loads(line) for line in open(jsonl_file_path, 'r')]

    # Convert JSONL to DataFrame
    llm_df = pd.DataFrame(llm_scores)

    # Load the CSV file
    csv_file_path = 'Human Evaluation - Sheet1.csv'
    human_df = pd.read_csv(csv_file_path)

    # Extract score from evaluation
    def extract_score(evaluation):
        try:
            score_data = json.loads(evaluation.split('\n')[-1])
            return score_data.get("Score")
        except (json.JSONDecodeError, IndexError):
            return None

    # Apply the score extraction to LLM evaluations
    llm_df['score'] = llm_df['evaluation'].apply(extract_score)

    # Clean and merge the task IDs for alignment
    llm_df['task_id'] = llm_df['task_id'].str.replace("HumanEval/", "").astype(int)

    # Rename columns to prepare for merge
    human_df.rename(columns={'Task ID': 'task_id', 'Score': 'human_score'}, inplace=True)

    # Merge on task_id
    merged_df = pd.merge(llm_df[['task_id', 'score']], human_df[['task_id', 'human_score']], on='task_id')

    # Calculate agreement rate
    agreement_rate = (merged_df['score'] == merged_df['human_score']).mean()

    # Print the agreement rate
    print(f"Agreement Rate: {agreement_rate:.2%}")

    # Save the merged data to a CSV for review
    merged_df.to_csv('merged_evaluation_results.csv', index=False)




