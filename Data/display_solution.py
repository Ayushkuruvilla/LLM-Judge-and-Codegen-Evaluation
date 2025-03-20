import json


def pretty_print_solutions_from_file(file_path):
    """
    Reads a JSONL file and pretty-prints each task solution.
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                task = json.loads(line.strip())  # Parse each JSON line
                task_id = task.get("task_id", "N/A")
                solution = task.get("solution", "").strip("")

                print("=" * 60)
                print(f"Task ID: {task_id}")
                print("=" * 60)
                print(solution)
                print("\n")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")


if __name__ == "__main__":
    file_path = "gemini-1.5-pro_google_temp_0.0.jsonl"
    pretty_print_solutions_from_file(file_path)