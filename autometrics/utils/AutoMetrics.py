import json
import os

import code_bert_score
import pandas as pd
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from autometrics.utils.chrF import *

class AutoMetrics:
    def __init__(self, filename):
        """
        Initialize the automatic metrics computer.
        """
        self.generated_datapath = os.path.join("autometrics/Data/model_outputs", filename) # replace with actual output of llm
        self.model_name = filename.split("_")[0]

    def load_gen_solutions(self):
        gss = []
        with open(self.generated_datapath, "r", encoding="utf-8") as file:
            for line in file:
                task_output = json.loads(line.strip())
                gss.append(task_output)
        return gss

    def compute_bleu(self, task, parsed_solution, cs):
        '''
        :param task: list of task ids
        :param parsed_solution: parsed LLM output solution for a SE task
        :param cs: canonical solution according to HumanEvalPlus
        :return: BLEU scores for each task solution
        '''
        # Compute BLEU score
        print(f'\nComputing BLEU score for task: {task}')

        gs_tokens = parsed_solution.split()  # Candidate should be a list
        cs_tokens = [cs.split()]  # Reference should be a list of lists

        smooth_fn = SmoothingFunction().method1  # Optional smoothing to avoid zero scores
        bleu_score = sentence_bleu(cs_tokens, gs_tokens, smoothing_function=smooth_fn)

        print(f"BLEU score: {bleu_score}")

        return bleu_score

    def compute_rouge(self, task, parsed_solution, cs):
        '''
        :param task: list of task ids
        :param parsed_solution: parsed LLM output solution for a SE task
        :param cs: canonical solution according to HumanEvalPlus
        :return: ROUGE-L scores for each task solution
        '''
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # Compute ROUGE-L score
        print(f'\nComputing ROUGE-L score for task: {task}')
        rouge_scores = scorer.score(cs, parsed_solution)['rougeL']

        print(f"Precision: {rouge_scores.precision:.2f}")
        print(f"Recall: {rouge_scores.recall:.2f}")
        print(f"F1-Score: {rouge_scores.fmeasure:.2f}")

        return rouge_scores.precision, rouge_scores.recall, rouge_scores.fmeasure

    def compute_bert(self, task, parsed_solution, cs):
        '''
        :param task: list of task ids
        :param parsed_solution: parsed LLM output solution for a SE task
        :param cs: canonical solution according to HumanEvalPlus
        :return: CodeBERTScore for each task solution
        '''

        # Compute ROUGE-L score
        print(f'\nComputing CodeBERTScore for task: {task}')

        solution = [parsed_solution]
        reference = [cs]

        # Compute CodeBERTScore (precision, recall, f1-score, f3-score (recall 3 times as important as precision)
        pr, rc, f1, f3 = code_bert_score.score(cands=solution, refs=reference, lang="python")

        # Print results
        print(f"Precision: {pr.item():.2f}")
        print(f"Recall: {rc.item():.2f}")
        print(f"F1-score: {f1.item():.2f}")
        print(f"F3-score: {f3.item():.2f}")

        return pr.item(), rc.item(), f1.item(), f3.item()

    def compute_metrics(self, css, gss, metric=None):
        '''
        Computes one of three automatic metrics
        :param css: canonical solutions according to HumanEvalPlus
        :param gss: generated solutions by the LLM of choice
        :param metric: selected metric to be computed
        :return: computed metric scores
        '''
        tasks = []  # task ids
        bleu_scores = []  # BLEU score
        rouge_pr = []  # ROUGE-L precision
        rouge_rc = []  # ROUGE-L recall
        rouge_f1 = []  # ROUGE-L f1-score
        cbs_pr = []  # CodeBertScore precision
        cbs_rc = []  # CodeBertScore recall
        cbs_f1 = []  # CodeBertScore f1-score
        cbs_f3 = []  # CodeBertScore f3-score (3x recall importance over precision)
        chrf_f1 = []  # Chrf++ f1-score

        if metric is not None:

            for gs, cs in zip(gss, css):
                task = gs['task_id']
                tasks.append(task)
                parsed_solution = gs['solution'].split('"""\n')[-1]  # remove redundant part of llm output
                cs = (cs.lstrip("\n")).strip("\n")  # remove empty lines around canonical solutions

                if metric == 'BLEU':  # Compute BLEU Metric
                    bleu_score = self.compute_bleu(task, parsed_solution, cs)
                    bleu_scores.append(bleu_score)

                elif metric == 'ROUGE':  # Compute ROUGE-L Metric
                    pr, rc, f1 = self.compute_rouge(task, parsed_solution, cs)
                    rouge_pr.append(pr)
                    rouge_rc.append(rc)
                    rouge_f1.append(f1)

                elif metric == 'BERT':  # Compute CodeBERTScore metric
                    bert_pr, bert_rc, bert_f1, bert_f3 = self.compute_bert(task, parsed_solution, cs)
                    cbs_pr.append(bert_pr)
                    cbs_rc.append(bert_rc)
                    cbs_f1.append(bert_f1)
                    cbs_f3.append(bert_f3)

                elif metric == 'CHRF':
                    with open('subject', 'w', encoding='utf-8') as sub:
                        sub.write(parsed_solution)
                    with open('reference', 'w', encoding='utf-8') as ref:
                        ref.write(cs)

                    # Compute ROUGE-L score
                    print(f'\nComputing chrF score for task: {task}')

                    chrf_f1_score = chrf_score()
                    chrf_f1.append(chrf_f1_score)

                else:
                    msg = 'Error! Metric other than BLEU, ROUGE & BERT specified.'
                    return msg

        msg = f'Metric {metric} computation complete.'
        return tasks, bleu_scores, rouge_pr, rouge_rc, rouge_f1, cbs_pr, cbs_rc, cbs_f1, cbs_f3, chrf_f1, msg

    def store_metrics(self, task_list, bleu_scores, rouge_pr, rouge_rc, rouge_f1, cbs_pr, cbs_rc, cbs_f1, cbs_f3, chrf_f1):
        bleu_data = {'Task Id': task_list, 'BLEU Score': bleu_scores}
        rouge_data = {'Task Id': task_list, 'ROUGE-L Precision': rouge_pr,
                      'ROUGE-L Recall': rouge_rc, 'ROUGE-L F1-score': rouge_f1}
        cbs_data = {'Task Id': task_list, 'CodeBERTScore Precision': cbs_pr,
                    'CodeBERTScore Recall': cbs_rc, 'CodeBERTScore F1-score': cbs_f1,
                    'CodeBERTScore F3-score': cbs_f3}
        chrf_data = {'Task Id': task_list, 'chrF F1-score': chrf_f1}


        bleu_df = pd.DataFrame(bleu_data)
        rouge_df = pd.DataFrame(rouge_data)
        cbs_df = pd.DataFrame(cbs_data)
        chrf_df = pd.DataFrame(chrf_data)

        bleu_score_filepath = os.path.join("autometrics/Data/metric_scores", "bleu_scores_" + self.model_name + ".csv")
        rouge_score_filepath = os.path.join("autometrics/Data/metric_scores", "rouge_scores_" + self.model_name + ".csv")
        cbs_filepath = os.path.join("autometrics/Data/metric_scores", "cbs_" + self.model_name + ".csv")
        chrf_filepath = os.path.join("autometrics/Data/metric_scores", "chrf_" + self.model_name + ".csv")
        bleu_df.to_csv(bleu_score_filepath, index=False)
        print(f'BLEU scores saved to {bleu_score_filepath}')
        rouge_df.to_csv(rouge_score_filepath, index=False)
        print(f'ROUGE scores saved to {rouge_score_filepath}')
        cbs_df.to_csv(cbs_filepath, index=False)
        print(f'CodeBERTScores saved to {cbs_filepath}')
        chrf_df.to_csv(chrf_filepath, index=False)
        print(f'chrF scores saved to {chrf_filepath}')

    # Normalize scores from (0, 1) to (1, 5) to allow for direct comparison with human evaluation results
    def normalize_bleu(self, task_list, bleu_scores):
        # Parameters for normalization
        old_min, old_max = 0, 1
        new_min, new_max = 1, 5

        # Normalize each value in the list
        normalized_scores = [new_min + (score - old_min) * (new_max - new_min) / (old_max - old_min) for score in bleu_scores]

        bleu_data = {'Task Id': task_list, 'BLEU Score': normalized_scores}

        bleu_df = pd.DataFrame(bleu_data)

        bleu_score_filepath = os.path.join("autometrics/Data/normalized_scores", "bleu_scores_" + self.model_name + ".csv")
        bleu_df.to_csv(bleu_score_filepath, index=False)
        print(f'Normalised BLEU scores saved to {bleu_score_filepath}')

    def normalize_rouge(self, task_list, rouge_pr, rouge_rc, rouge_f1):
        # Parameters for normalization
        old_min, old_max = 0, 1
        new_min, new_max = 1, 5

        # Normalize each value in the list
        normalized_pr = [new_min + (pr - old_min) * (new_max - new_min) / (old_max - old_min) for pr in rouge_pr]
        normalized_rc = [new_min + (rc - old_min) * (new_max - new_min) / (old_max - old_min) for rc in rouge_rc]
        normalized_f1 = [new_min + (f1 - old_min) * (new_max - new_min) / (old_max - old_min) for f1 in rouge_f1]

        rouge_data = {'Task Id': task_list, 'ROUGE-L Precision': normalized_pr, 'ROUGE-L Recall': normalized_rc, 'ROUGE-L F1-score': normalized_f1}

        rouge_df = pd.DataFrame(rouge_data)

        rouge_score_filepath = os.path.join("autometrics/Data/normalized_scores", "rouge_scores_" + self.model_name + ".csv")
        rouge_df.to_csv(rouge_score_filepath, index=False)
        print(f'Normalised ROUGE scores saved to {rouge_score_filepath}')

    def normalize_BERT(self, task_list, cbs_pr, cbs_rc, cbs_f1, cbs_f3):
        # Parameters for normalization
        old_min, old_max = 0, 1
        new_min, new_max = 1, 5

        # Normalize each value in the list
        normalized_pr = [new_min + (pr - old_min) * (new_max - new_min) / (old_max - old_min) for pr in cbs_pr]
        normalized_rc = [new_min + (rc - old_min) * (new_max - new_min) / (old_max - old_min) for rc in cbs_rc]
        normalized_f1 = [new_min + (f1 - old_min) * (new_max - new_min) / (old_max - old_min) for f1 in cbs_f1]
        normalized_f3 = [new_min + (f3 - old_min) * (new_max - new_min) / (old_max - old_min) for f3 in cbs_f3]

        cbs_data = {'Task Id': task_list, 'CodeBERTScore Precision': normalized_pr, 'CodeBERTScore Recall': normalized_rc,
                    'CodeBERTScore F1-score': normalized_f1, 'CodeBERTScore F3-score': normalized_f3}

        cbs_df = pd.DataFrame(cbs_data)

        cbs_filepath = os.path.join("autometrics/Data/normalized_scores", "cbs_" + self.model_name + ".csv")
        cbs_df.to_csv(cbs_filepath, index=False)
        print(f'Normalised CodeBERTScores saved to {cbs_filepath}')

    def normalize_CHRF(self, task_list, chrf_f1):
        # Parameters for normalization
        old_min, old_max = 0, 1
        new_min, new_max = 1, 5

        # Normalize each value in the list
        normalized_f1 = [new_min + (pr - old_min) * (new_max - new_min) / (old_max - old_min) for pr in chrf_f1]

        chrf_data = {'Task Id': task_list, 'chrF F1-score': normalized_f1}

        chrf_df = pd.DataFrame(chrf_data)

        chrf_filepath = os.path.join("autometrics/Data/normalized_scores", "chrf_" + self.model_name + ".csv")
        chrf_df.to_csv(chrf_filepath, index=False)
        print(f'Normalised chrF scores saved to {chrf_filepath}')

    def run(self):
        print('-----------------------------------------------------')
        print(f"\nProcessing for model output {self.model_name}...")

        gss = self.load_gen_solutions()  # get generated solutions (gemini-1.5)

        dataset = load_dataset("evalplus/humanevalplus")
        css = dataset["test"]["canonical_solution"]  # canonical solutions (humanevalplus)

        # get BLEU scores
        tasks, bleu, _, _, _, _, _, _, _, _, msg = self.compute_metrics(css, gss, metric='BLEU')
        print(msg)
        # get ROUGE-L scores
        _, _, rouge_pr, rouge_rc, rouge_f1, _, _, _, _, _, msg = self.compute_metrics(css, gss, metric='ROUGE')
        print(msg)
        # get CodeBERTScores
        _, _, _, _, _, cbs_pr, cbs_rc, cbs_f1, cbs_f3, _, msg = self.compute_metrics(css, gss, metric='BERT')
        print(msg)
        # get chrF scores
        _, _, _, _, _, _, _, _, _, chrf_f1, msg = self.compute_metrics(css, gss, metric='CHRF')
        print(msg)

        self.normalize_bleu(tasks, bleu)
        self.normalize_rouge(tasks, rouge_pr, rouge_rc, rouge_f1)
        self.normalize_BERT(tasks, cbs_pr, cbs_rc, cbs_f1, cbs_f3)
        self.normalize_CHRF(tasks, chrf_f1)

        self.store_metrics(tasks, bleu, rouge_pr, rouge_rc, rouge_f1, cbs_pr, cbs_rc, cbs_f1, cbs_f3, chrf_f1)
