import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import Levenshtein
import spacy
from tabulate import tabulate
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
nlp = spacy.load("en_core_web_md")
nltk.download('wordnet')
nltk.download('punkt_tab')

def evaluate_classification(y_true, y_pred):
    """
    Calculate accuracy, precision, and recall for letter classification.

    Parameters:
        y_true (list): True labels
        y_pred (list): Predicted labels

    Returns:
        dict: A dictionary with accuracy, precision, and recall
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }


def compute_bleu(reference, prediction):
    """
    Compute BLEU score between a reference and a prediction.

    Parameters:
        reference (str): The ground truth answer.
        prediction (str): The model's predicted answer.

    Returns:
        float: BLEU score (0 to 1)
    """
    # Tokenize by splitting on whitespace
    reference_tokens = [reference.split()]
    prediction_tokens = prediction.split()

    # Smoothing helps avoid zero scores for short predictions
    smoothie = SmoothingFunction().method4

    return sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothie)


def evaluate_bertscore(references, predictions, lang='en'):#https://milvus.io/ai-quick-reference/what-is-bertscore-or-other-embeddingbased-metrics-and-can-they-be-helpful-in-evaluating-the-similarity-between-a-generated-answer-and-a-reference-answer-or-source-text
    P, R, F1 = score(predictions, references, lang=lang)
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()}

def compute_cosine_similarity(reference, prediction):#https://www.comet.com/site/blog/bertscore-for-llm-evaluation/
    
    doc1 = nlp(reference)
    doc2 = nlp(prediction)
    return cosine_similarity([doc1.vector], [doc2.vector])[0][0]


def evaluate_reasoning_flow(prediction): #https://www.galileo.ai/blog/g-eval-metric
    sentences = sent_tokenize(prediction)
    flow_scores = []
    for i in range(1, len(sentences)):
        score = compute_cosine_similarity(sentences[i - 1], sentences[i])
        flow_scores.append(score)
    return np.mean(flow_scores) if flow_scores else 0

def semantic_match_score(reference, prediction, weights=(0.3, 0.3, 0.4)):
    # Tokenize sentences
    ref_doc = nlp(reference)
    pred_doc = nlp(prediction)

    # --- Word Level ---
    word_sim = cosine_similarity([ref_doc.vector], [pred_doc.vector])[0][0]

    # --- Sentence Level ---
    ref_sents = list(ref_doc.sents)
    pred_sents = list(pred_doc.sents)
    min_len = min(len(ref_sents), len(pred_sents))
    sent_sims = [
        cosine_similarity([ref_sents[i].vector], [pred_sents[i].vector])[0][0]
        for i in range(min_len)
    ]
    sentence_sim = np.mean(sent_sims) if sent_sims else 0

    # --- Paragraph Level ---
    paragraph_sim = cosine_similarity([ref_doc.vector], [pred_doc.vector])[0][0]

    # --- Weighted Sum ---
    w_word, w_sentence, w_paragraph = weights
    semantic_score = (
        w_word * word_sim +
        w_sentence * sentence_sim +
        w_paragraph * paragraph_sim
    )
    return {
        "word_similarity": word_sim,
        "sentence_similarity": sentence_sim,
        "paragraph_similarity": paragraph_sim,
        "semantic_match_score": semantic_score
    }

def compute_rouge(reference, prediction):
    """
    Compute ROUGE scores between a reference and a prediction.

    Parameters:
        reference (str): The ground truth answer.
        prediction (str): The model's predicted answer.

    Returns:
        dict: ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)

    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }






class EvaluationSuite():
    """
    Class that contains the evaluation scripts for each question type.
    """
        
    def evaluate_discrete_answers(self, predictions, ground_truth):
        """
        Evaluates the accuracy of multiple-choice and true false predictions.

        This function compares a list of predicted answers (e.g., "A", "B", "C", etc.)
        against the corresponding ground-truth labels and computes overall accuracy.

        Args:
            predictions (List[str]): A list of predicted choices (e.g., ["B", "C", "A", ...]).
            ground_truth (List[str]): A list of correct choices (same format and length as predictions).

        Returns:
            Saves a confusion matrix 
            dict: A dictionary containing:
            - 'accuracy' (float): Overall classification accuracy.
            - 'precision' (float): Macro-averaged precision across all classes.
            - 'recall' (float): Macro-averaged recall across all classes.
        """

        # Generate confusion matrix
        labels = sorted(list(set(ground_truth + predictions)))  # Get all possible classes
        cm = confusion_matrix(ground_truth, predictions, labels=labels)

        # Display the matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("output/confusion_matrix_MC.png", dpi=300)  # You can change filename and resolution
        plt.show()

        return evaluate_classification(ground_truth, predictions)
    
    def evaluate_string_answers(self, predictions, ground_truth):


        assert len(predictions) == len(ground_truth)

        bleu_scores = []
        meteor_scores = []
        lev_distances = []
        rouge_scores = []

        for ref, pred in zip(ground_truth, predictions):
            if pred == 'N/A':
                continue
            bleu_scores.append(compute_bleu(ref, pred))

            meteor_scores.append(meteor_score([ref.split()], pred.split()))
            lev_distances.append(Levenshtein.distance(ref, pred))
            rouge_scores.append(compute_rouge(ref, pred))

        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        avg_meteor = np.mean(meteor_scores) if meteor_scores else 0
        avg_levenshtein = np.mean(lev_distances) if lev_distances else 0
        avg_rouge = {
            'rouge1': np.mean([s['rouge1'] for s in rouge_scores]) if rouge_scores else 0,
            'rouge2': np.mean([s['rouge2'] for s in rouge_scores]) if rouge_scores else 0,
            'rougeL': np.mean([s['rougeL'] for s in rouge_scores]) if rouge_scores else 0
        }


        bert_scores = evaluate_bertscore(ground_truth, predictions)
        cosine_sims, coherence_scores, semantic_scores = [], [], []


        for ref, pred in zip(ground_truth, predictions):
            if pred == 'N/A':
                continue
            cosine_sims.append(compute_cosine_similarity(ref, pred))
            coherence_scores.append(evaluate_reasoning_flow(pred))
            semantic_scores.append(semantic_match_score(ref, pred))


        avg_cosine = np.mean(cosine_sims) if cosine_sims else 0
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0
        avg_semantic = {
            k: np.mean([s[k] for s in semantic_scores]) if semantic_scores else 0
            for k in semantic_scores[0].keys()
        } if semantic_scores else {}


        results = {
            "avg_bleu": avg_bleu,
            "avg_meteor": avg_meteor,
            "avg_levenshtein": avg_levenshtein,
            "avg_rouge": avg_rouge,
            "semantic_match_score": avg_semantic,
            "bertscore": bert_scores,
            "avg_cosine_similarity": avg_cosine,
            "avg_reasoning_coherence": avg_coherence
        }
        flat_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    flat_results[f"{k}_{subk}"] = subv
            else:
                flat_results[k] = v
        df = pd.DataFrame([flat_results])
        return df





def main():
    """The main function of my Python Application"""
    print('Testing evaluate_string_answers on these arrays:')
    ground_truth = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius.",
        "The Great Wall of China is visible from space.",
        "Shakespeare wrote Hamlet.",
        "Photosynthesis is the process by which plants make food."
    ]

    # Predicted answers from the LLM
    predictions = [
        "Paris is the capital of France.",
        "Water boils at one hundred degrees Celsius.",
        "The Great Wall of China can be seen from space.",
        "Hamlet was written by Shakespeare.",
        "Photosynthesis allows plants to produce their own food."
    ]

    
    evalsuit = EvaluationSuite()
    evals = evalsuit.evaluate_string_answers(predictions, ground_truth)
    print("Our Evaluation Metrics are: ")
    print(tabulate(evals, headers='keys', tablefmt='psql', showindex=False))

if __name__ == '__main__': 
    main()


