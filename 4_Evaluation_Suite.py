import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


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
    def evaluate_MC(self, predictions, ground_truth):
        """
        Evaluates the accuracy of multiple-choice predictions.

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


    def evaluate_FT(self):
        pass
    
    def evaluate_SA(self):
        pass

    def evaluate_MH(self):
        pass