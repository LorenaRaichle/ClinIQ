import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import spacy
import json
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import tqdm
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

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return obj.item()
    else:
        return obj




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
        
    def evaluate_string_answers(self, predictions, ground_truth, return_individual=False):
        """
        Evaluates a list of predicted strings against ground truth strings using multiple NLP metrics.

        Parameters:
            predictions (List[str]): List of predicted text outputs.
            ground_truth (List[str]): Corresponding list of ground truth text outputs.
            return_individual (bool): If True, returns per-sample metric scores. 
                                    If False, returns average scores across all samples.

        Returns:
            dict: If return_individual is False, returns a dictionary with average values for:
                - BLEU, METEOR, ROUGE-1/2/L
                - Cosine similarity
                - Reasoning coherence
                - Semantic similarity scores (word, sentence, paragraph, match)
                - BERTScore metrics (if implemented in `evaluate_bertscore`)
                
                If return_individual is True, returns a dictionary with lists of per-sample values for:
                - All metrics listed above (except BERTScore).
                
        Notes:
            - Samples with prediction 'N/A' are treated as zero scores.
            - Assumes `compute_bleu`, `meteor_score`, `compute_rouge`, 
            `compute_cosine_similarity`, `evaluate_reasoning_flow`, 
            `semantic_match_score`, and `evaluate_bertscore` are implemented.
        """
        assert len(predictions) == len(ground_truth)

        bleu_scores, meteor_scores, rouge_scores, cosine_sims, coherence_scores, semantic_scores = [], [], [], [], [], []

        for ref, pred in zip(ground_truth, predictions):
            if pred == 'N/A':
                bleu_scores.append(0)
                meteor_scores.append(0)
                rouge_scores.append(0) 
                cosine_sims.append(0)
                coherence_scores.append(0)
                semantic_scores.append(0)

            bleu_scores.append(compute_bleu(ref, pred))
            meteor_scores.append(meteor_score([ref.split()], pred.split()))
            rouge_scores.append(compute_rouge(ref, pred)) 
            cosine_sims.append(float(compute_cosine_similarity(ref, pred)))
            coherence_scores.append(float(evaluate_reasoning_flow(pred)))
            semantic_scores.append(semantic_match_score(ref, pred))

        if return_individual:
            rouge1_list = [d['rouge1'] for d in rouge_scores]
            rouge2_list = [d['rouge2'] for d in rouge_scores]
            rougeL_list = [d['rougeL'] for d in rouge_scores]


            word_similarity_list = [float(d['word_similarity']) for d in semantic_scores]
            sentence_similarity_list = [float(d['sentence_similarity']) for d in semantic_scores]
            paragraph_similarity_list = [float(d['paragraph_similarity']) for d in semantic_scores]
            semantic_match_score_list = [float(d['semantic_match_score']) for d in semantic_scores]

            return {
                "bleu": bleu_scores,
                "meteor": meteor_scores,
                "rouge1": rouge1_list,
                "rouge2": rouge2_list,
                "rougeL": rougeL_list,
                "cosine_similarity": cosine_sims,
                "reasoning_coherence": coherence_scores,
                "word_similarity": word_similarity_list,
                "sentence_similarity": sentence_similarity_list,
                "paragraph_similarity": paragraph_similarity_list,
                "semantic_match_score": semantic_match_score_list
            }

            
        bert_scores = evaluate_bertscore(ground_truth, predictions)


        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        avg_meteor = np.mean(meteor_scores) if meteor_scores else 0
        avg_rouge = {
            'rouge1': np.mean([s['rouge1'] for s in rouge_scores]) if rouge_scores else 0,
            'rouge2': np.mean([s['rouge2'] for s in rouge_scores]) if rouge_scores else 0,
            'rougeL': np.mean([s['rougeL'] for s in rouge_scores]) if rouge_scores else 0
        }
        avg_cosine = np.mean(cosine_sims) if cosine_sims else 0
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0
        avg_semantic = {
            k: np.mean([s[k] for s in semantic_scores]) if semantic_scores else 0
            for k in semantic_scores[0].keys()
        } if semantic_scores else {}

        return {
            "avg_bleu": avg_bleu,
            "avg_meteor": avg_meteor,
            **avg_rouge,
            **avg_semantic,
            **bert_scores,
            "avg_cosine_similarity": avg_cosine,
            "avg_reasoning_coherence": avg_coherence
        }








def main():
    """The main function of my Python Application"""
    print('Testing evaluate_string_answers on these arrays:')
    ground_truth = [
    "The capital of France is Paris. It is known for its art, fashion, and culture. The Eiffel Tower is one of its most famous landmarks.",
    "Water boils at 100 degrees Celsius. This is under standard atmospheric pressure. It's a key concept in physical science and cooking.",
    "The Great Wall of China is visible from space. It was built over centuries to protect against invasions. It's one of the New Seven Wonders of the World.",
    "Shakespeare wrote Hamlet. It is one of his most famous tragedies. The play explores themes of revenge, madness, and mortality.",
    "Photosynthesis is the process by which plants make food. It uses sunlight, carbon dioxide, and water. This process produces oxygen as a byproduct.",
    "The Earth revolves around the Sun. This takes approximately 365.25 days. It causes the changing of seasons.",
    "Mount Everest is the highest mountain on Earth. It reaches a height of 8,848 meters. It lies in the Himalayas between Nepal and Tibet.",
    "The human body has 206 bones. These bones support movement and protect organs. The femur is the longest bone in the body.",
    "The Pacific Ocean is the largest ocean on Earth. It covers more than 63 million square miles. It's home to the Mariana Trench, the deepest part of the ocean.",
    "The speed of light is approximately 299,792 kilometers per second. Nothing in the universe travels faster. This speed is critical in physics and cosmology."
    ]


    # Predicted answers from the LLM
    predictions = [
    # prediction0 – exact match
    [
        "The capital of France is Paris. It is known for its art, fashion, and culture. The Eiffel Tower is one of its most famous landmarks.",
        "Water boils at 100 degrees Celsius. This is under standard atmospheric pressure. It's a key concept in physical science and cooking.",
        "The Great Wall of China is visible from space. It was built over centuries to protect against invasions. It's one of the New Seven Wonders of the World.",
        "Shakespeare wrote Hamlet. It is one of his most famous tragedies. The play explores themes of revenge, madness, and mortality.",
        "Photosynthesis is the process by which plants make food. It uses sunlight, carbon dioxide, and water. This process produces oxygen as a byproduct.",
        "The Earth revolves around the Sun. This takes approximately 365.25 days. It causes the changing of seasons.",
        "Mount Everest is the highest mountain on Earth. It reaches a height of 8,848 meters. It lies in the Himalayas between Nepal and Tibet.",
        "The human body has 206 bones. These bones support movement and protect organs. The femur is the longest bone in the body.",
        "The Pacific Ocean is the largest ocean on Earth. It covers more than 63 million square miles. It's home to the Mariana Trench, the deepest part of the ocean.",
        "The speed of light is approximately 299,792 kilometers per second. Nothing in the universe travels faster. This speed is critical in physics and cosmology."
    ],
    
    # prediction1 – paraphrased but accurate
    [
        "Paris is the capital city of France. It is famous for its museums and historical sites. Tourists often visit the Eiffel Tower and the Louvre.",
        "Under normal pressure, water boils at 100°C. This temperature is important in both science and cooking. It's the basis for many temperature scales.",
        "From outer space, the Great Wall of China can be spotted. It stretches thousands of kilometers. Its historical purpose was defense against enemies.",
        "Hamlet was authored by William Shakespeare. It tells the story of a Danish prince seeking revenge. The play is known for its deep philosophical questions.",
        "Plants use photosynthesis to make their food. They need sunlight, water, and carbon dioxide. Oxygen is released during the process.",
        "Earth travels around the Sun once each year. This orbit defines our calendar. It influences the pattern of daylight and seasons.",
        "At 8,848 meters tall, Mount Everest is Earth’s tallest peak. Climbers come from around the world to summit it. It's part of the Himalaya range.",
        "An adult human skeleton has 206 bones. These form the structure of our body. Bones like the femur and skull are vital for movement and protection.",
        "The Pacific Ocean is Earth’s biggest body of water. It contains the deepest known oceanic point. The trench there plunges over 11,000 meters deep.",
        "Light moves at about 300,000 kilometers every second. This speed is used in calculating astronomical distances. It's the upper limit for matter and energy transmission."
    ],

    # prediction2 – humorous/informal
    [
        "Paris is basically France’s HQ for croissants and kissing. It’s where fashion and romance collide. Eiffel Tower selfies are practically mandatory.",
        "Water hits its boiling drama at 100°C. Steam goes wild at this point. It's like the water’s ultimate rage quit.",
        "That giant wall China built? You might catch it from orbit. It’s not just old — it’s legendary. Great for fending off invaders and impressing astronauts.",
        "Shakespeare dropped Hamlet like it was hot. Emo prince, ghost dad, revenge vibes — classic. The drama levels are off the charts.",
        "Photosynthesis: where leaves do light-eating magic. Sunlight in, sugar out. Trees are just nature’s green chefs.",
        "Earth dances around the Sun in a never-ending loop. One lap takes a year. Seasons? Just part of the choreography.",
        "Mount Everest towers like the Earth's nose poking space. It’s the ultimate hiking flex. But pack oxygen — it’s not beginner-friendly.",
        "Skeletons are hardcore: 206 bones and not one is chill. They hold you up, keep your guts safe. The femur’s the kingpin of leg bones.",
        "The Pacific Ocean? More like the planet’s water blanket. It's so deep, submarines get vertigo. That trench? Like nature’s bottomless pit.",
        "Light is the universe’s speedster. At nearly 300,000 km/s, it's untouchable. Even sci-fi can’t beat that."
    ],

    # prediction3 – unrelated fun facts
    [
        "A group of flamingos is called a flamboyance. They get their color from eating shrimp. Flamingos can sleep while standing on one leg.",
        "Bananas are technically berries. But strawberries aren’t. Nature loves exceptions.",
        "Octopuses have three hearts. Two pump blood to the gills, one to the body. They also have blue blood.",
        "Sloths can hold their breath for up to 40 minutes. That’s longer than dolphins. Slow and steady really does win sometimes.",
        "Scotland’s national animal is the unicorn. It’s a symbol of purity and strength. You’ll find it on royal coats of arms.",
        "Honey never spoils. Archaeologists found edible honey in ancient Egyptian tombs. It’s basically sugar in preservation mode.",
        "Wombats poop cubes. It helps prevent their droppings from rolling away. Weirdly practical.",
        "The Eiffel Tower can grow taller in summer. Heat expands the metal. It can gain about 15 centimeters.",
        "A day on Venus is longer than a year there. It rotates very slowly. Plus, it spins in the opposite direction to most planets.",
        "Sharks have been around longer than trees. They’ve existed for over 400 million years. They're true ancient survivors."
    ]
    ]

    
    evalsuit = EvaluationSuite()

    scores_for_predictions = []
    for pred in predictions:
        scores = evalsuit.evaluate_string_answers(pred, ground_truth)
        scores_for_predictions.append(scores)

    per_sample_values = evalsuit.evaluate_string_answers(predictions[1], ground_truth, return_individual=True)    
    print(per_sample_values)
  
    
    

    # Dictionary to hold the final JSON structure
    final_scores = {}

    final_scores['ground_truth'] = ground_truth

    for i, pred in enumerate(predictions):
        scores = evalsuit.evaluate_string_answers(pred, ground_truth)
        final_scores[f"prediction{i}"] = pred
        final_scores[f"scores{i}"] = to_serializable(scores)

    with open("all_prediction_scores.json", "w") as f:
        json.dump(final_scores, f, indent=4)

    print("Saved as all_prediction_scores.json")
    
    scores = {}
    for i in range(4):
        scores[f"scores{i}"] = final_scores[f"scores{i}"]

        
    # Create DataFrame
    df = pd.DataFrame(scores)

    # Plot
    df.plot(kind='bar', figsize=(14, 8))
    plt.title("Scores Comparison by Metric")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Score Sets")
    plt.tight_layout()
    plt.grid(axis='y')

    plt.show()






if __name__ == '__main__': 
    main()


