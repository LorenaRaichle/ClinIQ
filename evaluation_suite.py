import os
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
        
    def evaluate_discrete_answers(self, 
                                  predictions, 
                                  ground_truth,
                                  experiment_name,
                                  folder=None,
                                  RAG_sources=None
                                ):
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
        if folder:
            folder_name = f"{folder}/{experiment_name}"
        else:
            folder_name = f"/content/drive/MyDrive/NLP/05_Results/{experiment_name}"
        os.makedirs(folder_name, exist_ok=True) 

        # Generate confusion matrix
        labels = sorted(list(set(ground_truth + predictions)))  # Get all possible classes
        cm = confusion_matrix(ground_truth, predictions, labels=labels)

        # Display the matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{folder_name}/confusion_matrix.png", dpi=300)  # You can change filename and resolution
        plt.show()

        values = evaluate_classification(ground_truth, predictions)
        with open(f"{folder_name}/eval_scores.json", "w") as f:
            json.dump(values, f, indent=4)  # indent=4 for readable formatting

        pred_n_gt = {
            "predictions": predictions,
            "ground_truth": ground_truth
        }
        with open(f"{folder_name}/label_n_preds.json", "w") as f:
            json.dump(pred_n_gt, f, indent=4)  # indent=4 for readable formatting

        if RAG_sources:
            with open(f"{folder_name}/RAG_sources.json", "w") as f:
                json.dump(RAG_sources, f, indent=4)  # indent=4 for readable formatting

        return evaluate_classification(ground_truth, predictions)
        
    def evaluate_string_answers(self, 
                                predictions, 
                                ground_truth, 
                                experiment_name,
                                folder=None,
                                RAG_sources=None,
                                return_individual=False
                                ):
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


        if folder:
            folder_name = f"{folder}/{experiment_name}"
        else:
            folder_name = f"/content/drive/MyDrive/NLP/05_Results/{experiment_name}"
        os.makedirs(folder_name, exist_ok=True) 

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

        print("rouge_scores", rouge_scores)

        rouge1_list = [d['rouge1'] if isinstance(d, dict) else 0 for d in rouge_scores]
        rouge2_list = [d['rouge2'] if isinstance(d, dict) else 0 for d in rouge_scores]
        rougeL_list = [d['rougeL'] if isinstance(d, dict) else 0 for d in rouge_scores]


        word_similarity_list = [float(d['word_similarity']) if isinstance(d, dict) else 0 for d in semantic_scores]
        sentence_similarity_list = [float(d['sentence_similarity']) if isinstance(d, dict) else 0 for d in semantic_scores]
        paragraph_similarity_list = [float(d['paragraph_similarity']) if isinstance(d, dict) else 0 for d in semantic_scores]
        semantic_match_score_list = [float(d['semantic_match_score']) if isinstance(d, dict) else 0 for d in semantic_scores]

        per_sentence_scores = {
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


        avg_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0
        avg_meteor = float(np.mean(meteor_scores)) if meteor_scores else 0
        print("rouge_scores: ", rouge_scores)
        avg_rouge = {
            'rouge1': float(np.mean([s['rouge1'] if isinstance(s, dict) else 0 for s in rouge_scores])),
            'rouge2': float(np.mean([s['rouge2'] if isinstance(s, dict) else 0 for s in rouge_scores])),
            'rougeL': float(np.mean([s['rougeL'] if isinstance(s, dict) else 0 for s in rouge_scores]))
        }
        avg_cosine = float(np.mean(cosine_sims)) if cosine_sims else 0
        avg_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0
        avg_semantic = {
            k: float(np.mean([s[k] if isinstance(s, dict) else 0 for s in semantic_scores])) if semantic_scores else 0
            for k in semantic_scores[0].keys()
        } if semantic_scores else {}

        averaged_scores = {
                "avg_bleu": avg_bleu,
                "avg_meteor": avg_meteor,
                **avg_rouge,
                **avg_semantic,
                **bert_scores,
                "avg_cosine_similarity": avg_cosine,
                "avg_reasoning_coherence": avg_coherence
            }

        pred_n_gt = {
            "predictions": predictions,
            "ground_truth": ground_truth
        }
        with open(f"{folder_name}/label_n_preds.json", "w") as f:
            json.dump(pred_n_gt, f, indent=4)  # indent=4 for readable formatting

        if RAG_sources:
            with open(f"{folder_name}/RAG_sources.json", "w") as f:
                json.dump(RAG_sources, f, indent=4)  # indent=4 for readable formatting

        
        with open(f"{folder_name}/per_sentence_scores.json", "w") as f:
            json.dump(per_sentence_scores, f, indent=4)  # indent=4 for readable formatting
        
        
        with open(f"{folder_name}/averaged_scores.json", "w") as f:
            json.dump(averaged_scores, f, indent=4)  # indent=4 for readable formatting
        

        if return_individual == "both":
            return {
                "per_sentence_scores": per_sentence_scores,
                "averaged_scores": averaged_scores
            }
        if return_individual:
            return per_sentence_scores
        else:
            return averaged_scores








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

    gt_test = ['The cause of eosinophilic fasciitis is unknown. Fewer than 300 cases have been reported in the past 35 years.\nPeople with this condition have a buildup of eosinophils, a type of white blood cell, in the affected fascia and muscles. Eosinophils are related to allergic reactions, but their exact function is unknown.', 'Anterior hip dislocations are characterized by a flexed, abducted, and externally rotated hip. This means that the thigh bone is angled away from the body, the knee is bent, and the foot is turned outward. Anterior hip dislocations occur when the femoral head (the ball-shaped top of the thigh bone) is forced out of the hip socket in a forward direction. This type of injury is often caused by high-energy trauma, such as a car accident or a fall from a significant height. Anterior hip dislocations are considered a medical emergency and require prompt treatment to prevent complications such as nerve damage, blood vessel injury, or avascular necrosis (death of bone tissue due to lack of blood supply).', 'Polycythemia is a condition in which there is a net increase in the total number of red blood cells in the body. The overproduction of red blood cells may be due to a primary process in the bone marrow (a so-called myeloproliferative syndrome), or it may be a reaction to chronically low oxygen levels or, rarely, a malignancy.', 'Amyloidosis refers to a family of diseases in which there is a buildup of clumps of proteins called amyloids in body tissues and organs. These proteins slowly replace normal tissue, leading to failure of the involved organ. There are many forms of amyloidosis that may involve specific organs like the heart (cardiac amlyoidosis), the neurologic system with a peripheral neuropathy or the kidney. Cardiac amyloidosis usually occurs as part of primary amyloidosis (AL type amyloidosis). Primary amyloidosis is often seen in people with multiple myeloma and cancer. Cardiac amyloidosis (" stiff heart syndrome ") occurs when amyloid deposits take the place of normal heart muscle. It is the most typical form of restrictive cardiomyopathy. Cardiac amyloidosis may affect the way electrical impulses move through the heart (electrical conduction system). This can lead to irregular heart beating also known as arrhythmias and failure of the electrical impulses to conduct through the heart (heart block). Secondary amyloidosis (AA type amyloidosis) rarely affects the heart. However, one form of secondary amyloidosis called senile amyloidosis may involve the heart and blood vessels. Senile amyloidosis is caused by overproduction of a different protein. The condition is becoming more common as the average age of the population increases.\nThere is no difference in the incidence of cardiac amyloidosis between men and women as it affects both genders equally. The disease is rare in people under age 40.', "Medroxyprogesterone acetate (Provera) or oral contraceptive pills (OCPs) are commonly used to treat anovulatory bleeding. Anovulatory bleeding is a type of abnormal uterine bleeding that occurs when a woman's menstrual cycle is disrupted and ovulation does not occur. This can lead to irregular, heavy, or prolonged bleeding. Medroxyprogesterone acetate is a synthetic form of progesterone that can help regulate the menstrual cycle and reduce bleeding. It is usually taken for 5-10 days each month. OCPs contain a combination of estrogen and progesterone, and work by regulating the menstrual cycle and preventing ovulation. They are taken daily for 21 days, followed by a 7-day break during which bleeding occurs. The choice of medication will depend on the individual's specific circumstances, and should be discussed with a healthcare provider. In addition to medication, lifestyle changes such as regular exercise and a healthy diet can also help manage the symptoms of anovulatory bleeding.", 'What causes hypophosphatasia? Hypophosphatasia (HPP) is a genetic condition caused by mutations in the ALPL gene. This gene gives the body instructions to make an enzyme called alkaline phosphatase, which is needed for mineralization of the bones and teeth. Mutations in this gene lead to an abnormal version of the enzyme, thus affecting the mineralization process. A shortage of the enzyme also causes other substances to build up in the body. These abnormalities lead to the features of HPP. ALPL mutations that almost completely eliminate alkaline phosphatase activity generally cause the more severe forms of HPP, while mutations that reduce activity to a lesser extent often cause the milder forms of HPP.', 'Patients diagnosed with Bourbon virus disease have the following less common symptoms: \nNausea Vomiting Diarrhea', 'Thyroid follicular carcinoma is characterized by a malignant proliferation of follicles surrounded by a fibrous capsule with invasion through the capsule.', 'Sildenafil inhibits the enzyme PDE-5, which leads to an increase in the levels of nitric oxide. This, in turn, prolongs the vasodilatory effect of nitric oxide, resulting in increased blood flow to certain areas of the body.', 'Hydrocephalus is a medical condition characterized by an increase in cerebrospinal fluid (CSF) volume that causes ventricular dilation and may lead to increased intracranial pressure (ICP). CSF is a clear, colorless fluid that surrounds the brain and spinal cord, providing cushioning and nutrients. In hydrocephalus, the excess CSF puts pressure on the brain, which can lead to a variety of symptoms, including headaches, nausea, vomiting, and vision problems. In infants, hydrocephalus can cause the head to enlarge rapidly, while in adults, it can result in cognitive decline, difficulty walking, and urinary incontinence. Treatment for hydrocephalus typically involves surgical intervention to remove the excess fluid and restore normal CSF flow.']
    pred_test = ['Eosinophilic fasciitis is a rare disease caused by a combination of factors. It is often associated with other diseases like atopic dermatitis, food allergies, and other autoimmune disorders. The exact cause of Eosinophilic fasciitis is not known, but it is often associated with exposure to environmental allergens or irritants.\n\nThe disease is characterized by the development of eosinophilic infiltration in the dermis of the skin. This results in the development', '1. The hip joint is displaced to the side of the body, away from the patient.\n2. The patient will experience a severe amount of pain in the affected hip.\n3. The patient will be unable to bear weight on the affected hip.\n4. The patient may experience a limp or a loss of motion in the affected hip.\n5. The patient may have difficulty walking or standing.\n6. The patient may have a limp or a loss of motion in the affected', 'Polycythemia is a condition in which the bone marrow produces too many red blood cells. This can lead to a number of symptoms, including fatigue, shortness of breath, chest pain, and lightheadedness. Polycythemia can be caused by a number of factors, including certain medications, infections, and genetic disorders. It is important to treat polycythemia as it can lead to serious complications if left untreated. Treatment may involve medications to lower the number of red blood', 'The exact cause of cardiac amyloidosis is unknown. It is a progressive disease that usually affects the heart, causing it to enlarge. The disease is characterized by the deposition of amyloid-like proteins in the heart muscle and other organs.\n\nThere is no known cause for cardiac amyloidosis, although it is believed to be a result of a genetic predisposition combined with an environmental trigger. The trigger could be exposure to certain chemicals or toxins, such as heavy metals', '1. Clomiphene Citrate\n2. Metformin\n3. Tamoxifen\n4. Clomifene citrate\n5. Tamoxifen\n6. Clomifene citrate\n7. Tamoxifen\n8. Clomifene citrate\n9. Tamoxifen\n10. Clomifene citrate\n11. Tamoxifen\n12. Clomifene citrate\n13', 'Hypophosphatasia is a rare genetic disorder that causes a deficiency of phosphorus in the body. It is often associated with a deficiency of calcium, magnesium, and vitamin D, which are also needed for proper bone and muscle development. Hypophosphatasia is caused by mutations in the genes that code for phosphatase enzymes, which are involved in the breakdown of phosphorus in the body. The exact cause of hypophosphatasia is not known, but it is believed to be', 'N/A', '1. Follicular carcinoma is a subtype of thyroid cancer that arises from the follicular cells.\n2. Follicular carcinoma can present as a well-differentiated, non-seminomatous thyroid nodule.\n3. It is often asymptomatic or presents with a solitary thyroid nodule.\n4. Follicular carcinoma can be either invasive or non-invasive.\n5. Follicular carcinoma is more common in elderly patients.\n6. The incidence of follic', 'Sildenafil is an antihypertensive drug that acts by increasing the level of nitric oxide (NO) in the blood. Nitric oxide is a vasodilator, meaning it relaxes blood vessels and increases blood flow.\n\nSildenafil works by increasing the level of nitric oxide in the blood vessels, which leads to vasodilation and increased blood flow. This can help improve symptoms of erectile dysfunction, as well as reduce the risk of cardiovascular disease.', 'Hydrocephalus is a condition that refers to an excess of cerebrospinal fluid (CSF) in the brain or spinal cord. This excess fluid can cause pressure on the brain, leading to various symptoms, including headaches, seizures, and cognitive impairments. The CSF is a clear fluid that surrounds the brain and spinal cord, helping to maintain the pressure within the skull. In hydrocephalus, the production of CSF in the brain is either increased or decreased, leading to an imbalance']

    
    evalsuit = EvaluationSuite()

    print("Hier läufts irgendwie nicht")
    print(evalsuit.evaluate_string_answers(pred_test, gt_test, experiment_name="testNA", folder="output", RAG_sources={"med": 2, "test": 3}))

    scores_for_predictions = []
    for i, pred in enumerate(predictions):
        scores = evalsuit.evaluate_string_answers(pred, ground_truth, experiment_name=f"preds_{i}", folder="output", RAG_sources={"med": 2, "test": 3})
        scores_for_predictions.append(scores)
  
    
    

    # Dictionary to hold the final JSON structure
    final_scores = {}

    final_scores['ground_truth'] = ground_truth

    for i, pred in enumerate(predictions):
        scores = evalsuit.evaluate_string_answers(pred, ground_truth, experiment_name=f"predstest_{i}", folder="output", RAG_sources={"med": 2, "test": 3})
        print(scores)
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


