# ClinIQ: A Medical Expert-Level Question-Answering System

![Image](https://github.com/LorenaRaichle/ClinIQ/blob/main/visuals/Project%20Overview.png)



Goal and scope of this project is the training of a Q&A-system trained specifically on medical data, able to handle four different question types.
To optimize the output of our model, Data Augmentation, Preprocessing Enhancements, Parameter Choice and both Retrieval-Augmented Generation (RAG) as well as Fine-Tuning (FT) on a combination of 9+1 datasets are used to reach higher performance compared to the baseline model. 

Link to challenge: https://brandonio-c.github.io/ClinIQLink-2025/

## Table of Contents

1. [Project Description](#project-description)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
4. [Running the Project](#running-the-project)
5. [Reproducibility](#reproducibility)
6. [Team Contributions](#team-contributions)
7. [Results & Evaluation](#results--evaluation)
8. [References](#references)

---

## Project Description

[Provide a clear and concise summary of your project, outlining the problem statement, objectives, and your chosen approach using NLP and LLM techniques.]

---

## Project Structure

```
.
├── data/
│   ├── raw/                  # Raw dataset
│   └── processed/            # Processed dataset
├── logs/                     # Logs of training and evaluation runs
├── metrics/                  # Evaluation metrics and results
├── models/                   # Model checkpoints and exports
│   ├── checkpoints/
│   └── MyFirstModel.onnx
├── utils/
│   └── trainingMyCrazyModel.py
├── .gitignore
├── 1_Preprocessing.ipynb
├── 2_Baseline.ipynb
├── 3_Training.ipynb
├── 4_Evaluation.ipynb
├── 5_Demo.ipynb
├── CLEANCODE.MD
├── HELP.MD
├── README.MD
└── requirements.txt
```

---

## Setup Instructions

> [!NOTE]  
> This is only a Template. And you can add notes, Warnings and stuff with this format style. ([!NOTE], [!WARNING], [!IMPORTANT] )

### Clone Repository
```bash
git clone [repository-url]
cd [repository-folder]
```

### Create Environment
```bash
python -m venv venv
source venv/bin/activate  # Unix or MacOS
venv\Scripts\activate     # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Project

Follow these notebooks in order:
1. `1_Preprocessing.ipynb` - Data preprocessing
2. `2_Baseline.ipynb` - Establishing a baseline model
3. `3_Training.ipynb` - Model training
4. `4_Evaluation.ipynb` - Evaluating model performance
5. `5_Demo.ipynb` - Demonstration of the final model

You can also run custom scripts located in the `utils/` directory.

---

## Reproducibility

- **Random seeds:** Make sure random seeds are set and noted in your notebooks.
- **Environment:** Include the exact versions of libraries used (already covered by `requirements.txt`).
- **Data:** The data used for fine-tuning the model that has been processed in the preprocessing files can be accessed over websites such as Kaggle and Huggingface. 
- **Model Checkpoints:** Provide checkpoints clearly named and explained.

---

## Team Contributions

| Name              | Contributions                                  |
|-------------------|------------------------------------------------|
| L. R.| Preprocessing,  RAG, Visualization, Documentation|
| M. K. W.| Baseline model and Fine-tuning, Preprocessing, Visualization, Evaluation, Documentation |
| A. P.| Evaluation, Preprocessing, Visualization, Documentation|




---

## Results & Evaluation
- We have decided on a number of evaluation metrics to evaluate the model approaches for both closed-end and open-ended questions.
The Advanced RAG plus FT model consistently outperformed other models across most evaluation metrics, demonstrating strong performance across question types.
In Multiple Choice and True/False tasks, the FT model showed the highest accuracy for True/False questions (0.655), while Advanced RAG plus FT led in most other Multiple Choice metrics.
For Short Answer generation, Advanced RAG plus FT achieved the best scores in BLEU, METEOR, ROUGE, and similarity metrics, with notably high precision and recall.
The base model consistently underperformed, scoring significantly lower than the other models in all evaluation categories which has shown the continuous improvement over different model approaches. Multi-Hop reasoning results mirrored the Short Answer findings, with Advanced RAG plus FT again leading in terms of performance.

---

## References


## Datasets

| Dataset Name                                              | Source Description / Citation |
|-----------------------------------------------------------|-------------------------------|
| FreedomIntelligence/medical-o1-reasoning-SFT              | Chen et al. (2024). Huatuogpt-o1, towards medical complex reasoning with LLMs. |
| openlifescienceai/MedMCQA                                 | Pal et al. (2022). MedMCQA: A large-scale multi-subject multi-choice dataset for medical domain QA. |
| stellalisy/mediQ                                          | https://huggingface.co/datasets/stellalisy/mediQ |
| bigbio/MedQA                                              | Jin et al. (2021). What disease does this patient have? A large-scale open-domain QA dataset from medical exams. |
| UCSC-VLAA/MedReason                                       | Wu et al. (2025). MedReason: Eliciting factual medical reasoning steps in LLMs via knowledge graphs. |
| Ajayaadhi/Medical-QA                                      | https://huggingface.co/datasets/Ajayaadhi/Medical-QA |
| Comprehensive Medical Q&A Dataset (Kaggle)                | https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset |
| HPAI-BSC/OpenMedQA                                        | Bayarri Planas, J. (n.d.). https://huggingface.co/datasets/HPAI-BSC/OpenMedQA |
| qiaojin/PubMedQA                                          | https://huggingface.co/datasets/qiaojin/PubMedQA |
| MedRAG/pubmed                                             | Xiong et al. (2024). Benchmarking retrieval-augmented generation for medicine. |


---
## Directory of Writing Aids

| Aid                 | Usage / Application                                                                 | Affected Areas |
|---------------------|--------------------------------------------------------------------------------------|----------------|
| **Google's Gemini**     | Debugging and generation of code    | Preprocessing |
| **Chat GPT 4.0 – OpenAI** |Brainstorming relevant metrics for Evaluation, Writing, Formatting| Entire repository   | 
| **Perplexity.AI** |Writing, Formatting| Entire repository   |
<sub>Table 1: Writing Aids (Art. 57 AB)</sub>

