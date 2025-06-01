# ClinIQ: A Medical Expert-Level Question-Answering System
**by Konstantin, Adria & Lorena**
## Evaluate the ability of generative models to produce factually accurate medical information


Goal of this project is the setup of a domain specific (medical) Q&A-system that is able to answer four possible question types: **Multiple Choice**, **True-False**, **Short Answer** and **Multi-Hop** with higher performance compared to the baseline model.
We fine-tuned the **deepseek-coder-7b-instruct-v1.5** model and embedded it in a Retrieval-Augmented Generation (RAG) pipeline to access additional knowledge.


[Challenge description (University of Maryland)](https://brandonio-c.github.io/ClinIQLink-2025/)

---
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

<img src="visuals/Project Overview.png" alt="Project Overview" width="600"/>

To solve the challenge, we fine-tuned DeepSeek Coder-7B within a RAG framework.  
Our pipeline performs **hybrid retrieval** from a large training dataset and PubMed abstracts, combining semantic search and metadata filtering. The retrieved context is passed to the fine-tuned model to generate expert-level medical answers. Data sources are integrated via Pinecone and Google Drive, ensuring scalable and efficient knowledge access.

5 experiment types
k eyperiment
idnex 1 and index 2
source tracking retrieval
slide
---

## Project Structure

```
.
├── content/
│   ├── k_experiments/                  
│   └── Topic_modeling/  
├── data/
│   ├── raw/                            # Raw dataset
│   └── processed/                      # Processed dataset for /Pubmed /testdata /trainingdata
├── metrics/     
│   ├── evaluation_suite.py             # Evaluation suite for all Question Types
│   ├── final_results/                  # Evaluation suite results for all experiments
│   │   ├── AdvRAG+FT/
│   │   ├── BalancedAdvRAG+FT/
│   │   ├── Baseline/
│   │   ├── FineTuning/
│   │   └── NaiveRAG/
├── models/                             # Fine-tuned deepseek model
├── utils/
│   ├── prompt_utils.py                 # templates to generate model input
│   ├── RAG_adv_pipeline.py             # defining Hybrid Retriever & QA-chain
│   ├── RAG_answer.py                   # extraction of answer (letters) for all Question types
│   ├── RAG_metadata.py                 # metadata extraction for pinecone upsert & query
│   ├── RAG_naive_pipeline.py           # defining Naive Retriever & QA-chain
│   ├── RAG_pinecone.py                 # pinecone index creation, inserting data, index stats
│   └── RAG_preprocessing.py            # defining data paths, adding id & source info to all datasets
├── visuals/
├── .gitignore
├── 1a_Preprocessing_dataset.ipynb      # Preprocessing of 9 datasets for all Question types
├── 1b_Preprocessing_RAG.ipynb          # RAG specific preprocessing of 9 datasets for pinecone insertion
├── 2a_Baseline_7bcoder.ipynb           # Baseline deepseek
├── 2b_NaiveRAG_k_experiment.ipynb      # experiments to determine optimal k retrieved contexts parameter
├── 2c_TopicModeling_PubMed.ipynb       # PubMed articles Topic Modeling & upsert to pinecone (INDEX 1)
├── 2d_PubMed_train_balanced.ipynb      # populating balanced-index (INDEX 2) 
├── 3_Training_7b_LoRA_balanced.ipynb   # Fine-tuning deepseek
├── 4_Evaluation_RAG+FT.ipynb           # Eval
├── 5_Demo.ipynb
├── config.py                           # defining temp / max. new token / INDEX_NAME for experiments
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

