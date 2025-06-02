# ClinIQ: A Medical Expert-Level Question-Answering System
### Evaluate the ability of generative models to produce factually accurate medical information
**by Konstantin, Adria & Lorena**
--- 

[Challenge description (University of Maryland)](https://brandonio-c.github.io/ClinIQLink-2025/)

To solve the challenge, we fine-tuned DeepSeek Coder-7B within a Retrieval-Augmented Generation (RAG) framework.  

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
### Question Types

The ClinIQ Challenge evaluates a domain-specific (medical) question-answering system across four question types: 

| **Question Type**             | **Example**                                                                                                                                                                                                                                                                          |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MC<br/> **Multiple Choice**   | ``` [{"correct_answer": "A", "options": { "A": "Localized immune complex", "B": "Ag- Ab reaction", "C": "Complement mediated", "D": "Ab mediated"}, "question": "Ahus reaction is…", "source": "MC4-UCSC-VLAA/MedReason", "type": "multiple_choice"}]```                             |
| TF<br/> **True-False**        | ``` [{"answer": "True",  "question": "Is reduced Klotho associated with the presence …?",  "source": "TF2-qiaojin/PubMedQA",  "type": "true_false"} ]```                                                                                                                                 |
| SA<br/> **Short Answer**      | ``` [{"question": "What is the value of …?",  "answer": "In the late distal tubule, [TF/P]osm is less than 1 when there is low ADH.",  "source": "SA2-Ajayaadhi/Medical-QA",  "type": "short_answer"}    ]```                                                                            |
| MH<br/> **Multi-Hop**         | ``` [{"question": "In a study assessing …?",  "answer": "In a normal...",  "reasoning": [  "Step 1: Alright, ...",  "Step 2: When a student’s...",  "Step 3: I remember from stats that ...",…],  "source": "MH-FreedomIntelligence/medical-o1-reasoning-SFT",  "type": "multi_hop"}]``` |

The goal is to achieve a higher performance compared to the baseline model by fine-tuning the **deepseek-coder-7b-instruct-v1.5** model and embedding it in a RAG pipeline to access additional knowledge.

### Strategies
We explored and evaluated a range of modeling strategies to tackle the ClinIQ challenge:

| **Approach**             | **Explanation**                                                                             |
|--------------------------|---------------------------------------------------------------------------------------------|
| **1 Baseline**           | performance evaluation of the **deepseek-coder-7b-instruct-v1.5**                           |
| **2 FT (Fine-tuning)**   | fine-tuned **deepseek-coder-7b-instruct-v1.5** using LoRA on 9 medical datasets             |
| **3 RAG**                | (Naive) adding a RAG pipeline to the baseline model (Pinecone INDEX 1)                      |
| **4 RAG + FT**           | (Advanced) adding a RAG pipeline to the fine-tuned model (knowledge base: Pinecone INDEX 1) |
| **5 [RAG + FT balanced]** | (Advanced) same as **RAG + FT** but knowledge base in Pinecone (INDEX 2) is **balanced**    |


<img src="visuals/project_architecture.png" alt="Project Overview" width="600"/>


We perform question type specific evaluation for each of the 5 modeling strategies:
- Discrete Evaluation for MC and TF (classification accuracy)
- Text Evaluation for SA and MH (generative quality, BLEU/ROUGE, etc.)

### System Architecture

Our Advanced RAG pipeline performs **hybrid retrieval** from a large training dataset and PubMed abstracts, combining **semantic search** and **metadata filtering**. The retrieved top k contexts are passed to the fine-tuned model to generate expert-level medical answers. Data sources are integrated via Pinecone and Google Drive, ensuring scalable and efficient knowledge access.

<img src="visuals/Project Overview.png" alt="Architecture Overview" width="600"/>


### Workflow

Our workflow consists of several key stages. Below is a summary of each step, along with links to the relevant notebooks for code execution. For in-depth explanations, please refer to our [Technical Documentation](TECHNICAL_DETAILS.md).

### 1 Data Collection & Processing Training set  
We unified and preprocessed **9 datasets** covering all question types (MC, TF, SA, MH). More than 500k questions were gathered across 9 different datasets with different layouts,  producing a consolidated train-test set.
[1a_Preprocessing_dataset Notebook →](1a_Preprocessing_dataset.ipynb) [Technical Documentation - data-collection-preprocessing](TECHNICAL_DETAILS.md#data-collection-preprocessing) 


### 2 RAG Knowledge Base Population & Retrieval Setup
We designed a RAG pipeline with a hybrid retriever and populated its Pinecone vector store with two core sources:
  - **Training Dataset** (400K questions with metadata)  
  - **PubMed Abstracts** (reduced to ~15K via topic modeling)

Multiple steps in preprocessing and filtering were required to populate our Pinecone knowledge base:
- **RAG-Specific Preprocessing**
 We assigned unique, type-specific IDs to all training samples and PubMed abstracts. This step ensures reliable document tracking across Retrieval, Model context injection and Evaluation. 
Since we designed the retriever to return only document IDs of the Pinecone knowledge base, this mapping is crucial to reconstruct full context from the Google Drive-stored content before passing it to deepseek.
[1b_Preprocessing_RAG Notebook →](1b_Preprocessing_RAG.ipynb) [Technical Documentation - rag-specific-preprocessing](TECHNICAL_DETAILS.md#rag-specific-preprocessing)


- **PubMed Topic Modeling (INDEX 1)**
To overcome Pinecone’s index size limits, we reduced the original **2.5M PubMed abstracts** to a more representative sample of 11 k abstracts using topic modeling. This ensured better coverage and diversity while staying within memory constraints.
 [2c_TopicModeling_PubMed Notebook →](2c_TopicModeling_PubMed.ipynb) [Technical Documentation - topic-modeling](TECHNICAL_DETAILS.md#topic-modeling-index1)


- **PubMed Balanced Index Creation (INDEX 2)**
Post-presentation, we observed significant imbalance in **Index 1:** 400K training questions vs. 11K PubMed entries.
To address this, we created a balanced **Index 2** with: 100K training samples (20K per question type) and 100K PubMed abstracts  
[2d_PubMed_train_balanced Notebook →](2d_PubMed_train_balanced.ipynb) [Technical Documentation - balanced index](TECHNICAL_DETAILS.md#pubmed-balanced-index2)


After populating the Pinecone vectorstore, we conducted experiments on a subset of our data to determine the optimal number of top-k retrieved contexts to augment our LLM with:
- **k-parameter experiments**
We experimented with different `k` values in our retriever, validating through both literature and empirical tests. We finalized **k = 5** based on performance and efficiency.
 [2b_NaiveRAG_k_experiment →](2b_NaiveRAG_k_experiment.ipynb) [Technical Documentation - retrieve-k-experiments](TECHNICAL_DETAILS.md#retrieve-k-experiments)




  
### 3 Baseline for Comparison

As part of our evaluation, we used an "off-the-shelf" open-weight model. We used the DeepSeek model coder-7b-instruct-v1.5. This model is an instruction-based model that was trained mainly on code to aid programmers. We used best-practice parameters (e.g. for temperature and token length) and DeepSeek's guidelines for prompting (e.g. simple, concise prompts).
 [2a_Baseline_7bcoder.ipynb →](2a_Baseline_7bcoder.ipynb)  
[Technical Documentation - baseline ](TECHNICAL_DETAILS.md#baseline)




### 4 Fine-Tuning  
We fine-tuned the same model that was used in the baseline evaluation using LoRA (Low-Rank Adaptation). The training data was a mix of the different question types. Each question-answer-pair was used as a whole for training (no masking on the question to just train on the answer). This notebook also includes the evaluation of the fine-tuned model using the same parameters and prompts as the baseline evaluation.
 [3a Training Notebook →](3a_Training_7b_LoRA_balanced.ipynb) [Technical Documentation - fine-tuning](TECHNICAL_DETAILS.md#fine-tuning)



### 5 Evaluation
Evaluates the RAG+FT approach using the evaluation suite but also processes the indices that were retrieved for augmentation. The retrieved indices (sources) serve as the data basis for the bubble charts.
 [4_Evaluation_RAG+FT Notebook →](4_Evaluation_RAG+FT.ipynb) [Technical Documentation - evaluation](TECHNICAL_DETAILS.md#evaluation) 


  


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
├── config.py                           # defining temp / max. new token / INDEX_NAME for experiments
├── README.MD
├── requirements.txt
└── TECHNICAL_DETAILS.MD                # further explanations on technical details and decisions

```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/LorenaRaichle/ClinIQ.git
cd ClinIQ
```

### 2. Create a Python Virtual Environment

For Unix/MacOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Data

- Obtain the referenced medical datasets from Kaggle and HuggingFace (see the “References” section above).

### 5. (Optional) Set Up Jupyter Notebooks

If you wish to run the project via Jupyter:
```bash
pip install notebook
jupyter notebook
```
Then open the notebooks in your browser, starting with `1a_Preprocessing_dataset.ipynb`.

---

> [!NOTE]  
> For large-scale retrieval (Pinecone), ensure you have API keys and credentials configured if running the full pipeline. See [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) for details.
>
> GPU is recommended for training and evaluation steps.


## Running the Project - Notebook Overview

### 1. Preprocessing
- **`1a_Preprocessing_dataset.ipynb`**  
  Initial dataset preprocessing: loading, cleaning, and formatting of the data.

- **`1b_Preprocessing_RAG.ipynb`**  
  Prepares inputs specific to Retrieval-Augmented Generation (RAG).

---

### 2. Baselines & Experiments
- **`2a_Baseline_7bcoder.ipynb`**  
  Runs a baseline using a 7B model (`7bcoder`) to establish initial performance benchmarks.

- **`2b_NaiveRAG_k_experiment.ipynb`**  
  Tests naïve RAG configuration with different k parameters

- **`2c_TopicModeling_PubMed.ipynb`**  
  Applies topic modeling on the PubMed dataset to explore latent topics and document clusters.

- **`2d_PubMed_train_balanced.ipynb`**  
  Prepares a second index (balanced training data & Pubmed data)

---

### 3. Training
- **`3a_Training_7b_LoRA_balanced.ipynb`**  
  Fine-tunes a 7B model using Low-Rank Adaptation (LoRA) on a balanced training set.

---

### 4. Evaluation
- **`4_Evaluation_RAG+FT.ipynb`**  
  This notebook evaluates the performance of a Retrieval-Augmented Generation (RAG) pipeline combined with a finetuned DeepSeek model on various medical question types using two different Pinecone indices. It runs inference, extracts answers, and computes evaluation metrics for multiple choice, short answer, true/false, and multi-hop questions.


## Reproducibility
--------------- TO DO-------------------

- **Random seeds:** Random seeds are set and noted in your notebooks.
- **Environment:** Exact versions of libraries used are already covered by `requirements.txt`.
- **Data:** The data used for fine-tuning the model and setting up the RAG pipeline has been processed in the preprocessing files 1a_ and 1b_ can be accessed over websites such as Kaggle and Huggingface as referenced in the Datasets section below References. 
- **Model Checkpoints:** Checkpoints clearly named and explained.

---

## Team Contributions

| Name       | Contributions                                                                           |
|------------|-----------------------------------------------------------------------------------------|
| Lorena     | RAG-Preprocessing & Set-Up, Visualization, RAG-Evaluation, Documentation                |
| Konstantin | Baseline model and Fine-tuning, Preprocessing, Visualization, Evaluation, Documentation |
| Adria      | Data collection, Preprocessing, Visualization, Evaluation, Documentation                |

---

## Results & Evaluation

Multiple Choice and True/False Evaluation
| Model                | MC Acc. | MC Prec. | MC Rec. | T/F Acc. | T/F Prec. | T/F Rec. |
| -------------------- | ------: | -------: | ------: | -------: | --------: | -------: |
| Advanced RAG plus FT |   0.467 |    0.388 |   0.377 |    0.595 |     0.440 |    0.405 |
| Advanced RAG         |   0.337 |    0.263 |   0.249 |    0.437 |     0.401 |    0.305 |
| FT                   |   0.404 |    0.334 |   0.318 |    0.655 |     0.488 |    0.440 |
| Base                 |   0.254 |    0.054 |   0.169 |    0.477 |     0.488 |    0.500 |

Short Answer Evaluation
| Model                |  BLEU | METEOR | ROUGE1 | ROUGE2 | ROUGEL | Prec. |  Rec. |    F1 | CosSim | ReasonCoh | SentSim | ParaSim |
| -------------------- | ----: | -----: | -----: | -----: | -----: | ----: | ----: | ----: | -----: | --------: | ------: | ------: |
| Advanced RAG plus FT | 0.206 |  0.402 |  0.401 |  0.314 |  0.355 | 0.843 | 0.868 | 0.854 |  0.913 |     0.794 |   0.867 |   0.913 |
| Advanced RAG         | 0.118 |  0.295 |  0.339 |  0.210 |  0.269 | 0.848 | 0.870 | 0.858 |  0.850 |     0.727 |   0.801 |   0.850 |
| FT                   | 0.049 |  0.207 |  0.263 |  0.109 |  0.188 | 0.834 | 0.838 | 0.835 |  0.836 |     0.654 |   0.665 |   0.836 |
| Base                 | 0.014 |  0.061 |  0.084 |  0.033 |  0.061 | 0.781 | 0.791 | 0.785 |  0.262 |     0.191 |   0.237 |   0.262 |

Multi Hop Evaluation
| Model                |  BLEU | METEOR | ROUGE1 | ROUGE2 | ROUGEL | Prec. |  Rec. |    F1 | CosSim | ReasonCoh | SentSim | ParaSim |
| -------------------- | ----: | -----: | -----: | -----: | -----: | ----: | ----: | ----: | -----: | --------: | ------: | ------: |
| Advanced RAG plus FT | 0.241 |  0.501 |  0.544 |  0.367 |  0.406 | 0.893 | 0.922 | 0.907 |  0.980 |     0.887 |   0.926 |   0.980 |
| Advanced RAG         | 0.156 |  0.359 |  0.407 |  0.245 |  0.295 | 0.862 | 0.892 | 0.876 |  0.855 |     0.710 |   0.799 |   0.855 |
| FT                   | 0.079 |  0.303 |  0.406 |  0.162 |  0.247 | 0.862 | 0.879 | 0.870 |  0.947 |     0.749 |   0.790 |   0.947 |
| Base                 | 0.018 |  0.086 |  0.131 |  0.040 |  0.077 | 0.796 | 0.813 | 0.804 |  0.352 |     0.253 |   0.306 |   0.352 |

We have decided on a number of evaluation metrics to evaluate the model approaches for both closed-end and open-ended questions.
The Advanced RAG plus FT model consistently outperformed other models across most evaluation metrics, demonstrating strong performance across question types.
In Multiple Choice and True/False tasks, the FT model showed the highest accuracy for True/False questions (0.655), while Advanced RAG plus FT led in most other Multiple Choice metrics.
For Short Answer generation, Advanced RAG plus FT achieved the best scores in BLEU, METEOR, ROUGE, and similarity metrics, with notably high precision and recall.
The base model consistently underperformed, scoring significantly lower than the other models in all evaluation categories which has shown the continuous improvement over different model approaches. Multi-Hop reasoning results mirrored the Short Answer findings, with Advanced RAG plus FT again leading in terms of performance.

---

## References


## Datasets

| Dataset Name                                              | Source Description / Citation |
|-----------------------------------------------------------|-------------------------------|
| FreedomIntelligence/medical-o1-reasoning-SFT              | Chen et al. (2024). Huatuogpt-o1, towards medical complex reasoning with LLMs. https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT |
| openlifescienceai/MedMCQA                                 | Pal et al. (2022). MedMCQA: A large-scale multi-subject multi-choice dataset for medical domain QA. https://huggingface.co/datasets/openlifescienceai/medmcqa|
| stellalisy/mediQ                                          | https://huggingface.co/datasets/stellalisy/mediQ |
| bigbio/MedQA                                              | Jin et al. (2021). What disease does this patient have? A large-scale open-domain QA dataset from medical exams. https://huggingface.co/datasets/bigbio/med_qa|
| UCSC-VLAA/MedReason                                       | Wu et al. (2025). MedReason: Eliciting factual medical reasoning steps in LLMs via knowledge graphs. https://huggingface.co/datasets/UCSC-VLAA/MedReason |
| Ajayaadhi/Medical-QA                                      | https://huggingface.co/datasets/Ajayaadhi/Medical-QA |
| Comprehensive Medical Q&A Dataset (Kaggle)                | https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset |
| HPAI-BSC/OpenMedQA                                        | Bayarri Planas, J. (n.d.). https://huggingface.co/datasets/HPAI-BSC/OpenMedQA |
| qiaojin/PubMedQA                                          | https://huggingface.co/datasets/qiaojin/PubMedQA |
| MedRAG/pubmed                                             | Xiong et al. (2024). Benchmarking retrieval-augmented generation for medicine. https://huggingface.co/datasets/MedRAG/pubmed|


---
## Directory of Writing Aids

| Aid                 | Usage / Application                                                                 | Affected Areas |
|---------------------|--------------------------------------------------------------------------------------|----------------|
| **Google's Gemini**     | Debugging and generation of code    | Preprocessing |
| **Chat GPT 4.0 – OpenAI** |Brainstorming relevant metrics for Evaluation, Writing, Formatting| Entire repository   | 
| **Perplexity.AI** |Writing, Formatting| Entire repository   |
---
<sub>Table 1: Writing Aids (Art. 57 AB)</sub>

