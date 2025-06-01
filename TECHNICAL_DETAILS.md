# TECHNICAL DOCUMENTATION

## Data-Collection-Preprocessing
# Technical Documentation: 1_Preprocessing.ipynb

# File Description

This Jupyter Notebook, `1_Preprocessing.ipynb`, is responsible for the data acquisition, cleaning, transformation, and preparation of datasets for training a medical Question and Answer (Q&A) system. It integrates data from various sources, structures it into a unified format, and splits it into training and testing sets for subsequent model development.

# Key Functionality

The notebook performs the following main tasks:

1.  **Import Libraries:** Imports necessary libraries for data handling, API interaction, regular expressions, machine learning utilities, and visualization.
2.  **Data Acquisition:**
    *   Loads datasets from the Hugging Face Hub using the `datasets` library.
    *   Downloads datasets from Kaggle using the Kaggle API. Authentication is handled through environment variables for username and API key.
3.  **Data Transformation:**
    *   Defines functions to transform data from each source into a standardized JSON-like structure. This involves extracting questions, answers, options, and source information, and assigning a question `type` (e.g., `multiple_choice`, `true_false`, `short_answer`, `multi_hop`).
    *   Handles variations in source data structures and formats, including parsing options and extracting reasoning steps.
4.  **Dataset Consolidation:** Combines the transformed data from all sources into a single dictionary, categorized by question type.
5.  **Data Cleaning:** Removes entries with missing critical information (e.g., `correct_answer` for multiple-choice questions).
6.  **Dataset Splitting:** Divides the consolidated dataset into training and testing sets using `sklearn.model_selection.train_test_split`. A stratified split is employed based on the question `type` to ensure each question type is represented proportionally in both sets.
7.  **Formatting for Model Training:** Transforms the split datasets into a format suitable for language model training, creating "input" and "output" pairs tailored to each question type.
8.  **Saving Processed Data:** Saves the formatted training and testing datasets as zipped JSON files (`train_dataset.zip` and `test_dataset.zip`). Each zip file contains separate JSON files for each question type (`short_answer_data.json`, `true_false_data.json`, etc.).
9.  **Uploading to GitHub:** Uses the `github3.py` library to upload the generated zip files to a specified GitHub repository. GitHub authentication is handled using a token stored in Colab Secrets.
10. **Data Visualization:** Includes code to generate bar plots showing the distribution of question types and statistics on the multi-hop reasoning data (number of steps and character length) in the training and testing sets.
11. **Model Loading (Partial):** Loads a pre-trained language model (`unsloth/DeepSeek-R1-Distill-Llama-8B`) and its tokenizer using the `transformers` library, in preparation for potential fine-tuning (although the fine-tuning process itself is only partially defined in the provided code snippet).

# Dependencies

The notebook relies on the following key Python libraries:

*   `IPython`
*   `datasets`
*   `json`
*   `requests`
*   `re`
*   `pandas`
*   `kaggle`
*   `os`
*   `google.colab`
*   `sklearn`
*   `random`
*   `torch`
*   `transformers`
*   `github3`
*   `joblib`
*   `tqdm`
*   `nltk`
*   `seaborn`
*   `matplotlib`
*   `numpy`
*   `zipfile`

# Usage

To execute this notebook, you need:

*   A Google Colab environment.
*   A Kaggle account and API credentials configured in the Colab environment variables (`KAGGLE_USERNAME` and `KAGGLE_KEY`).
*   A GitHub token stored in Colab Secrets with the key `git` for uploading the processed datasets to a GitHub repository.

Running the cells sequentially will perform the data preprocessing steps, generate the training and testing datasets, and upload them to the specified GitHub repository.

## Rag-Specific-Preprocessing

k eyperiment
idnex 1 and index 2
source trackingretrieval

## Topic-Modeling


## Pubmed-Balanced-Index2

## Retrieve-k-Experiments

## Baseline

## Fine-tuning

## Evaluation
 --> check results folder in structure 
