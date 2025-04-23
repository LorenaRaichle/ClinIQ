# clin_IQ: Medical expert-level Question-Answering

Goal and scope of this project is the training of a Q&A-system trained specifically on medical data, able to handle 4 types of questions.
To optimize the output of our model, Data Augmentation, Preprocessing Enhancements, Parameter Choice and both Retrieval-Augmented Generation as well as fine-tuning on a combination of 16 datasets are used to reach higher performance compared to the baseline model. 

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
- **Data:** The data used for fine-tuning the model that has been processed in the "1_Preprocessing.ipynb" file can be accessed over websites such as Kaggle, Huggingface and Github. In case of any local data to this repository, this can be found under data/raw.
Additionally we have provided a list with all datasets used, in an Excel sheet "sources_links_to_raw_data.xlsx" (data/raw) which includes the respective link to each data source.
- **Model Checkpoints:** Provide checkpoints clearly named and explained.

---

## Team Contributions

| Name              | Contributions                                  |
|-------------------|------------------------------------------------|
| L. R.| Data preprocessing, baseline model evaluation.|
| K. M. W.| Model training, hyperparameter tuning.|
| A. P.| Evaluation, Data preprocessing, visualization, documentation.|


*[Each team member should describe their contributions clearly here.]*

---

## Results & Evaluation

- [Briefly summarize your evaluation metrics, improvements from baseline, and insights drawn from experiments.]
- We have decided on a number of evaluation metrics, that can be found listed under metrics in the Excel sheet "Evaluation_Metrics.xlsx". We evaluate the model versions in the "4_Evaluation.ipynb" file.

---

## References

[List here any relevant papers, sources, libraries, or resources used in your project.]

- Doe, J. (2024). *Great NLP Paper*. Conference Name.
- [Library Used](https://example-library.com)

---
