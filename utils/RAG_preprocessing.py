"""
RAG_preprocessing.py

This file contains utility classes and functions to prepare our training and the PubMed datasets for RAG-based experiments.
It focuses on assigning unique, type-specific IDs and checking source information and file stats in the training data.

These IDs and sources enable consistent tracking across training, evaluation, and especially the retrieval process.
In the RAG pipeline, this is crucial because the retriever typically returns only document IDs, which need to be traced back to the original data stored in Google Drive.

Example:
                {'correct_answer': 'B',
             'id': 'mc_0',
             'options': {'A': '1-1 ½ year after eruption',
                         'B': '2-3 year after eruption',
                         'C': '6 months after eruption',
                         'D': 'None of the above',
                         'E': None},
             'question': 'Root completion of permanent tooth occurs',
             'source': 'MC3-openlifescienceai/medmcqa',
             'type': 'multiple_choice'}

Key Components:

- class DataPaths       # Centralized class for managing and loading dataset paths from the /Data directory.
- class DataStats       # Summary/inspection of loaded data
- class AddingIDs       # Assigns unique, type-prefixed IDs to dataset examples based on question type.
- class CheckingSources   # Checking source metadata to each entry for easier retrieval analysis.


"""


import json
from datasets import DatasetDict, Dataset
from pathlib import Path
import matplotlib.pyplot as plt



class DataPaths:
    """
    Centralized class for managing dataset paths.
    """
    BASE_PATH = Path(__file__).resolve().parents[1] / "data"

    paths = {
        # test data
        "test_raw": BASE_PATH / "raw/test_dataset.json",
        # final testset: shuffled, balanced, with id
        "test_processed" : BASE_PATH /"processed/testdata/test_dataset_w_id.json", # final test data set

        # training data
        "train_raw": BASE_PATH / "raw/train_dataset.json",
        "train_processed": BASE_PATH / "processed/trainingdata/train_datatset_RAG.json", # final train set for RAG upsert


        # Pubmed data
        "pubmed_part1" : BASE_PATH / "raw/pubmed_partial_1.jsonl",
        "pubmed_part2" : BASE_PATH / "raw/pubmed_partial_2.jsonl",
        "pubmed_raw": BASE_PATH / "raw/combined_pubmed.jsonl",
        "pubmed_processed_all": BASE_PATH / "processed/Pubmed/pubmed_processed.jsonl", # only the id is added
        "pubmed_preprocessed_250k": BASE_PATH / "processed/Pubmed/pubmed_sample_prepro.jsonl" # added "clean_content" field (spacy processed) for topic modeling on subset

    }

    @staticmethod
    def load(name):
        path = DataPaths.paths[name]
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    @staticmethod
    def save(name, data):
        path = DataPaths.paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        elif path.suffix == ".jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")






class DataStats:
    """
    Provides quick stats and summaries for datasets managed by DataPaths.
    """

    @staticmethod
    def get_stats(name, show_sources=False):
        data = DataPaths.load(name)
        print(f"\nStats for '{name}':")

        if isinstance(data, dict):  # Structured by question type
            total = 0
            for key, items in data.items():
                print(f" - {key}: {len(items)} examples")
                total += len(items)
            print(f"Total entries: {total}")

            if show_sources:
                CheckingSources.print_source_distribution(data, plot=True)

        elif isinstance(data, list):
            print(f"Total entries: {len(data)}")
            if data:
                sample_keys = data[0].keys()
                print(f"Sample keys: {list(sample_keys)}")
                if "id" in sample_keys:
                    missing_ids = sum(1 for item in data if "id" not in item)
                    print(f"Entries missing 'id': {missing_ids}")
        else:
            print("Unknown data format.")

        return data



class AddingIDs:
    """
    - Adds unique, question-type-specific IDs to each question in the training dataset.
    - This function expects a Python dictionary (typically loaded from a JSON file),
    where keys are question types (e.g., "multiple_choice", "multi_hop", etc.), and values are lists of question dictionaries.

    - Each question gets an "id" field with a type-based prefix (e.g., "mc_0", "mh_1").

    Example:
        {
            "multiple_choice": [{"question": "Q1", "answer": "A1"}],
            "multi_hop": [{"question": "Q2", "answer": "A2"}]
        }

    becomes:

        {
            "multiple_choice": [{"question": "Q1", "answer": "A1", "id": "mc_0"}],
            "multi_hop": [{"question": "Q2", "answer": "A2", "id": "mh_0"}]
        }

    Args:
        data (dict): A training dataset loaded from JSON.

    Returns:
        dict: The same dataset with added "id" fields.
    """

    @staticmethod
    def add_ids_traindata(dataset: dict) -> dict:
        for typ, questions in dataset.items():
            if typ == "multiple_choice":
                counter = 0
                for question in questions:
                    question["id"] = "mc_" + str(counter)
                    counter += 1
            elif typ == "multi_hop":
                counter = 0
                for question in questions:
                    question["id"] = "mh_" + str(counter)
                    counter += 1
            elif typ == "short_answer":
                counter = 0
                for question in questions:
                    question["id"] = "sa_" + str(counter)
                    counter += 1
            elif typ == "true_false":
                counter = 0
                for question in questions:
                    question["id"] = "tf_" + str(counter)
                    counter += 1
        return dataset

    @staticmethod
    def add_ids_pubmed(pubmed_data: list) -> list:
        """
        Adds IDs to flat list of PubMed documents.
        Each item gets an 'id' field like 'pubmed_1', 'pubmed_2', ...
        """
        for idx, item in enumerate(pubmed_data):
            item["id"] = f"pubmed_{idx}"
        return pubmed_data






class CheckingSources:

        @staticmethod
        def print_source_distribution(dataset: dict, plot=True):
            print("\n Source Distribution:")

            for qtype, questions in dataset.items():
                sources = {}
                missing_count = 0  # Track missing sources

                for q in questions:
                    source = q.get("source")
                    if not source:
                        source = "MISSING"
                        missing_count += 1
                    sources[source] = sources.get(source, 0) + 1

                print(f"\n- {qtype}:")
                for source, count in sorted(sources.items(), key=lambda x: -x[1]):
                    print(f"  {source}: {count}")

                if missing_count > 0:
                    print(f" {missing_count} items in '{qtype}' are missing the 'source' field.")
                else:
                    print(f"All items in '{qtype}' have a 'source' field.")

                if plot:
                    CheckingSources._plot_source_bar_chart(sources, qtype)

        @staticmethod
        def _plot_source_bar_chart(sources: dict, title: str):
            labels = list(sources.keys())
            counts = list(sources.values())

            plt.figure(figsize=(10, 4))
            plt.barh(labels, counts)
            plt.title(f"Source Distribution for '{title}'")
            plt.xlabel("Number of Entries")
            plt.tight_layout()
            plt.show()


