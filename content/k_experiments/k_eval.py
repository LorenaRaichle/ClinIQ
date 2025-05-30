import os
import json
from pathlib import Path

from metrics.evaluation_suite import EvaluationSuite


base_path = Path(__file__).resolve().parent


evaluator = EvaluationSuite()


for k_folder in sorted(os.listdir(base_path)):
    k_path = os.path.join(base_path, k_folder)


    if not os.path.isdir(k_path):
        continue


    for file in os.listdir(k_path):
        if file.endswith(".json"):
            json_path = os.path.join(k_path, file)
            with open(json_path, "r") as f:
                predictions = json.load(f)

            allowed_labels = {"A", "B", "C", "na"}

            y_true = [p["true_answer"].strip().upper() if p["true_answer"].strip().upper() in allowed_labels else "na"
                      for p in predictions]
            y_pred = [p["predicted_letter"].strip().upper() if p[
                                                                   "predicted_letter"].strip().upper() in allowed_labels else "na"
                      for p in predictions]

            experiment_name = file.replace(".json", "")
            metrics = evaluator.evaluate_discrete_answers(
                predictions=y_pred,
                ground_truth=y_true,
                experiment_name=experiment_name,
                folder=k_path
            )

            print(f"Evaluated {file} in {k_folder}")
            print(f"Evaluated {file} in {k_folder} — Accuracy: {metrics['accuracy']:.2%}")

