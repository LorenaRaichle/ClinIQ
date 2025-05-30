import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from metrics.evaluation_suite import EvaluationSuite

## evaluate k param choices and save plot

base_path = Path(__file__).resolve().parent
k_dirs = ["k_1", "k_3", "k_5", "k_8", "k_12"]

evaluator = EvaluationSuite()


for k_folder in k_dirs:
    k_path = base_path / k_folder

    if not k_path.is_dir():
        print(f"Skipping missing directory: {k_path}")
        continue

    for file in os.listdir(k_path):
        if file.endswith(".json") and "generated_answers" in file:
            json_path = k_path / file
            with open(json_path, "r") as f:
                predictions = json.load(f)

            allowed_labels = {"A", "B", "C", "na"}

            y_true = [p["true_answer"].strip().upper() if p["true_answer"].strip().upper() in allowed_labels else "na"
                      for p in predictions]
            y_pred = [p["predicted_letter"].strip().upper() if p["predicted_letter"].strip().upper() in allowed_labels else "na"
                      for p in predictions]

            experiment_name = file.replace(".json", "")
            metrics = evaluator.evaluate_discrete_answers(
                predictions=y_pred,
                ground_truth=y_true,
                experiment_name=experiment_name,
                folder=k_path
            )

            print(f"Evaluated {file} in {k_folder} — Accuracy: {metrics['accuracy']:.2%}")


k_values = []
accuracies = []

for k_dir in k_dirs:
    k_path = base_path / k_dir
    eval_file = k_path / "eval_scores.json"

    if not eval_file.exists():

        for sub in k_path.iterdir():
            if sub.is_dir():
                try_path = sub / "eval_scores.json"
                if try_path.exists():
                    eval_file = try_path
                    break

    if eval_file.exists():
        with open(eval_file, "r") as f:
            scores = json.load(f)
            k_val = int(k_dir.split("_")[1])
            k_values.append(k_val)
            accuracies.append(scores["accuracy"] * 100)


k_values, accuracies = zip(*sorted(zip(k_values, accuracies)))


plt.figure(figsize=(8, 5))
plt.bar(k_values, accuracies, width=1.2)
plt.title("Accuracy vs Number of Retrieved Contexts (k)", fontsize=14)
plt.xlabel("Number of Retrieved Contexts (k)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.ylim(40, 70)
plt.grid(axis="y")
plt.xticks(k_values)


plot_path = base_path / "accuracy_vs_k_bar.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved bar plot to {plot_path}")
