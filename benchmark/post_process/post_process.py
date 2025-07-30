import pandas as pd
import numpy as np
from tabulate import tabulate

class PostProcessor:
    def __init__(self, models: list, results: list):
        """
        Args:
            models (list): List of models that were benchmarked.
            results (dict): List of result as dict per model following benchmark.
        """
        self.models = models
        self.results = results

        def compare_model_metrics(self, models: list, all_results: list):
            mean_data = {}

            for model in models:
                model_results = [r for r in all_results if r["model"] == model]

                if not model_results:
                    continue 

                f1_scores = [r["f1"] for r in model_results]
                precision_scores = [r["precision"] for r in model_results]
                recall_scores = [r["recall"] for r in model_results]

                mean_data[model] = {
                    "f1_mean": np.mean(f1_scores),
                    "precision_mean": np.mean(precision_scores),
                    "recall_mean": np.mean(recall_scores),
                }

            df = pd.DataFrame(mean_data).T
            return df


        def round_df(df):
            return df.round({"f1_mean": 2, "precision_mean": 2, "recall_mean": 2})

        def print_output(self, models: list, results: list):
            all_df = compare_model_metrics(models, results)
            print("\nMEAN METRICS FOR ALL PREDICTIONS:\n")
            print(tabulate(round_df(all_df), headers="keys", tablefmt="pretty"))


        def print_best_models(self, models: list, best_models: dict):
            best_results = list(best_models.values())
            best_df = compare_model_metrics(models, best_results)
            print("\nBEST MODEL METRICS:\n")
            print(tabulate(round_df(best_df), headers="keys", tablefmt="pretty"))

            print("\nBEST MODEL CONFIGS (based on f1):\n")
            for model, config in best_models.items():
                print(f"\n→ {model.upper()}")
                for k, v in config.items():
                    if k not in {"preds_ms", "confidence", "preds_s"}:
                        print(f"  {k}: {v}")
