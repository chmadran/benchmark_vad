from collections import defaultdict
import pandas as pd
import numpy as np
from tabulate import tabulate

class PostProcessor:
    def __init__(self, models: list, results: list):
        """
        Args:
            models (list): List of models that were benchmarked.
            results (list): List of result dictionaries from benchmarking.
        """
        self.models = models
        self.results = results
    
    
    def mean_round_df(self, df: pd.DataFrame, threshold: int = 2) -> pd.DataFrame:
        """
        Round the values in a DataFrame containing mean evaluation metrics.

        Args:
            df (pd.DataFrame): DataFrame with mean scores for each model.
            threshold (int): Number of decimal places to round to (default is 2).

        Returns:
            pd.DataFrame: Rounded DataFrame.
        """
        return df.round({"F1_mean": threshold, "Precision_mean": threshold, "Recall_mean": threshold})


    def round_df(self, df: pd.DataFrame, threshold: int = 2) -> pd.DataFrame:
        """
        Round the values in a DataFrame containing individual evaluation metrics.

        Args:
            df (pd.DataFrame): DataFrame with individual model results.
            threshold (int): Number of decimal places to round to (default is 2).

        Returns:
            pd.DataFrame: Rounded DataFrame.
        """
        return df.round({"F1": threshold, "Precision": threshold, "Recall": threshold})

    def print_experiment_hyperparameters(self, model: str, id: int):
        """
        Print the hyperparameters and other details of a specific experiment run
        for a given model and experiment ID.

        Args:
            model (str): The name of the model to filter results by.
            id (int): The experiment ID to filter results by.

        Prints:
            A formatted table displaying the hyperparameters and metrics
            for the first matching experiment run.
        """
        for result in self.results:
            if result["model"] == model and result["experiment_id"] == id:
                print(f"\nPRINTING EXPERIMENT {id} for {model.upper()}\n")
                df = pd.DataFrame([result])
                print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
                return
        print(f"\nEXPERIMENT {id} for {model.upper()} not found.\n")

    
    def print_mean_results_per_experiment(self) -> pd.DataFrame:
        """
        Compute and display the mean F1, Precision, and Recall for each (model, experiment_id) pair.

        Returns:
            pd.DataFrame: DataFrame containing mean metrics per (model, experiment_id).
        """

        grouped_results = defaultdict(list)
        for result in self.results:
            model = result["model"]
            experiment_id = result["experiment_id"]
            grouped_results[(model, experiment_id)].append(result)

        records = []
        for (model, exp_id), results in grouped_results.items():
            f1s = [r["F1"] for r in results]
            ps = [r["Precision"] for r in results]
            rs = [r["Recall"] for r in results]

            records.append({
                "Model": model.upper(),
                "Experiment_ID": exp_id,
                "F1_mean": np.mean(f1s),
                "Precision_mean": np.mean(ps),
                "Recall_mean": np.mean(rs),
            })

        df = pd.DataFrame(records)
        df = df.sort_values(by=["Model", "Experiment_ID"]).reset_index(drop=True)

        print("\n\n")
        print(f"MEAN METRICS PER MODEL AND EXPERIMENT:\n")
        print(tabulate(self.mean_round_df(df, threshold=2), headers="keys", tablefmt="fancy_grid"))
        print("\n\n")

        return df


    def print_best_set_hyperparams(self, best_model: dict, metric: str) -> None:
        """
        Print the hyperparameters and evaluation metrics for a single best-performing model.

        Args:
            best_model (dict): Dictionary containing model configuration and evaluation metrics.
            metric (str): The metric used to select the best model (e.g., "F1").
        """
        df = pd.DataFrame([best_model]) 
        print("\n")
        print(f"Best hyperparameters for {best_model["model"].upper()} based on metric : {metric}\n")
        print(tabulate(self.round_df(df, threshold=2), headers="keys", tablefmt="fancy_grid"))
        print("\n")


    def compute_best_set_hyperparams(self, model: str, metric: str) -> pd.DataFrame:
        """
        Computes and display the hyperparameters and the metrics (F1, precision, and recall) for
            the best model of each model that was benchmarked based on its performance on 
            the specified metric.

        Args:
            metric(str): Metric to choose what "best" means.

        Returns:
            pd.DataFrame: DataFrame containing the hyperparameters and results based on the specified metric, per model.
        """   

        best_score = -1
        best_model = {}
        model_results = [r for r in self.results if r["model"] == model]
        for result in model_results:
            if result[metric] > best_score:
                best_score =  result[metric] 
                best_model = result
        return best_model


    def print_best_model(self, metric: str, model_specific: str = None) -> None:
        """
        Computes and prints the best hyperparameter set and corresponding metrics 
        (F1, Precision, Recall) for each model based on the specified metric.

        If a specific model is provided, only that model is processed. Otherwise, 
        all models in `self.models` are processed.

        Args:
            metric (str): The metric used to determine the best model (e.g., "F1", "Precision", "Recall").
            model_specific (str, optional): The name of a specific model to evaluate. Defaults to None.

        Returns:
            None
        """
        metric = metric.title() #in case i write the metric wrong, putting it back in CamelCases
        if model_specific:
            best_model = self.compute_best_set_hyperparams(model_specific, metric)
            if best_model:
                self.print_best_set_hyperparams(best_model, metric)
        else:
            for model in self.models:
                best_model = self.compute_best_set_hyperparams(model, metric)
                if best_model:
                    self.print_best_set_hyperparams(best_model, metric)
