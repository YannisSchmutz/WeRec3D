
import os
from pathlib import Path
import yaml
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = Path(SCRIPT_DIR).parent.absolute()

MLFLOW_DIR = str(SRC_DIR) + "/experiments_evaluation/mlruns/"


def get_experiment_metrics(experiment_name, additional_metrics=None):
    """
    Pulls the metric values of the last epoch for each run.

    :param experiment_name: Name of the desired experiment.
    :param additional_metrics: Additional metrics to consider.
    :return:
    """
    possible_experiments = os.listdir(MLFLOW_DIR)

    for pos_exp in possible_experiments:
        pos_exp_path = MLFLOW_DIR + pos_exp
        if os.path.isdir(pos_exp_path) and os.listdir(pos_exp_path) and os.path.exists(pos_exp_path + "/meta.yaml"):
            with open(pos_exp_path + "/meta.yaml", 'r') as fh:
                meta = yaml.safe_load(fh)
            if meta['name'] == experiment_name:
                print(f"Experiment '{experiment_name}' found!")
                break
    else:
        print(f"Could not find experiment '{experiment_name}'")
        return

    metrics_to_consider = ['loss', 'val_loss', 'masked_mae', 'val_masked_mae', 'mae', 'val_mae']
    if additional_metrics:
        metrics_to_consider = metrics_to_consider + additional_metrics

    run_metric_mapping = defaultdict(dict)
    runs = list(filter(lambda d: os.path.isdir(pos_exp_path + "/" + d), os.listdir(pos_exp_path)))
    for run in runs:
        run_path = f"{pos_exp_path}/{run}/"
        metrics_path = f"{run_path}metrics/"
        with open(run_path + "/meta.yaml", 'r') as fh:
            run_meta = yaml.safe_load(fh)
        run_name = run_meta['run_name']
        print(f"Run name: {run_name}")
        for metric in metrics_to_consider:
            with open(metrics_path + metric, "r") as fh:
                metric_values = fh.readlines()
                metric_values = list(map(lambda x: float(x.split(" ")[1]), metric_values))
            run_metric_mapping[run_name][metric] = metric_values
    return dict(run_metric_mapping)
