"""MLflow experiment tracking wrapper."""

import mlflow
from datetime import datetime
from pathlib import Path

class ExperimentTracker:
    def __init__(self, experiment_name="cardiometabolic_risk"):
        tracking_uri = Path('logs/mlruns')
        tracking_uri.mkdir(parents=True, exist_ok=True)
        
        mlflow.set_tracking_uri(str(tracking_uri))
        mlflow.set_experiment(experiment_name)
        self.run = None
    
    def start_run(self, run_name=None):
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = mlflow.start_run(run_name=run_name)
        return self.run
    
    def log_params(self, params):
        mlflow.log_params(params)
    
    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, file_path):
        mlflow.log_artifact(file_path)
    
    def end_run(self):
        if self.run:
            mlflow.end_run()