import mlflow
import os

import src.data_augmentation_time_series.main as ts_gan_augmentation
import src.data_augmentation_time_series.utils.inference as ts_gan_inference
import src.config_env


def train_model(config: dict):

    method = config.get('type', 'TTS_GAN')

    if method in ['TTS_GAN', 'MTS_GAN']:
        try:
            ts_gan_augmentation.main(config)
        except Exception as e:
            mlflow.log_param("error", str(e))
            mlflow.end_run(status="FAILED")
            raise ValueError(f"Failed training: {e}")
    else:
        raise ValueError(f"Data augmentation failed for model type: {method}")


def generate_samples(run_id: str, n_samples: int):

    syn_samples = ts_gan_inference.generate_synthetic_data(n_samples, os.getenv('MLFLOW_URI'), run_id)
    return syn_samples


def get_experiment_runs(experiment_name: str = None) -> dict:

    mlflow.set_tracking_uri(os.getenv('MLFLOW_URI', 'http://localhost:5005'))
    client = mlflow.tracking.MlflowClient()
    
    if experiment_name:
 
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found.")
        
        experiment_ids = [experiment.experiment_id]
    else:
 
        experiments = client.search_experiments()
        experiment_ids = [exp.experiment_id for exp in experiments]
    
    all_runs = []
    
    # experiment_id must be retrieved always in order to do the search
    for exp_id in experiment_ids:
        runs = client.search_runs(experiment_ids=[exp_id])
        for run in runs:
            all_runs.append({
                "experiment_name": client.get_experiment(exp_id).name,
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "metrics": {
                    "best_val_metric": run.data.metrics.get("best_val_metric", None),
                    "average_fid_score_temporal": run.data.metrics.get("average_fid_score_temporal", None),
                    "average_jensen_shannon_divergence_temporal": run.data.metrics.get("average_jensen_shannon_divergence_temporal", None),
                    "average_cosine_similarity_temporal": run.data.metrics.get("average_cosine_similarity_temporal", None)}
            })
    
    return all_runs

