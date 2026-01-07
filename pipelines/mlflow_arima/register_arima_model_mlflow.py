"""
Register the best ARIMA model in MLflow Model Registry.
Finds the run with the lowest RMSE and registers it for production use.
Only considers runs with proper MLflow Model Format.
"""

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "arima_gold_price"
MODEL_NAME = "arima_gold_price_production"

def has_mlflow_model_format(client, run_id):
    """Check if run has MLflow Model Format (MLmodel file)."""
    try:
        artifacts = client.list_artifacts(run_id, path="model")
        # Check if MLmodel file exists
        has_mlmodel = any(a.path == "model/MLmodel" or a.path.endswith("/MLmodel") for a in artifacts)
        return has_mlmodel
    except Exception as e:
        print(f"   ⚠️  Could not check artifacts for run {run_id}: {e}")
        return False

def register_best_model():
    """Register the best model based on RMSE (next day) that has MLflow Model Format."""
    client = MlflowClient()

    # Find experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found")
        return
    
    # Get top 20 runs sorted by RMSE (to find one with correct format)
    print(f"🔍 Searching for runs with MLflow Model Format...")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse_next_day ASC"],
        max_results=20
    )
    
    if not runs:
        print("❌ No runs found in experiment")
        return
    
    # Filter runs that have MLflow Model Format
    valid_runs = []
    for run in runs:
        run_id = run.info.run_id
        if has_mlflow_model_format(client, run_id):
            valid_runs.append(run)
            print(f"   ✓ Run {run_id[:8]}... has MLflow Model Format (RMSE: {run.data.metrics.get('rmse_next_day', 'N/A'):.3f})")
        else:
            print(f"   ✗ Run {run_id[:8]}... missing MLflow Model Format (skipping)")
    
    if not valid_runs:
        print("\n❌ No runs found with MLflow Model Format")
        print("   Please train a new model with:")
        print("   python pipelines/mlflow_arima/train_arima_mlflow.py --data-file gold_GCF_10y_1d.csv --experiment-name arima_gold_price")
        return
    
    # Select best run (already sorted by RMSE)
    best_run = valid_runs[0]
    run_id = best_run.info.run_id
    
    print(f"\n🏆 Best Run with MLflow Format: {run_id}")
    print(f"   RMSE (Next Day): {best_run.data.metrics['rmse_next_day']:.3f}")
    print(f"   MAE (Next Day): {best_run.data.metrics['mae_next_day']:.3f}")
    print(f"   MAPE (Next Day): {best_run.data.metrics['mape_next_day']:.2f}%")
    print(f"   Parameters: p={best_run.data.params.get('arima_p')}, "
          f"d={best_run.data.params.get('arima_d')}, "
          f"q={best_run.data.params.get('arima_q')}")
    
    # Register model
    try:
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, MODEL_NAME)
        
        print(f"\n✅ Model registered as: {MODEL_NAME}")
        
        # Get latest version (using modern API)
        model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if model_versions:
            # Sort by version number (descending) and get the latest
            latest_version = max(model_versions, key=lambda v: int(v.version))
            print(f"   Version: {latest_version.version}")
            print(f"   Run ID: {latest_version.run_id}")
        else:
            print(f"   Version: 1 (newly created)")
            print(f"   Run ID: {run_id}")
        
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    register_best_model()

