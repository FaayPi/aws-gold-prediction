"""
Compare all models (ARIMA, XGBoost) and automatically register/promote the best one.
This script implements the proper MLOps workflow:
1. Compare all models across experiments
2. Select the best model based on metrics
3. Register only the best model
4. Optionally promote to Production (with confirmation)
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import argparse

# Model configurations
MODEL_CONFIGS = {
    "ARIMA": {
        "experiment_name": "arima_gold_price",
        "model_name": "arima_gold_price_production",
        "register_script": "pipelines/mlflow_arima/register_arima_model_mlflow.py"
    },
    "XGBoost": {
        "experiment_name": "xgboost_gold_price",
        "model_name": "xgboost_gold_price_production",
        "register_script": "pipelines/mlflow_xgboost/register_xgboost_model_mlflow.py"
    }
}


def has_mlflow_model_format(client, run_id):
    """Check if run has MLflow Model Format (MLmodel file)."""
    try:
        artifacts = client.list_artifacts(run_id, path="model")
        has_mlmodel = any(a.path == "model/MLmodel" or a.path.endswith("/MLmodel") for a in artifacts)
        return has_mlmodel
    except Exception as e:
        return False


def get_best_run_for_experiment(client, experiment_name, model_type):
    """Get the best run from an experiment that has MLflow Model Format."""
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"   ⚠️  Experiment '{experiment_name}' not found")
        return None
    
    # Get top 20 runs sorted by RMSE
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse_next_day ASC"],
        max_results=20
    )
    
    if not runs:
        print(f"   ⚠️  No runs found in '{experiment_name}'")
        return None
    
    # Find first run with MLflow Model Format
    for run in runs:
        run_id = run.info.run_id
        if has_mlflow_model_format(client, run_id):
            metrics = run.data.metrics
            params = run.data.params
            
            return {
                "model_type": model_type,
                "run_id": run_id,
                "experiment_name": experiment_name,
                "rmse_day": metrics.get("rmse_next_day", float("inf")),
                "mae_day": metrics.get("mae_next_day", float("inf")),
                "mape_day": metrics.get("mape_next_day", float("inf")),
                "rmse_week": metrics.get("rmse_next_week", float("inf")),
                "mae_week": metrics.get("mae_next_week", float("inf")),
                "mape_week": metrics.get("mape_next_week", float("inf")),
                "params": params
            }
    
    print(f"   ⚠️  No runs with MLflow Model Format found in '{experiment_name}'")
    return None


def get_all_valid_runs_for_experiment(client, experiment_name, model_type):
    """Get all valid runs from an experiment that have MLflow Model Format."""
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"   ⚠️  Experiment '{experiment_name}' not found")
        return []
    
    # Get all runs sorted by RMSE
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse_next_day ASC"],
        max_results=50
    )
    
    if not runs:
        print(f"   ⚠️  No runs found in '{experiment_name}'")
        return []
    
    # Filter runs with MLflow Model Format
    valid_runs = []
    for run in runs:
        run_id = run.info.run_id
        if has_mlflow_model_format(client, run_id):
            metrics = run.data.metrics
            params = run.data.params
            
            valid_runs.append({
                "model_type": model_type,
                "run_id": run_id,
                "experiment_name": experiment_name,
                "rmse_day": metrics.get("rmse_next_day", float("inf")),
                "mae_day": metrics.get("mae_next_day", float("inf")),
                "mape_day": metrics.get("mape_next_day", float("inf")),
                "rmse_week": metrics.get("rmse_next_week", float("inf")),
                "mae_week": metrics.get("mae_next_week", float("inf")),
                "mape_week": metrics.get("mape_next_week", float("inf")),
                "params": params
            })
    
    return valid_runs


def compare_all_models(show_all_runs=False):
    """Compare all models and return comparison results.
    
    Args:
        show_all_runs: If True, show all runs per model type. If False, only show best per type.
    """
    client = MlflowClient()
    
    print("🔍 Comparing all models...\n")
    
    all_results = []
    best_per_type = []
    
    for model_type, config in MODEL_CONFIGS.items():
        print(f"📊 Checking {model_type} models...")
        valid_runs = get_all_valid_runs_for_experiment(
            client, 
            config["experiment_name"], 
            model_type
        )
        
        if valid_runs:
            all_results.extend(valid_runs)
            
            # Get best run for this model type
            best_run = min(valid_runs, key=lambda x: x['rmse_day'])
            best_per_type.append(best_run)
            
            print(f"   ✓ Found {len(valid_runs)} valid {model_type} run(s)")
            print(f"   ✓ Best {model_type} run: {best_run['run_id'][:8]}... "
                  f"(RMSE Day: {best_run['rmse_day']:.3f})")
            
            # Show top 3 if show_all_runs
            if show_all_runs and len(valid_runs) > 1:
                print(f"   Top {model_type} runs:")
                for i, run in enumerate(valid_runs[:3], 1):
                    print(f"     {i}. Run {run['run_id'][:8]}... "
                          f"RMSE: {run['rmse_day']:.3f}, "
                          f"MAE: {run['mae_day']:.3f}")
        else:
            print(f"   ✗ No valid {model_type} model found")
    
    if not all_results:
        print("\n❌ No valid models found to compare")
        return None
    
    # Use best per type for final comparison
    results = best_per_type
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON RESULTS")
    print("=" * 80)
    print("\nNext Day Predictions:")
    print(df[["model_type", "rmse_day", "mae_day", "mape_day"]].to_string(index=False))
    
    print("\nNext Week Predictions:")
    print(df[["model_type", "rmse_week", "mae_week", "mape_week"]].to_string(index=False))
    
    # Select best model based on RMSE (next day) - primary metric
    best_idx = df["rmse_day"].idxmin()
    best_model = df.loc[best_idx].to_dict()
    
    print("\n" + "=" * 80)
    print("🏆 BEST MODEL SELECTED")
    print("=" * 80)
    print(f"Model Type: {best_model['model_type']}")
    print(f"Run ID: {best_model['run_id']}")
    print(f"RMSE (Next Day): {best_model['rmse_day']:.3f}")
    print(f"MAE (Next Day): {best_model['mae_day']:.3f}")
    print(f"MAPE (Next Day): {best_model['mape_day']:.2f}%")
    print(f"RMSE (Next Week): {best_model['rmse_week']:.3f}")
    print(f"MAE (Next Week): {best_model['mae_week']:.3f}")
    
    # Show parameters
    print(f"\nParameters:")
    for key, value in best_model['params'].items():
        print(f"  {key}: {value}")
    
    return best_model


def register_best_model(best_model_info, auto_promote=False):
    """Register the best model and optionally promote to Production."""
    model_type = best_model_info["model_type"]
    run_id = best_model_info["run_id"]
    config = MODEL_CONFIGS[model_type]
    model_name = config["model_name"]
    
    client = MlflowClient()
    
    print("\n" + "=" * 80)
    print("📝 REGISTERING BEST MODEL")
    print("=" * 80)
    
    # Check if model already exists
    try:
        existing_versions = client.search_model_versions(f"name='{model_name}'")
        if existing_versions:
            print(f"⚠️  Model '{model_name}' already exists with {len(existing_versions)} version(s)")
            print("   Existing versions will be archived when new version is promoted to Production")
    except:
        pass
    
    # Register model
    try:
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, model_name)
        
        print(f"✅ Model registered: {model_name}")
        
        # Get latest version
        model_versions = client.search_model_versions(f"name='{model_name}'")
        if model_versions:
            latest_version = max(model_versions, key=lambda v: int(v.version))
            version = latest_version.version
            print(f"   Version: {version}")
            print(f"   Run ID: {latest_version.run_id}")
            
            # Promote to Production if requested (using alias)
            if auto_promote:
                print("\n🚀 Setting Production alias...")
                client.set_registered_model_alias(
                    name=model_name,
                    alias="Production",
                    version=str(version)
                )
                
                # Remove Production alias from previous version if exists
                model = client.get_registered_model(model_name)
                if "Production" in model.aliases:
                    old_prod_version = model.aliases["Production"]
                    if old_prod_version != str(version):
                        # Delete old alias (it's automatically replaced, but we can be explicit)
                        try:
                            client.delete_registered_model_alias(model_name, "Production")
                            client.set_registered_model_alias(model_name, "Production", str(version))
                            print(f"   📦 Moved Production alias from v{old_prod_version} to v{version}")
                        except:
                            pass
                
                print(f"✅ Model {model_name} v{version} marked as Production (using alias)")
            else:
                print(f"\n💡 To set Production alias, run:")
                if model_type == "ARIMA":
                    print(f"   python pipelines/mlflow_arima/promote_arima_model_mlflow.py --version {version}")
                else:
                    print(f"   python pipelines/mlflow_xgboost/promote_xgboost_model_mlflow.py --version {version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare all models and register/promote the best one"
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically promote best model to Production (default: False)"
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare models, don't register (default: False)"
    )
    parser.add_argument(
        "--show-all-runs",
        action="store_true",
        help="Show all runs per model type, not just the best (default: False)"
    )
    args = parser.parse_args()
    
    # Compare all models
    best_model = compare_all_models(show_all_runs=args.show_all_runs)
    
    if not best_model:
        print("\n❌ No valid models found. Please train models first.")
        return
    
    # Register and optionally promote
    if not args.compare_only:
        if args.auto_promote:
            register_best_model(best_model, auto_promote=True)
        else:
            response = input("\n❓ Register this model? (yes/no): ")
            if response.lower() in ["yes", "y"]:
                promote_response = input("❓ Promote to Production? (yes/no): ")
                register_best_model(best_model, auto_promote=(promote_response.lower() in ["yes", "y"]))
            else:
                print("❌ Registration cancelled")
    else:
        print("\n💡 Comparison only mode - no registration performed")
        print(f"   To register manually, run:")
        if best_model["model_type"] == "ARIMA":
            print(f"   python pipelines/mlflow_arima/register_arima_model_mlflow.py")
        else:
            print(f"   python pipelines/mlflow_xgboost/register_xgboost_model_mlflow.py")


if __name__ == "__main__":
    main()

