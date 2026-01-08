"""
Quick script to set Production alias for the best model.
Compares ARIMA and XGBoost and sets Production alias for the best one.
"""

from mlflow.tracking import MlflowClient
import mlflow

def get_best_model():
    """Get the best model based on RMSE."""
    client = MlflowClient()
    
    models_to_check = [
        ("arima_gold_price_production", "arima_gold_price"),
        ("xgboost_gold_price_production", "xgboost_gold_price")
    ]
    
    best_model = None
    best_rmse = float("inf")
    
    for model_name, experiment_name in models_to_check:
        try:
            # Get registered model versions
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"⚠️  No versions found for {model_name}")
                continue
            
            # Get experiment to find best run
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.rmse_next_day ASC"],
                    max_results=1
                )
                if runs:
                    run = runs[0]
                    rmse = run.data.metrics.get("rmse_next_day", float("inf"))
                    
                    # Find corresponding version
                    for v in versions:
                        if v.run_id == run.info.run_id:
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_model = {
                                    "model_name": model_name,
                                    "version": v.version,
                                    "run_id": run.info.run_id,
                                    "rmse": rmse
                                }
                            break
        except Exception as e:
            print(f"⚠️  Error checking {model_name}: {e}")
    
    return best_model

def set_production_alias(model_name, version):
    """Set Production alias for a model version."""
    client = MlflowClient()
    
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias="Production",
            version=str(version)
        )
        print(f"✅ Production alias set for {model_name} v{version}")
        return True
    except Exception as e:
        print(f"❌ Error setting alias: {e}")
        return False

def main():
    print("🔍 Finding best model...\n")
    
    best_model = get_best_model()
    
    if not best_model:
        print("❌ No valid models found")
        return
    
    print(f"🏆 Best Model: {best_model['model_name']}")
    print(f"   Version: {best_model['version']}")
    print(f"   RMSE (Next Day): {best_model['rmse']:.3f}")
    print(f"   Run ID: {best_model['run_id'][:8]}...")
    
    # Set Production alias
    print(f"\n🚀 Setting Production alias...")
    success = set_production_alias(best_model['model_name'], best_model['version'])
    
    if success:
        print(f"\n✅ Done! You can now use:")
        print(f"   models:/{best_model['model_name']}@Production")

if __name__ == "__main__":
    main()








