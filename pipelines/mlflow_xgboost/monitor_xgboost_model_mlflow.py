"""
Monitor XGBoost model performance over time.
Compares current metrics with historical values and alerts on degradation.
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import sys

EXPERIMENT_NAME = "xgboost_gold_price"
MODEL_NAME = "xgboost_gold_price_production"
DEGRADATION_THRESHOLD_PCT = 10  # Alert if RMSE increases by more than 10%


def get_model_performance_history():
    """Get all runs for the experiment."""
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found")
        return None
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    
    metrics = []
    for run in runs:
        metrics.append({
            'run_id': run.info.run_id,
            'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
            'rmse_day': run.data.metrics.get('rmse_next_day'),
            'mae_day': run.data.metrics.get('mae_next_day'),
            'mape_day': run.data.metrics.get('mape_next_day'),
            'rmse_week': run.data.metrics.get('rmse_next_week'),
            'mae_week': run.data.metrics.get('mae_next_week'),
            'mape_week': run.data.metrics.get('mape_next_week'),
            'n_estimators': run.data.params.get('n_estimators'),
            'max_depth': run.data.params.get('max_depth'),
            'learning_rate': run.data.params.get('learning_rate'),
        })
    
    return pd.DataFrame(metrics)


def check_performance_degradation(df, threshold_pct=DEGRADATION_THRESHOLD_PCT):
    """Check if performance has degraded."""
    if df is None or len(df) < 2:
        return None
    
    latest_rmse = df.iloc[0]['rmse_day']
    previous_rmse = df.iloc[1]['rmse_day']
    
    if pd.isna(latest_rmse) or pd.isna(previous_rmse):
        return None
    
    degradation = ((latest_rmse - previous_rmse) / previous_rmse) * 100
    
    if degradation > threshold_pct:
        return {
            'alert': True,
            'message': f"⚠️ Performance degradation detected: {degradation:.2f}%",
            'latest_rmse': latest_rmse,
            'previous_rmse': previous_rmse,
            'degradation_pct': degradation
        }
    
    return {'alert': False, 'degradation_pct': degradation}


def main():
    """Main monitoring function."""
    print("=" * 60)
    print("MLflow XGBoost Model Performance Monitoring")
    print("=" * 60)
    
    df = get_model_performance_history()
    
    if df is None or len(df) == 0:
        print("❌ No runs found in experiment")
        sys.exit(1)
    
    print(f"\n📊 Model Performance History (Last 10 Runs):")
    print("-" * 60)
    display_cols = ['start_time', 'rmse_day', 'mae_day', 'mape_day', 
                   'n_estimators', 'max_depth', 'learning_rate']
    print(df[display_cols].head(10).to_string(index=False))
    
    # Check for performance degradation
    alert = check_performance_degradation(df)
    
    if alert:
        if alert['alert']:
            print(f"\n{alert['message']}")
            print(f"   Latest RMSE: {alert['latest_rmse']:.3f}")
            print(f"   Previous RMSE: {alert['previous_rmse']:.3f}")
            print(f"   Degradation: {alert['degradation_pct']:.2f}%")
            print("\n⚠️  RECOMMENDATION: Investigate model performance and consider retraining")
            sys.exit(1)  # Exit with error code for alerting systems
        else:
            print(f"\n✅ Performance stable")
            print(f"   Change: {alert['degradation_pct']:.2f}%")
    
    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"   Total Runs: {len(df)}")
    print(f"   Best RMSE (Next Day): {df['rmse_day'].min():.3f}")
    print(f"   Worst RMSE (Next Day): {df['rmse_day'].max():.3f}")
    print(f"   Average RMSE (Next Day): {df['rmse_day'].mean():.3f}")
    print(f"   Latest RMSE (Next Day): {df.iloc[0]['rmse_day']:.3f}")
    
    print("\n✅ Monitoring completed successfully")


if __name__ == "__main__":
    main()

