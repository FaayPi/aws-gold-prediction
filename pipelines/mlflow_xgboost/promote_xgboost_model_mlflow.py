"""
Promote a registered XGBoost model to Production using Aliases (MLflow 2.0+).
"""

import argparse
from mlflow.tracking import MlflowClient

MODEL_NAME = "xgboost_gold_price_production"

def promote_model(version=None):
    """Promote model to Production using alias."""
    client = MlflowClient()
    
    if version is None:
        # Get latest version
        try:
            # Try to get current Production alias
            model = client.get_registered_model(MODEL_NAME)
            if "Production" in model.aliases:
                current_prod_version = model.aliases["Production"]
                print(f"Current Production alias: version {current_prod_version}")
            
            # Get latest version (without Production alias)
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if not versions:
                print(f"❌ No versions found for model '{MODEL_NAME}'")
                return
            
            # Get version without Production alias
            non_prod_versions = [v for v in versions if "Production" not in (v.aliases or {})]
            if non_prod_versions:
                version = non_prod_versions[0].version
            else:
                version = max(versions, key=lambda v: int(v.version)).version
            print(f"Using version: {version}")
        except Exception as e:
            print(f"⚠️  Error getting versions: {e}")
            return
    
    try:
        # Set Production alias (new MLflow 2.0+ method)
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="Production",
            version=str(version)
        )
        
        print(f"✅ Model {MODEL_NAME} v{version} marked as Production (using alias)")
        
        # Show model info
        model = client.get_registered_model(MODEL_NAME)
        if "Production" in model.aliases:
            print(f"   Production alias points to version: {model.aliases['Production']}")
        
    except Exception as e:
        print(f"❌ Error promoting model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote XGBoost model to Production stage")
    parser.add_argument("--version", type=int, default=None, 
                       help="Model version to promote (default: latest)")
    args = parser.parse_args()
    
    promote_model(args.version)

