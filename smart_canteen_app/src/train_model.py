"""
Model Training & Evaluation Module for SmartCanteen System
Splits data chronologically, trains multiple ML regressors (Linear Regression,
Decision Tree, Random Forest, XGBoost), evaluates metrics (MAE, RMSE, R²),
and saves the best model along with metadata.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Optional XGBoost import with graceful fallback
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.feature_engineering import FEATURE_COLUMNS, create_features


def train_and_evaluate_models(
    df_clean,
    test_ratio=0.20,
    save_path=None
):
    """
    Orchestrates feature engineering, chronological train-test split,
    model training across 4 candidate algorithms, evaluation, and serialization.
    
    Args:
        df_clean (pd.DataFrame): Preprocessed canteen dataframe.
        test_ratio (float): Ratio of most recent data to reserve for testing.
        save_path (str, optional): Target file path for the best model pkl.
        
    Returns:
        dict: Training summary containing:
            - 'results_df': Model comparison table (MAE, RMSE, R²)
            - 'best_model_name': Name of best model
            - 'best_model': Trained model instance
            - 'encoders': Label mappings dictionary
            - 'feature_names': List of feature column names
            - 'feature_importances': Feature importance series or None
            - 'test_actual': Actual test y
            - 'test_pred': Best model test predictions
    """
    # 1. Feature Engineering
    df_feat, encoders = create_features(df_clean)
    
    # 2. Chronological Time-based Train/Test Split
    # Since it's time-series forecasting, earlier dates are used for training,
    # and the most recent dates are reserved for testing.
    df_feat = df_feat.sort_values(by="Date").reset_index(drop=True)
    
    split_idx = int(len(df_feat) * (1 - test_ratio))
    split_date = df_feat.iloc[split_idx]["Date"]
    
    train_mask = df_feat["Date"] < split_date
    test_mask = df_feat["Date"] >= split_date
    
    X_train = df_feat.loc[train_mask, FEATURE_COLUMNS]
    y_train = df_feat.loc[train_mask, "Quantity_Sold"]
    
    X_test = df_feat.loc[test_mask, FEATURE_COLUMNS]
    y_test = df_feat.loc[test_mask, "Quantity_Sold"]
    
    # 3. Model Dictionary
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1)
    }
    
    if XGB_AVAILABLE:
        models["XGBoost Regressor"] = xgb.XGBRegressor(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            random_state=42,
            n_jobs=-1
        )
        
    # 4. Train and Evaluate each candidate
    comparison_records = []
    trained_instances = {}
    test_predictions = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        trained_instances[name] = model
        test_predictions[name] = y_pred
        
        comparison_records.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2_Score": round(r2, 4)
        })
        
    results_df = pd.DataFrame(comparison_records).sort_values(by="R2_Score", ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_instances[best_model_name]
    
    # 5. Extract Feature Importance (for tree-based models or coefficients for linear)
    feature_importances = None
    if hasattr(best_model, "feature_importances_"):
        feature_importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    elif hasattr(best_model, "coef_"):
        feature_importances = pd.Series(np.abs(best_model.coef_), index=FEATURE_COLUMNS).sort_values(ascending=False)
        
    # 6. Save Best Model Bundle
    save_bundle = {
        "best_model_name": best_model_name,
        "model": best_model,
        "encoders": encoders,
        "feature_names": FEATURE_COLUMNS,
        "results_df": results_df,
        "feature_importances": feature_importances,
        "trained_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(save_bundle, save_path)
        print(f"Best model ({best_model_name}) saved successfully at: {save_path}")
        
    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "encoders": encoders,
        "feature_names": FEATURE_COLUMNS,
        "feature_importances": feature_importances,
        "test_actual": y_test,
        "test_pred": test_predictions[best_model_name],
        "split_date": split_date
    }


if __name__ == "__main__":
    from src.data_preprocessing import clean_canteen_data
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(project_root, "data", "sample_canteen_data.csv")
    model_path = os.path.join(project_root, "models", "food_demand_model.pkl")
    
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        cleaned = clean_canteen_data(raw_df)
        res = train_and_evaluate_models(cleaned, save_path=model_path)
        print("\nModel Comparison Results:")
        print(res["results_df"])
    else:
        print("Data file not found. Run data_generator.py first.")
