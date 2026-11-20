"""
Prediction Module for SmartCanteen System
Loads serialized trained model bundle and generates demand prediction
and preparation recommendations for future dates.
"""

import os
import joblib
import pandas as pd
from src.feature_engineering import prepare_single_prediction_features


def load_trained_model(model_path="models/food_demand_model.pkl"):
    """
    Loads model bundle from disk.
    
    Returns:
        dict: bundle containing model, encoders, feature_names, etc.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please train the model first.")
        
    bundle = joblib.load(model_path)
    return bundle


def predict_demand(input_dict, model_bundle=None, model_path="models/food_demand_model.pkl"):
    """
    Predicts food demand for given operational parameters.
    
    Args:
        input_dict (dict): Dictionary with user inputs:
            - Date, Meal_Type, Food_Item, Food_Category, Expected_Attendance,
              Holiday_Flag, Exam_Week_Flag, Temperature_C, Rainfall_mm,
              Price_per_Item, Sales_Lag_1, Sales_Lag_7, Moving_Avg_7,
              Moving_Avg_14, Weekday_Avg_Sales.
        model_bundle (dict, optional): Pre-loaded bundle.
        model_path (str, optional): Path to saved pkl file if bundle not passed.
        
    Returns:
        float: Predicted demand (quantity sold) rounded to integer
    """
    if model_bundle is None:
        model_bundle = load_trained_model(model_path)
        
    model = model_bundle["model"]
    encoders = model_bundle["encoders"]
    
    # Feature preparation matching training exactly
    X_single = prepare_single_prediction_features(input_dict, encoders)
    
    # Run prediction
    pred_val = model.predict(X_single)[0]
    
    # Floor to reasonable minimum (cannot sell negative items)
    predicted_demand = max(5, int(round(pred_val)))
    return predicted_demand
