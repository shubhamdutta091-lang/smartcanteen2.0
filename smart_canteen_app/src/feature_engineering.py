"""
Feature Engineering Module for SmartCanteen System
Creates calendar features, historical lag sales, moving averages, 
and encoding for machine learning models without data leakage.
"""

import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    "Year", "Month", "Day", "Day_of_Week_Num", "Weekend_Flag",
    "Holiday_Flag", "Exam_Week_Flag", "Expected_Attendance",
    "Temperature_C", "Rainfall_mm", "Price_per_Item",
    "Sales_Lag_1", "Sales_Lag_7", "Moving_Avg_7", "Moving_Avg_14", "Weekday_Avg_Sales",
    "Meal_Type_Code", "Food_Item_Code", "Food_Category_Code"
]


def create_features(df, encoders=None):
    """
    Transforms preprocessed dataframe into ML feature set.
    Includes calendar features, lags, rolling averages, and categorical codes.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe sorted by date.
        encoders (dict, optional): Dictionary with label mappings for inference.
        
    Returns:
        df_feat (pd.DataFrame): DataFrame with engineered features.
        encoders (dict): Dictionary with category mappings.
    """
    df = df.copy()
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])
    
    df = df.sort_values(by=["Food_Item", "Date"]).reset_index(drop=True)
    
    # 1. Calendar Features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Day_of_Week_Num"] = df["Date"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["Weekend_Flag"] = df["Day_of_Week_Num"].apply(lambda x: 1 if x in [5, 6] else 0)
    
    # 2. Historical Lag Features and Moving Averages per Food_Item
    # Group by Food_Item and calculate strictly past stats using shift to prevent data leakage
    df["Sales_Lag_1"] = df.groupby("Food_Item")["Quantity_Sold"].shift(1)
    df["Sales_Lag_7"] = df.groupby("Food_Item")["Quantity_Sold"].shift(7)
    
    # 7-day and 14-day rolling averages of past sales (shift by 1 so current day isn't included)
    df["Moving_Avg_7"] = (
        df.groupby("Food_Item")["Quantity_Sold"]
        .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    )
    df["Moving_Avg_14"] = (
        df.groupby("Food_Item")["Quantity_Sold"]
        .transform(lambda x: x.shift(1).rolling(window=14, min_periods=1).mean())
    )
    
    # Same-weekday historical average sales for each item
    # Calculate cumulative mean of previous matching weekdays for that item
    df["Weekday_Avg_Sales"] = (
        df.groupby(["Food_Item", "Day_of_Week_Num"])["Quantity_Sold"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    
    # Handle initial NaNs from lag operations gracefully using expanding mean or overall item mean
    item_means = df.groupby("Food_Item")["Quantity_Sold"].transform("mean")
    df["Sales_Lag_1"] = df["Sales_Lag_1"].fillna(item_means)
    df["Sales_Lag_7"] = df["Sales_Lag_7"].fillna(item_means)
    df["Moving_Avg_7"] = df["Moving_Avg_7"].fillna(item_means)
    df["Moving_Avg_14"] = df["Moving_Avg_14"].fillna(item_means)
    df["Weekday_Avg_Sales"] = df["Weekday_Avg_Sales"].fillna(item_means)
    
    # 3. Categorical Encoders
    if encoders is None:
        meal_types = sorted(df["Meal_Type"].dropna().unique().tolist())
        food_items = sorted(df["Food_Item"].dropna().unique().tolist())
        food_cats = sorted(df["Food_Category"].dropna().unique().tolist())
        
        encoders = {
            "Meal_Type": {val: idx for idx, val in enumerate(meal_types)},
            "Food_Item": {val: idx for idx, val in enumerate(food_items)},
            "Food_Category": {val: idx for idx, val in enumerate(food_cats)}
        }
        
    df["Meal_Type_Code"] = df["Meal_Type"].map(encoders["Meal_Type"]).fillna(0).astype(int)
    df["Food_Item_Code"] = df["Food_Item"].map(encoders["Food_Item"]).fillna(0).astype(int)
    df["Food_Category_Code"] = df["Food_Category"].map(encoders["Food_Category"]).fillna(0).astype(int)
    
    # Ensure all required features are non-null and numeric
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
    # Re-sort chronologically by Date
    df = df.sort_values(by=["Date", "Meal_Type", "Food_Item"]).reset_index(drop=True)
    
    return df, encoders


def prepare_single_prediction_features(input_dict, encoders):
    """
    Constructs a 1-row DataFrame strictly matching FEATURE_COLUMNS for live prediction.
    
    Args:
        input_dict (dict): Keys matching input form
        encoders (dict): Trained categorical mappings
        
    Returns:
        pd.DataFrame: 1-row dataframe ready for model.predict()
    """
    pred_date = pd.to_datetime(input_dict["Date"])
    dow_num = pred_date.dayofweek
    is_weekend = 1 if dow_num in [5, 6] else 0
    
    meal_type = str(input_dict.get("Meal_Type", "")).strip()
    food_item = str(input_dict.get("Food_Item", "")).strip()
    food_cat = str(input_dict.get("Food_Category", "")).strip()
    
    meal_code = encoders["Meal_Type"].get(meal_type, 0)
    item_code = encoders["Food_Item"].get(food_item, 0)
    cat_code = encoders["Food_Category"].get(food_cat, 0)
    
    row = {
        "Year": pred_date.year,
        "Month": pred_date.month,
        "Day": pred_date.day,
        "Day_of_Week_Num": dow_num,
        "Weekend_Flag": is_weekend,
        "Holiday_Flag": int(input_dict.get("Holiday_Flag", 0)),
        "Exam_Week_Flag": int(input_dict.get("Exam_Week_Flag", 0)),
        "Expected_Attendance": float(input_dict.get("Expected_Attendance", 1000)),
        "Temperature_C": float(input_dict.get("Temperature_C", 25.0)),
        "Rainfall_mm": float(input_dict.get("Rainfall_mm", 0.0)),
        "Price_per_Item": float(input_dict.get("Price_per_Item", 50.0)),
        "Sales_Lag_1": float(input_dict.get("Sales_Lag_1", 150.0)),
        "Sales_Lag_7": float(input_dict.get("Sales_Lag_7", 150.0)),
        "Moving_Avg_7": float(input_dict.get("Moving_Avg_7", 150.0)),
        "Moving_Avg_14": float(input_dict.get("Moving_Avg_14", 150.0)),
        "Weekday_Avg_Sales": float(input_dict.get("Weekday_Avg_Sales", 150.0)),
        "Meal_Type_Code": meal_code,
        "Food_Item_Code": item_code,
        "Food_Category_Code": cat_code
    }
    
    df_single = pd.DataFrame([row])[FEATURE_COLUMNS]
    return df_single
