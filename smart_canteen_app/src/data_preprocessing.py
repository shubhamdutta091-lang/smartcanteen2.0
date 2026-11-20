"""
Data Preprocessing Module for SmartCanteen System
Cleans and validates the canteen dataset, handling types, null values,
deduplication, date parsing, and logical consistency.
"""

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "Date", "Day_of_Week", "Meal_Type", "Food_Item", "Food_Category",
    "Quantity_Prepared", "Quantity_Sold", "Leftover_Quantity",
    "Holiday_Flag", "Exam_Week_Flag", "Expected_Attendance",
    "Temperature_C", "Rainfall_mm", "Price_per_Item"
]


def validate_dataset_columns(df):
    """
    Checks if all required columns exist in the DataFrame.
    Returns (bool, list_of_missing_cols)
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return (len(missing) == 0, missing)


def clean_canteen_data(df):
    """
    Cleans raw canteen dataframe:
    - Checks required columns
    - Parses Date to datetime
    - Sorts records chronologically
    - Drops duplicates
    - Fills or computes missing Leftover_Quantity
    - Handles invalid / negative numeric inputs
    - Strips whitespace from strings
    
    Returns:
        pd.DataFrame: Cleaned and structured DataFrame
    """
    df_clean = df.copy()
    
    # Check columns
    valid, missing = validate_dataset_columns(df_clean)
    if not valid:
        # If Leftover_Quantity is missing but Quantity_Prepared and Quantity_Sold exist, we can compute it
        if set(missing) == {"Leftover_Quantity"} and "Quantity_Prepared" in df_clean.columns and "Quantity_Sold" in df_clean.columns:
            df_clean["Leftover_Quantity"] = df_clean["Quantity_Prepared"] - df_clean["Quantity_Sold"]
        else:
            raise ValueError(f"Dataset is missing required columns: {missing}")

    # Convert Date column to datetime
    df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
    df_clean = df_clean.dropna(subset=["Date"])
    
    # Sort chronologically
    df_clean = df_clean.sort_values(by=["Date", "Meal_Type", "Food_Item"]).reset_index(drop=True)
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates(subset=["Date", "Meal_Type", "Food_Item"], keep="first")
    
    # Clean text columns
    text_cols = ["Day_of_Week", "Meal_Type", "Food_Item", "Food_Category"]
    for col in text_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        
    # Standardize Day_of_Week if incorrect
    df_clean["Day_of_Week"] = df_clean["Date"].dt.day_name()
    
    # Numeric column validation and cleaning
    numeric_cols = [
        "Quantity_Prepared", "Quantity_Sold", "Holiday_Flag", 
        "Exam_Week_Flag", "Expected_Attendance", "Temperature_C", 
        "Rainfall_mm", "Price_per_Item"
    ]
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        
    # Fill remaining NaNs with appropriate defaults or medians
    df_clean["Holiday_Flag"] = df_clean["Holiday_Flag"].fillna(0).astype(int)
    df_clean["Exam_Week_Flag"] = df_clean["Exam_Week_Flag"].fillna(0).astype(int)
    df_clean["Rainfall_mm"] = df_clean["Rainfall_mm"].fillna(0.0).clip(lower=0.0)
    df_clean["Temperature_C"] = df_clean["Temperature_C"].fillna(df_clean["Temperature_C"].median()).clip(lower=-10.0, upper=60.0)
    df_clean["Expected_Attendance"] = df_clean["Expected_Attendance"].fillna(df_clean["Expected_Attendance"].median()).clip(lower=0)
    df_clean["Price_per_Item"] = df_clean["Price_per_Item"].fillna(df_clean["Price_per_Item"].median()).clip(lower=0.0)
    
    # Quantities must be positive integers
    df_clean["Quantity_Prepared"] = df_clean["Quantity_Prepared"].fillna(df_clean["Quantity_Prepared"].median()).clip(lower=0).round().astype(int)
    df_clean["Quantity_Sold"] = df_clean["Quantity_Sold"].fillna(df_clean["Quantity_Sold"].median()).clip(lower=0).round().astype(int)
    
    # Compute or correct Leftover_Quantity: formula Leftover_Quantity = Quantity_Prepared - Quantity_Sold
    df_clean["Leftover_Quantity"] = df_clean["Quantity_Prepared"] - df_clean["Quantity_Sold"]
    df_clean["Leftover_Quantity"] = df_clean["Leftover_Quantity"].clip(lower=0)
    
    return df_clean
