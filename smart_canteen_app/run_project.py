"""Entry point for the SmartCanteen backend workflow.

This script performs the full pipeline without any UI:
1️⃣ Load or generate the synthetic dataset.
2️⃣ Clean and validate the data.
3️⃣ Train the demand‑forecast model (or load a saved model).
4️⃣ Predict tomorrow's demand.
5️⃣ Compute waste metrics and export a CSV report.
"""

import os
import sys
import pandas as pd
import joblib

# -------------------------------------------------------------------
# Ensure the repository root is on the Python path
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import project modules
from src.data_generator import generate_canteen_dataset
from src.data_preprocessing import clean_canteen_data
from src.train_model import train_and_evaluate_models
from src.predict import load_trained_model, predict_demand
from src.waste_calculator import calculate_waste_metrics

def main():
    # Paths for data, model, and report
    data_path = os.path.join(PROJECT_ROOT, "data", "sample_canteen_data.csv")
    model_path = os.path.join(PROJECT_ROOT, "models", "food_demand_model.pkl")
    report_path = os.path.join(PROJECT_ROOT, "reports", "prediction_report.csv")

    # ---------------------------------------------------------------
    # 1️⃣ Load or generate dataset
    # ---------------------------------------------------------------
    if not os.path.exists(data_path):
        print("Dataset not found – generating synthetic data...")
        df = generate_canteen_dataset(output_path=data_path)
    else:
        print(f"Loading existing dataset from {data_path}")
        df = pd.read_csv(data_path)

    # ---------------------------------------------------------------
    # 2️⃣ Clean the data
    # ---------------------------------------------------------------
    df_clean = clean_canteen_data(df)
    print(f"Cleaned dataset – {len(df_clean)} rows ready for modelling")

    # ---------------------------------------------------------------
    # 3️⃣ Train or load model
    # ---------------------------------------------------------------
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}")
        model_bundle = load_trained_model(model_path)
    else:
        print("Training new model…")
        # Train the model and save the full bundle (includes model, encoders, etc.)
        model_bundle = train_and_evaluate_models(df_clean, save_path=model_path)
        print(f"Saved trained model bundle to {model_path}")
        # Save the best model for future runs
        best_model = model_bundle.get("best_model")
        if best_model is not None:
            joblib.dump(best_model, model_path)
            print(f"Saved trained model to {model_path}")

    # ---------------------------------------------------------------
    # 4️⃣ Predict tomorrow's demand
    # ---------------------------------------------------------------
    # Build a simple input dict from the most recent clean record
    last_row = df_clean.iloc[-1]
    input_dict = {
        "Date": (pd.to_datetime(last_row["Date"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "Meal_Type": last_row["Meal_Type"],
        "Food_Item": last_row["Food_Item"],
        "Food_Category": last_row["Food_Category"],
        "Expected_Attendance": last_row["Expected_Attendance"],
        "Holiday_Flag": int(last_row["Holiday_Flag"]),
        "Exam_Week_Flag": int(last_row["Exam_Week_Flag"]),
        "Temperature_C": float(last_row["Temperature_C"]),
        "Rainfall_mm": float(last_row["Rainfall_mm"]),
        "Price_per_Item": float(last_row["Price_per_Item"]),
        # placeholder lag/average features – the model can handle NaNs or defaults
        "Sales_Lag_1": int(last_row["Quantity_Sold"]),
        "Sales_Lag_7": int(last_row["Quantity_Sold"]),
        "Moving_Avg_7": float(last_row["Quantity_Sold"]),
        "Moving_Avg_14": float(last_row["Quantity_Sold"]),
        "Weekday_Avg_Sales": float(last_row["Quantity_Sold"]),
    }
    prediction = predict_demand(input_dict, model_bundle=model_bundle)
    print(f"Predicted demand for tomorrow: {prediction:.2f} units")


    # ---------------------------------------------------------------
    # 5️⃣ Compute waste metrics and export report
    # ---------------------------------------------------------------
    # Obtain engineered features for the whole cleaned dataframe
    from src.feature_engineering import create_features, FEATURE_COLUMNS
    df_feat, encoders = create_features(df_clean)
    # Align columns for the model (features only)
    X_all = df_feat[FEATURE_COLUMNS]
    # Predict demand for every record in a vectorised fashion
    predictions = model_bundle["model"].predict(X_all)
    # Attach predictions to the original cleaned dataframe
    df_clean["Predicted_Demand"] = predictions
    # Compute waste metrics row‑wise using the scalar helper
    waste_records = []
    for _, row in df_clean.iterrows():
        waste = calculate_waste_metrics(row["Predicted_Demand"], row["Quantity_Prepared"])
        waste_record = {
            "Date": row["Date"],
            "Meal_Type": row["Meal_Type"],
            "Food_Item": row["Food_Item"],
            "Quantity_Prepared": row["Quantity_Prepared"],
            "Predicted_Demand": row["Predicted_Demand"],
            **waste,
        }
        waste_records.append(waste_record)
    waste_df = pd.DataFrame(waste_records)
    waste_df.to_csv(report_path, index=False)
    print(f"Waste report written to {report_path}")

if __name__ == "__main__":
    main()
