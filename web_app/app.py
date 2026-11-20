"""
SmartCanteen - Production Web Server Application
FastAPI Server powering the full frontend website and REST APIs
Includes CSV file uploads, sample generation, single-record and new food item entry,
model training, demand inference, analytics charts, and downloadable reports.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
APP_SRC = os.path.join(PROJECT_ROOT, "smart_canteen_app")
if APP_SRC not in sys.path:
    sys.path.insert(0, APP_SRC)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from smart_canteen_app.src.predict import load_trained_model, predict_demand
from smart_canteen_app.src.waste_calculator import calculate_waste_metrics
from smart_canteen_app.src.train_model import train_and_evaluate_models
from smart_canteen_app.src.data_preprocessing import clean_canteen_data
from smart_canteen_app.src.data_generator import generate_canteen_dataset

app = FastAPI(title="SmartCanteen Web Application")

# Static files mount
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

DATA_PATH = os.path.join(APP_SRC, "data", "sample_canteen_data.csv")
MODEL_PATH = os.path.join(APP_SRC, "models", "food_demand_model.pkl")
REPORT_PATH = os.path.join(APP_SRC, "reports", "prediction_report.csv")

# Global in-memory cache for trained model
try:
    model_bundle = load_trained_model(MODEL_PATH)
except Exception:
    model_bundle = None


class PredictionInput(BaseModel):
    Date: str
    Meal_Type: str
    Food_Item: str
    Food_Category: str
    Expected_Attendance: float
    Holiday_Flag: int
    Exam_Week_Flag: int
    Temperature_C: float
    Rainfall_mm: float
    Price_per_Item: float
    Usual_Preparation: int
    Sales_Lag_1: float
    Sales_Lag_7: float
    Moving_Avg_7: float
    Moving_Avg_14: float
    Weekday_Avg_Sales: float


class NewFoodDataInput(BaseModel):
    Date: str
    Meal_Type: str
    Food_Item: str
    Food_Category: str
    Quantity_Prepared: int
    Quantity_Sold: int
    Holiday_Flag: int
    Exam_Week_Flag: int
    Expected_Attendance: int
    Temperature_C: float
    Rainfall_mm: float
    Price_per_Item: float


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    index_file = os.path.join(BASE_DIR, "templates", "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


@app.get("/api/dataset-info")
async def get_dataset_info():
    if not os.path.exists(DATA_PATH):
        generate_canteen_dataset(output_path=DATA_PATH)
        
    df = pd.read_csv(DATA_PATH)
    unique_items = sorted(df["Food_Item"].dropna().unique().tolist())
    categories = sorted(df["Food_Category"].dropna().unique().tolist())
    
    preview_df = df.head(10).fillna("")
    preview_records = preview_df.to_dict(orient="records")
    
    stats_df = df.describe().T.reset_index()
    stats_records = []
    for _, row in stats_df.iterrows():
        stats_records.append({
            "feature": row["index"],
            "mean": round(float(row.get("mean", 0)), 2),
            "std": round(float(row.get("std", 0)), 2),
            "min": round(float(row.get("min", 0)), 2),
            "max": round(float(row.get("max", 0)), 2)
        })
    
    return {
        "success": True,
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "unique_items": len(unique_items),
        "food_items": unique_items,
        "categories": categories,
        "date_range": f"{df['Date'].min()} to {df['Date'].max()}",
        "preview_rows": preview_records,
        "stats_rows": stats_records
    }


@app.post("/api/add-food-data")
async def add_food_data(record: NewFoodDataInput):
    global model_bundle
    try:
        if not os.path.exists(DATA_PATH):
            generate_canteen_dataset(output_path=DATA_PATH)
            
        df = pd.read_csv(DATA_PATH)
        
        # Calculate Day of week and leftover
        date_obj = pd.to_datetime(record.Date)
        day_of_week = date_obj.day_name()
        leftover = max(0, record.Quantity_Prepared - record.Quantity_Sold)
        
        new_row = {
            "Date": record.Date,
            "Day_of_Week": day_of_week,
            "Meal_Type": record.Meal_Type.strip(),
            "Food_Item": record.Food_Item.strip(),
            "Food_Category": record.Food_Category.strip(),
            "Quantity_Prepared": int(record.Quantity_Prepared),
            "Quantity_Sold": int(record.Quantity_Sold),
            "Leftover_Quantity": leftover,
            "Holiday_Flag": int(record.Holiday_Flag),
            "Exam_Week_Flag": int(record.Exam_Week_Flag),
            "Expected_Attendance": int(record.Expected_Attendance),
            "Temperature_C": float(record.Temperature_C),
            "Rainfall_mm": float(record.Rainfall_mm),
            "Price_per_Item": float(record.Price_per_Item)
        }
        
        df_new = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df_new_cleaned = clean_canteen_data(df_new)
        df_new_cleaned.to_csv(DATA_PATH, index=False)
        
        return {
            "success": True,
            "message": f"Successfully logged record for '{record.Food_Item}' on {record.Date}. Total records now: {len(df_new_cleaned)}.",
            "record": new_row
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        temp_path = os.path.join(APP_SRC, "data", "uploaded_temp.csv")
        with open(temp_path, "wb") as f:
            f.write(contents)
            
        user_df = pd.read_csv(temp_path)
        cleaned_user_df = clean_canteen_data(user_df)
        cleaned_user_df.to_csv(DATA_PATH, index=False)
        return {"success": True, "message": f"Successfully loaded and validated {len(cleaned_user_df)} rows from {file.filename}."}
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})


@app.post("/api/reset-dataset")
async def reset_dataset():
    fresh_df = generate_canteen_dataset(output_path=DATA_PATH)
    clean_canteen_data(fresh_df).to_csv(DATA_PATH, index=False)
    return {"success": True, "message": "Fresh 365-day canteen dataset generated successfully!"}


@app.get("/api/analytics")
async def get_analytics():
    if not os.path.exists(DATA_PATH):
        generate_canteen_dataset(output_path=DATA_PATH)
        
    df = pd.read_csv(DATA_PATH)
    
    total_prepared = int(df["Quantity_Prepared"].sum())
    total_sold = int(df["Quantity_Sold"].sum())
    total_leftover = int(df["Leftover_Quantity"].sum())
    waste_pct = round((total_leftover / total_prepared * 100), 1) if total_prepared > 0 else 0
    avg_daily = int(df.groupby("Date")["Quantity_Sold"].sum().mean())
    
    top_item = df.groupby("Food_Item")["Quantity_Sold"].sum().idxmax()
    top_meal = df.groupby("Meal_Type")["Quantity_Sold"].sum().idxmax()
    top_day = df.groupby("Day_of_Week")["Quantity_Sold"].sum().idxmax()
    
    daily = df.groupby("Date")[["Quantity_Prepared", "Quantity_Sold", "Leftover_Quantity"]].sum().reset_index()
    step = max(1, len(daily) // 60)
    daily_sampled = daily.iloc[::step]
    
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df.groupby("Day_of_Week")["Quantity_Sold"].mean().reindex(dow_order).fillna(0)
    
    meal = df.groupby("Meal_Type")["Quantity_Sold"].sum()
    cat_demand = df.groupby("Food_Category")["Quantity_Sold"].sum().sort_values(ascending=True)
    top_items = df.groupby("Food_Item")["Quantity_Sold"].sum().sort_values(ascending=False).head(8)
    waste_items = df.groupby("Food_Item")["Leftover_Quantity"].sum().sort_values(ascending=False).head(8)
    
    def classify_period(r):
        if r["Holiday_Flag"] == 1:
            return "Holiday / Break"
        elif r["Exam_Week_Flag"] == 1:
            return "Exam Week"
        return "Normal Day"
        
    df["Period"] = df.apply(classify_period, axis=1)
    period_summary = df.groupby("Period")[["Quantity_Prepared", "Quantity_Sold", "Leftover_Quantity"]].mean().round(1)
    
    return {
        "success": True,
        "kpis": {
            "total_prepared": total_prepared,
            "total_sold": total_sold,
            "total_leftover": total_leftover,
            "waste_pct": waste_pct,
            "avg_daily": avg_daily,
            "top_item": top_item,
            "top_meal": top_meal,
            "top_day": top_day
        },
        "charts": {
            "daily_dates": daily_sampled["Date"].tolist(),
            "daily_prepared": daily_sampled["Quantity_Prepared"].tolist(),
            "daily_sold": daily_sampled["Quantity_Sold"].tolist(),
            "daily_leftover": daily_sampled["Leftover_Quantity"].tolist(),
            "dow_labels": dow.index.tolist(),
            "dow_values": [round(v, 1) for v in dow.values.tolist()],
            "meal_labels": meal.index.tolist(),
            "meal_values": meal.values.tolist(),
            "cat_labels": cat_demand.index.tolist(),
            "cat_values": cat_demand.values.tolist(),
            "top_item_labels": top_items.index.tolist(),
            "top_item_values": top_items.values.tolist(),
            "waste_item_labels": waste_items.index.tolist(),
            "waste_item_values": waste_items.values.tolist(),
            "period_labels": period_summary.index.tolist(),
            "period_prepared": period_summary["Quantity_Prepared"].tolist(),
            "period_sold": period_summary["Quantity_Sold"].tolist(),
            "period_leftover": period_summary["Leftover_Quantity"].tolist()
        }
    }


@app.post("/api/predict")
async def run_predict(input_data: PredictionInput):
    global model_bundle
    if model_bundle is None:
        model_bundle = load_trained_model(MODEL_PATH)
        
    input_dict = input_data.model_dump()
    predicted_demand = predict_demand(input_dict, model_bundle=model_bundle)
    metrics = calculate_waste_metrics(predicted_demand, input_data.Usual_Preparation)
    
    new_record = {
        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "Date": input_data.Date,
        "Food_Item": input_data.Food_Item,
        "Meal_Type": input_data.Meal_Type,
        "Predicted_Demand": metrics["predicted_demand"],
        "Safety_Stock": metrics["safety_stock"],
        "Recommended_Preparation": metrics["recommended_preparation"],
        "Usual_Preparation": metrics["usual_preparation"],
        "Meals_Saved": metrics["meals_saved"],
        "Waste_Reduction_Pct": metrics["waste_reduction_pct"],
        "Status": metrics["status_category"]
    }
    
    if os.path.exists(REPORT_PATH):
        hist_df = pd.read_csv(REPORT_PATH)
        hist_df = pd.concat([hist_df, pd.DataFrame([new_record])], ignore_index=True)
    else:
        hist_df = pd.DataFrame([new_record])
        
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    hist_df.to_csv(REPORT_PATH, index=False)
    
    return {
        "success": True,
        **metrics
    }


@app.get("/api/history")
async def get_history():
    if not os.path.exists(REPORT_PATH):
        return {"records": []}
    hist_df = pd.read_csv(REPORT_PATH).tail(25)
    return {"records": hist_df.fillna("").to_dict(orient="records")}


@app.post("/api/train")
async def trigger_train():
    global model_bundle
    df = pd.read_csv(DATA_PATH)
    clean_df = clean_canteen_data(df)
    summary = train_and_evaluate_models(clean_df, save_path=MODEL_PATH)
    model_bundle = load_trained_model(MODEL_PATH)
    
    leaderboard = summary["results_df"].to_dict(orient="records")
    feat_imp = summary["feature_importances"]
    feat_dict = {}
    if feat_imp is not None:
        feat_dict = {str(k): round(float(v), 4) for k, v in feat_imp.head(10).items()}
        
    return {
        "success": True, 
        "best_model_name": summary["best_model_name"],
        "leaderboard": leaderboard,
        "feature_importances": feat_dict
    }


@app.get("/api/model-info")
async def get_model_info():
    global model_bundle
    if model_bundle is None:
        try:
            model_bundle = load_trained_model(MODEL_PATH)
        except Exception:
            return {"trained": False}
            
    res_df = model_bundle.get("results_df")
    leaderboard = res_df.to_dict(orient="records") if res_df is not None else []
    
    feat_imp = model_bundle.get("feature_importances")
    feat_labels = []
    feat_vals = []
    if feat_imp is not None:
        feat_labels = list(feat_imp.head(10).index)
        feat_vals = [round(float(v), 4) for v in feat_imp.head(10).values]
        
    return {
        "trained": True,
        "best_model_name": model_bundle.get("best_model_name", "Trained Model"),
        "trained_date": model_bundle.get("trained_date", "Recent"),
        "leaderboard": leaderboard,
        "feature_labels": feat_labels,
        "feature_values": feat_vals
    }


@app.get("/api/download-data")
async def download_data():
    if os.path.exists(DATA_PATH):
        return FileResponse(DATA_PATH, media_type="text/csv", filename="cleaned_canteen_data.csv")
    return {"error": "File not found"}


@app.get("/api/download-report")
async def download_report():
    if os.path.exists(REPORT_PATH):
        return FileResponse(REPORT_PATH, media_type="text/csv", filename="canteen_waste_reduction_report.csv")
    return {"error": "No report found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
