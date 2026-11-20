"""
Data Generator Module for SmartCanteen System
Generates realistic canteen sales and preparation records over at least 365 days.
Includes breakfast, lunch, and dinner records, multiple food items, weekdays/weekends,
holidays, exam weeks, weather conditions, prepared, sold, leftover quantities, and prices.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# List of realistic food items per meal type with baseline prices and base demand
FOOD_CATALOG = [
    # Breakfast
    {"Meal_Type": "Breakfast", "Food_Item": "Masala Dosa", "Food_Category": "South Indian", "Base_Price": 45.0, "Base_Demand": 160},
    {"Meal_Type": "Breakfast", "Food_Item": "Idli Vada Combo", "Food_Category": "South Indian", "Base_Price": 40.0, "Base_Demand": 140},
    {"Meal_Type": "Breakfast", "Food_Item": "Aloo Paratha", "Food_Category": "North Indian", "Base_Price": 50.0, "Base_Demand": 130},
    {"Meal_Type": "Breakfast", "Food_Item": "Poha & Tea", "Food_Category": "Snacks", "Base_Price": 30.0, "Base_Demand": 120},
    {"Meal_Type": "Breakfast", "Food_Item": "Veg Sandwich", "Food_Category": "Continental", "Base_Price": 40.0, "Base_Demand": 100},
    
    # Lunch
    {"Meal_Type": "Lunch", "Food_Item": "Special Veg Thali", "Food_Category": "Thali", "Base_Price": 85.0, "Base_Demand": 260},
    {"Meal_Type": "Lunch", "Food_Item": "Paneer Butter Masala Meal", "Food_Category": "North Indian", "Base_Price": 110.0, "Base_Demand": 210},
    {"Meal_Type": "Lunch", "Food_Item": "Chicken Biryani", "Food_Category": "Rice Bowls", "Base_Price": 140.0, "Base_Demand": 290},
    {"Meal_Type": "Lunch", "Food_Item": "Veg Fried Rice & Manchurian", "Food_Category": "Chinese", "Base_Price": 90.0, "Base_Demand": 180},
    {"Meal_Type": "Lunch", "Food_Item": "Dal Khichdi & Curd", "Food_Category": "North Indian", "Base_Price": 70.0, "Base_Demand": 130},

    # Dinner
    {"Meal_Type": "Dinner", "Food_Item": "Mini Executive Thali", "Food_Category": "Thali", "Base_Price": 75.0, "Base_Demand": 190},
    {"Meal_Type": "Dinner", "Food_Item": "Chole Bhature", "Food_Category": "North Indian", "Base_Price": 70.0, "Base_Demand": 170},
    {"Meal_Type": "Dinner", "Food_Item": "Egg Curry with Roti", "Food_Category": "North Indian", "Base_Price": 90.0, "Base_Demand": 150},
    {"Meal_Type": "Dinner", "Food_Item": "Veg Hakka Noodles", "Food_Category": "Chinese", "Base_Price": 80.0, "Base_Demand": 160},
    {"Meal_Type": "Dinner", "Food_Item": "South Indian Meals", "Food_Category": "South Indian", "Base_Price": 65.0, "Base_Demand": 140},
]


def generate_canteen_dataset(
    start_date="2025-01-01",
    days=365,
    random_seed=42,
    output_path=None
):
    """
    Generates a realistic multi-meal, multi-item canteen dataset spanning 'days' days.
    
    Returns:
        pd.DataFrame: Generated canteen historical dataset.
    """
    np.random.seed(random_seed)
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    records = []
    
    # Pre-calculate exam periods (e.g., mid-sem & end-sem periods)
    # Day ranges within 365 days that represent exams
    exam_day_windows = [
        (60, 72),    # March Mid-sem
        (130, 145),  # May End-sem
        (260, 272),  # October Mid-sem
        (335, 350)   # December End-sem
    ]
    
    # Public & College holidays (selected day offsets)
    holiday_offsets = {
        25, 26,       # Republic Day window
        75,           # Holi
        104,          # Ambedkar Jayanti / Eid
        227,          # Independence day
        274, 275,     # Gandhi Jayanti / Navratri
        304, 305,     # Diwali
        358           # Christmas
    }
    
    for day_idx in range(days):
        current_date = start + timedelta(days=day_idx)
        date_str = current_date.strftime("%Y-%m-%d")
        day_of_week = current_date.strftime("%A")
        
        # Holiday flag: specific holiday dates or every 4th Sunday break
        is_holiday = 1 if (day_idx in holiday_offsets) else 0
        
        # Exam week flag: if current day falls into any exam window
        is_exam_week = 1 if any(start_w <= day_idx <= end_w for start_w, end_w in exam_day_windows) else 0
        
        # Base campus attendance
        # Normal weekday: ~1100-1300 students
        # Saturday: ~600-800 students
        # Sunday: ~250-400 students (hostelers mainly)
        # Exam weeks: attendance rises 20-30% on weekdays
        # Holidays: attendance plummets ~70%
        if day_of_week == "Sunday":
            base_att = np.random.normal(320, 30)
        elif day_of_week == "Saturday":
            base_att = np.random.normal(700, 50)
        else:
            base_att = np.random.normal(1200, 60)
            
        if is_exam_week:
            base_att *= 1.25  # Library & revision rush
        if is_holiday:
            base_att *= 0.35  # Most students stay back / day scholars absent
            
        attendance = max(150, int(base_att))
        
        # Simulated seasonal weather
        # Temperature sinusoidal variation across the year
        day_angle = (day_idx / 365.0) * 2 * np.pi
        temp_c = 28.0 + 8.0 * np.sin(day_angle - 1.2) + np.random.normal(0, 1.8)
        temp_c = round(max(15.0, min(42.0, temp_c)), 1)
        
        # Monsoon rainfall mainly in days 160 to 260
        if 160 <= day_idx <= 260:
            rain_mm = round(max(0.0, np.random.exponential(12.0) - 2.0), 1) if np.random.rand() > 0.4 else 0.0
        else:
            rain_mm = round(max(0.0, np.random.exponential(2.0) - 1.5), 1) if np.random.rand() > 0.85 else 0.0
            
        # For each food item across breakfast, lunch, and dinner
        for item_info in FOOD_CATALOG:
            meal_type = item_info["Meal_Type"]
            food_item = item_info["Food_Item"]
            food_cat = item_info["Food_Category"]
            price = item_info["Base_Price"]
            base_item_demand = item_info["Base_Demand"]
            
            # Demand multiplier based on attendance ratio vs baseline 1200
            att_factor = attendance / 1200.0
            
            # Day of week factor
            # Weekdays: higher lunch and breakfast
            # Weekends: dinner holds relatively stable for hostelers, lunch drops
            dow_multiplier = 1.0
            if day_of_week in ["Saturday", "Sunday"]:
                if meal_type == "Lunch":
                    dow_multiplier = 0.55
                elif meal_type == "Breakfast":
                    dow_multiplier = 0.65
                else: # Dinner
                    dow_multiplier = 0.85
            else:
                if meal_type == "Lunch":
                    dow_multiplier = 1.15
                elif day_of_week in ["Tuesday", "Thursday"]:
                    dow_multiplier = 1.05
                    
            # Weather impact: On rainy days, students prefer eating on campus canteen rather than going outside
            weather_bonus = 1.0 + (0.008 * rain_mm)
            if rain_mm > 15:
                weather_bonus += 0.10
                
            # Exam week boost for quick snacks, noodles, chai/tea, biryani, thali
            exam_boost = 1.0
            if is_exam_week:
                exam_boost = 1.22
            if is_holiday:
                exam_boost *= 0.50
                
            # Small random market noise
            random_factor = np.random.normal(1.0, 0.06)
            
            # Expected final sold demand
            calc_sold = base_item_demand * att_factor * dow_multiplier * weather_bonus * exam_boost * random_factor
            calc_sold = max(10, int(round(calc_sold)))
            
            # Realistic canteen preparation habit:
            # Canteens often overprepare by 10% to 25% to avoid running out,
            # or sometimes underprepare if weather was unexpected
            overprep_pct = np.random.uniform(0.08, 0.28)
            quantity_prepared = int(np.round(calc_sold * (1.0 + overprep_pct)))
            
            # In real operations, quantity sold cannot exceed quantity prepared
            # Occasionally (e.g. 8% of times) demand exceeds preparation (stockout situation)
            if np.random.rand() < 0.08:
                quantity_prepared = int(np.round(calc_sold * np.random.uniform(0.88, 0.98)))
                quantity_sold = quantity_prepared
            else:
                quantity_sold = calc_sold
                
            leftover_quantity = max(0, quantity_prepared - quantity_sold)
            
            records.append({
                "Date": date_str,
                "Day_of_Week": day_of_week,
                "Meal_Type": meal_type,
                "Food_Item": food_item,
                "Food_Category": food_cat,
                "Quantity_Prepared": quantity_prepared,
                "Quantity_Sold": quantity_sold,
                "Leftover_Quantity": leftover_quantity,
                "Holiday_Flag": is_holiday,
                "Exam_Week_Flag": is_exam_week,
                "Expected_Attendance": attendance,
                "Temperature_C": temp_c,
                "Rainfall_mm": rain_mm,
                "Price_per_Item": price
            })
            
    df = pd.DataFrame(records)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Sample canteen dataset successfully generated at: {output_path} with {len(df)} records.")
        
    return df


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    out_file = os.path.join(project_root, "data", "sample_canteen_data.csv")
    generate_canteen_dataset(output_path=out_file)
