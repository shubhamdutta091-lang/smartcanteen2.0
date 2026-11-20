# SmartCanteen: ML-Based Food Demand and Waste Reduction System

🌱 **SmartCanteen** is an end-to-end Machine Learning web application designed to help college and institutional canteens forecast tomorrow’s food demand, optimize kitchen preparation quantities with automated safety stock buffers, prevent meal shortages, and drastically cut food waste.

Built specifically for a **2nd-year college machine-learning project**, it is modular, beginner-friendly, visually engaging, and ready for classroom or project jury demonstrations.

---

## 🎯 Problem Statement & Objective
College canteens face severe demand volatility driven by exam schedules, day-of-week trends, weather disruptions, and campus attendance. Canteen managers traditionally rely on guesswork, resulting in:
- **15% to 35% food waste** on low-attendance days.
- **Stockouts and student shortages** during exam weeks and peak hours.
- **Severe financial loss** from wasted ingredients and utilities.

**SmartCanteen** addresses this problem using supervised Machine Learning regression models (Linear Regression, Decision Tree, Random Forest, and XGBoost) along with historical lag features, moving averages, and contextual indicators to recommend the exact quantity of food to cook tomorrow with an automated **10% safety stock**.

---

## 🌟 Key Features
- **Automatic 365-Day Synthetic Dataset Generation**: Realistic simulated records containing breakfast, lunch, and dinner, multi-item menus, exam weeks, weather conditions, price, prepared, sold, and leftover quantities.
- **Dynamic Dataset Uploader & Inspector**: Upload custom CSVs or use the built-in dataset; inspect missing values, duplicates, datatypes, summary statistics, and export cleaned CSVs.
- **Exploratory Data Analysis (EDA)**: Interactive Plotly visual charts for daily sales trends, day-of-week averages, meal type distribution, top-selling items, highest-leftover foods, and academic impact comparisons (normal days vs exam weeks vs holidays).
- **Time-Based ML Training & Benchmarking**:
  - Compares 4 regression models: **Linear Regression, Decision Tree Regressor, Random Forest Regressor, and XGBoost Regressor**.
  - Strictly prevents data leakage by using a chronological time-series split.
  - Automatically selects and serializes the best-performing model (`models/food_demand_model.pkl`) based on **MAE, RMSE, and R² Score**.
  - Displays feature importances and predicted vs actual scatter plots.
- **Future Demand Prediction & Safety Buffer Calculator**:
  - Interactive prediction form for tomorrow's date, meal type, food item, campus attendance, exam status, holiday flag, weather, and recent sales.
  - Recommends preparation quantities using:
    $$\text{Safety Stock} = 10\% \times \text{Predicted Demand}$$
    $$\text{Recommended Preparation} = \text{Predicted Demand} + \text{Safety Stock}$$
    $$\text{Waste Reduction Potential} = \frac{\text{Usual Preparation} - \text{Recommended Preparation}}{\text{Usual Preparation}} \times 100$$
- **Live Waste & Financial Audit Report**:
  - What-if waste calculator with color-coded alerts (Green: Low Waste, Yellow: Moderate Waste, Red: High Waste/Shortage).
  - Downloadable CSV audit trail for historical predictions and waste reduction estimates.
- **Sustainability Theme & Modern UI**: Responsive design styled with green, white, and light-grey sustainability aesthetics and consistent navigation.

---

## 🏗️ System Architecture
```mermaid
flowchart TD
    A[Data Generation / CSV Upload] --> B[Data Preprocessing & Validation]
    B --> C[Feature Engineering & Lags / Moving Averages]
    C --> D[Time-Based Train/Test Split]
    D --> E[Model Benchmarking: Linear, DT, RF, XGBoost]
    E --> F[Select & Save Best Model (.pkl)]
    F --> G[Demand Prediction & 10% Safety Buffer]
    G --> H[Waste Calculation & Batch Sizing Recommendation]
    H --> I[Streamlit Dashboard & CSV Audit Export]
```

---

## 📁 Project Folder Structure
```text
smart_canteen_app/
│
├── app/
│   └── streamlit_app.py        # Streamlit interactive multi-page web dashboard
│
├── src/
│   ├── data_generator.py       # Realistic 365-day canteen data synthesis
│   ├── data_preprocessing.py   # Data validation, cleaning, and null handling
│   ├── feature_engineering.py  # Lag features, moving averages, and encodings
│   ├── train_model.py          # ML model training, benchmarking, and serialization
│   ├── predict.py              # Inference pipeline for single-record prediction
│   └── waste_calculator.py     # Waste reduction, safety buffer, and metrics logic
│
├── data/
│   └── sample_canteen_data.csv # Generated 365-day multi-meal dataset
│
├── models/
│   └── food_demand_model.pkl   # Serialized best trained ML model bundle
│
├── reports/
│   └── prediction_report.csv   # Historical prediction logs and waste audit export
│
├── requirements.txt            # Python dependencies
└── README.md                   # Comprehensive project documentation
```

---

## 📊 Dataset Schema
The dataset contains 14 primary fields:
| Column | Description | Type |
| :--- | :--- | :--- |
| `Date` | Date of canteen operation | `datetime` (YYYY-MM-DD) |
| `Day_of_Week` | Monday through Sunday | `string` |
| `Meal_Type` | Breakfast, Lunch, or Dinner | `string` |
| `Food_Item` | Specific dish name (e.g., Chicken Biryani, Special Veg Thali) | `string` |
| `Food_Category` | Category (e.g., South Indian, North Indian, Thali, Chinese) | `string` |
| `Quantity_Prepared` | Number of food units cooked by canteen staff | `integer` |
| `Quantity_Sold` | **Target Variable**: Number of food units actually purchased | `integer` |
| `Leftover_Quantity` | Surplus food unconsumed ($\text{Prepared} - \text{Sold}$) | `integer` |
| `Holiday_Flag` | Binary flag ($1$ = Holiday / Weekend break, $0$ = Normal day) | `integer` |
| `Exam_Week_Flag` | Binary flag ($1$ = Mid-sem/End-sem exam week, $0$ = Regular) | `integer` |
| `Expected_Attendance` | Headcount of students and faculty on campus | `integer` |
| `Temperature_C` | Ambient temperature in Celsius | `float` |
| `Rainfall_mm` | Daily precipitation in millimeters | `float` |
| `Price_per_Item` | Price of the food item in ₹ | `float` |

---

## 🤖 Machine Learning Models & Evaluation Metrics

### 1. Algorithms Evaluated:
- **Linear Regression**: Baseline regression testing linear relationships.
- **Decision Tree Regressor**: Non-linear splits capturing exam spikes and weather thresholds.
- **Random Forest Regressor**: Ensemble of bagging decision trees reducing variance.
- **XGBoost Regressor**: Extreme Gradient Boosting optimized for tabular time-series patterns.

### 2. Evaluation Metrics:
- **Mean Absolute Error (MAE)**: Average magnitude of prediction errors in meal units.
- **Root Mean Squared Error (RMSE)**: Penalizes larger forecasting errors heavily.
- **R² Score (Coefficient of Determination)**: Proportion of variance explained by the model (reaches up to $0.98$).

---

## ⚙️ Installation & Run Instructions

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Clone or Navigate to the Repository
```bash
cd smartcanteen2.0/smart_canteen_app
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Run Scripts Independently
- Generate sample dataset:
  ```bash
  python src/data_generator.py
  ```
- Train ML models and generate `.pkl` model file:
  ```bash
  python -c "import sys; sys.path.append('.'); from src.train_model import train_and_evaluate_models; from src.data_preprocessing import clean_canteen_data; import pandas as pd; df = pd.read_csv('data/sample_canteen_data.csv'); train_and_evaluate_models(clean_canteen_data(df), save_path='models/food_demand_model.pkl')"
  ```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app/streamlit_app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🔮 Future Improvements
1. **IoT RFID Gate Integration**: Syncing real-time turnstile card taps for live campus population tracking.
2. **Student Mobile Pre-Ordering**: Allowing students to preorder 2 hours in advance to tighten buffer margins.
3. **Automated NGO / Composting Dispatch**: Connecting surplus batches with local food banks or composting units before spoilage.
4. **Item Perishability Factors**: Dynamic safety stocks according to dish shelf life (e.g. 5% for salads vs 15% for curries).

---

## 👥 Authors & Team
- **Team Members:** [Student Name 1, Student Name 2, Student Name 3]
- **Department:** Department of Computer Science & Engineering
- **Institution:** College of Engineering & Technology
- **Project Mentor:** [Prof. Guide Name]

---

*SmartCanteen | Reducing Food Waste with Machine Learning*
