"""
Waste Calculator Module for SmartCanteen System
Calculates safety stock, recommended preparation quantities, expected leftover,
waste reduction potential, and actionable recommendations.
"""

def calculate_waste_metrics(predicted_demand, usual_preparation, safety_stock_pct=0.10):
    """
    Computes food demand planning and waste metrics according to SmartCanteen formulas:
    - Predicted_Demand = ML model prediction
    - Safety_Stock = 10% of Predicted_Demand
    - Recommended_Preparation = Predicted_Demand + Safety_Stock
    - Expected_Leftover = Quantity_Prepared - Predicted_Demand (for usual prep)
    - Waste_Percentage = (Expected_Leftover / Quantity_Prepared) * 100
    - Waste_Reduction_Potential = ((Usual_Preparation - Recommended_Preparation) / Usual_Preparation) * 100
    
    Args:
        predicted_demand (float or int): Predicted meals from ML model.
        usual_preparation (float or int): Routine meals prepared by canteen staff.
        safety_stock_pct (float): Safety stock fraction (default 0.10 = 10%).
        
    Returns:
        dict: Computed metrics and status summary.
    """
    predicted_demand = max(0, int(round(predicted_demand)))
    usual_preparation = max(0, int(round(usual_preparation)))
    
    # 1. Safety Stock
    safety_stock = int(round(predicted_demand * safety_stock_pct))
    
    # 2. Recommended Preparation
    recommended_preparation = predicted_demand + safety_stock
    
    # 3. Expected Leftover with Usual Preparation
    expected_leftover_usual = max(0, usual_preparation - predicted_demand)
    
    # Expected Leftover with SmartCanteen Recommended Preparation
    expected_leftover_recommended = max(0, recommended_preparation - predicted_demand)
    
    # 4. Waste Percentages
    waste_pct_usual = (expected_leftover_usual / usual_preparation * 100.0) if usual_preparation > 0 else 0.0
    waste_pct_recommended = (expected_leftover_recommended / recommended_preparation * 100.0) if recommended_preparation > 0 else 0.0
    
    # 5. Waste Reduction Potential
    # Potential meal quantity saved
    meals_saved = max(0, usual_preparation - recommended_preparation)
    
    if usual_preparation > 0:
        waste_reduction_pct = ((usual_preparation - recommended_preparation) / usual_preparation) * 100.0
    else:
        waste_reduction_pct = 0.0
        
    # Categorize Waste Level & Practical Actionable Recommendation
    if usual_preparation < predicted_demand:
        status_category = "Shortage Risk"
        status_color = "red"
        recommendation_text = f"Shortage warning! Usual preparation ({usual_preparation}) is lower than expected demand ({predicted_demand}). Prepare extra stock ({recommended_preparation} meals) to avoid running out."
    elif waste_pct_usual <= 12.0:
        status_category = "Low Waste"
        status_color = "green"
        recommendation_text = f"Efficient planning! Your usual preparation aligns closely with demand. Maintain current kitchen portioning."
    elif waste_pct_usual <= 25.0:
        status_category = "Moderate Waste"
        status_color = "yellow"
        recommendation_text = f"Moderate waste detected ({waste_pct_usual:.1f}%). Adjust kitchen batching. Reduce preparation by ~{meals_saved} meals to save costs."
    else:
        status_category = "High Waste"
        status_color = "red"
        recommendation_text = f"High waste alert ({waste_pct_usual:.1f}% surplus)! Consider scaling back preparation by {abs(waste_reduction_pct):.1f}%, or promote low-demand food items with student combo discounts."
        
    return {
        "predicted_demand": predicted_demand,
        "usual_preparation": usual_preparation,
        "safety_stock": safety_stock,
        "recommended_preparation": recommended_preparation,
        "expected_leftover_usual": expected_leftover_usual,
        "expected_leftover_recommended": expected_leftover_recommended,
        "meals_saved": meals_saved,
        "waste_pct_usual": round(waste_pct_usual, 1),
        "waste_pct_recommended": round(waste_pct_recommended, 1),
        "waste_reduction_pct": round(waste_reduction_pct, 1),
        "status_category": status_category,
        "status_color": status_color,
        "recommendation_text": recommendation_text
    }
