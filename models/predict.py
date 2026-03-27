import pandas as pd
from xgboost import XGBRegressor

def predict_my_house(beds, baths, sqft_liv, sqft_lot, floors):
    # Load the trained model
    model = XGBRegressor()
    model.load_model('models/price_model.json')
    
    # Match the column names exactly to your preprocess.py
    columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']
    new_data = pd.DataFrame([[beds, baths, sqft_liv, sqft_lot, floors]], columns=columns)
    
    price = model.predict(new_data)[0]
    print(f"--- Prediction Results ---")
    print(f"Estimated Price: ${price:,.2f}")

# Example: 3 bed, 2 bath, 2000 sqft living, 5000 sqft lot, 1 floor
predict_my_house(3, 2, 2000, 5000, 1)