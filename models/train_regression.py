from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from preprocess import load_and_clean_data

# 1. Get Data
X_train, X_test, y_train, y_test = load_and_clean_data('data/house_data.csv')

# 2. Initialize and Train Model
reg_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
reg_model.fit(X_train, y_train)

# 3. Evaluate
preds = reg_model.predict(X_test)
print(f"Regression R2 Score: {r2_score(y_test, preds):.4f}")
print(f"Average Error (MAE): ${mean_absolute_error(y_test, preds):,.2f}")

# 4. Save model (Optional)
reg_model.save_model('models/price_model.json')