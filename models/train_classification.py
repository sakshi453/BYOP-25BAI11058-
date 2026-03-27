import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocess import load_and_clean_data

# 1. Get Data
X_train, X_test, y_train, y_test = load_and_clean_data('data/house_data.csv')

# 2. Convert Prices to Categories (Classification Target)
median_val = y_train.median()
y_train_cat = (y_train > median_val).astype(int)
y_test_cat = (y_test > median_val).astype(int)

# 3. Train Classifier
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_train_cat)

# 4. Evaluate
clf_preds = clf_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test_cat, clf_preds))