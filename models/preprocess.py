import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Fill missing values with median (to avoid errors with empty cells)
    df = df.fillna(df.median(numeric_only=True))
    
    # Updated Features based on your CSV screenshot:
    # columns: price, bedrooms, bathrooms, sqft_living, sqft_lot, floors
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors']
    
    X = df[features]
    y = df['price']
    
    # Split: 80% for training, 20% for testing
    return train_test_split(X, y, test_size=0.2, random_state=42)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler