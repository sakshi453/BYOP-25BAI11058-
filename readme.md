

🏠 House Sales Analysis: Prediction & Classification
A Dual-Model Machine Learning Pipeline

📖 Project Overview
This project addresses the complexity of real estate valuation by implementing two distinct machine learning approaches. It doesn't just predict a price; it categorizes market segments to provide a 360-degree view of property data.

Regression Engine: Uses XGBoost (Gradient Boosting) to predict the continuous price variable with high precision.

Classification Engine: Uses Random Forest to label houses as "Premium" or "Budget" based on the dataset's median price point.

🛠️ Core Features
Modular Architecture: Logic is separated into preprocess, train, and predict modules for production-ready code.

Automated Preprocessing: Handles missing values, performs feature scaling, and manages data splits automatically.

Inference Engine: A dedicated predict.py script to allow quick testing of new house parameters without retraining.

Data-Driven Insights: Includes feature importance mapping to identify which variables (like sqft_living) drive market value.

🚀 Installation & Usage
1. Environment Setup
Bash
# Clone the repository
git clone https://github.com/sakshi-453/BYOP(25BAI11058).git
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt
2. Training the Pipeline
To train both the regression and classification models, run the scripts from the root directory:

Bash
python models/train_regression.py
python models/train_classification.py
3. Running Inference
To predict the price of a specific house:
python predict.py