import sys
import io

# Force UTF-8 stdout for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

def train_house_price_model():
    print("Starting House Price Model Training Pipeline with CatBoost...")
    
    # 1. Load Raw Data
    raw_path = "Bengaluru_House_Data.csv"
    df = pd.read_csv(raw_path)
    print(f"Raw Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Drop unused/sparse columns
    df = df.drop(columns=['area_type', 'balcony', 'society', 'availability'], errors='ignore')
    
    # 2. Clean Location
    df['location'] = df['location'].fillna('Sarjapur  Road').apply(lambda x: str(x).strip())
    loc_counts = df['location'].value_counts()
    loc_less_10 = loc_counts[loc_counts <= 10]
    df['location'] = df['location'].apply(lambda x: 'others' if x in loc_less_10 else x)
    
    # 3. Clean BHK / Size
    df['size'] = df['size'].fillna('2 BHK')
    df['bhk'] = df['size'].apply(lambda x: int(str(x).split()[0]))
    
    # 4. Clean Bath
    med_bath = df['bath'].median()
    df['bath'] = df['bath'].fillna(med_bath).astype(int)
    
    # 5. Clean Total Sqft (Fixing Math Bug)
    def clean_sqft(sqft):
        parts = str(sqft).split('-')
        try:
            if len(parts) == 2:
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2.0
            return float(parts[0].strip())
        except:
            return np.nan

    df['total_sqft'] = df['total_sqft'].apply(clean_sqft)
    df = df.dropna(subset=['total_sqft'])
    
    # Drop exact duplicates
    df = df.drop_duplicates()
    
    # 6. Price Per Sqft Calculation & Outlier Filtering
    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
    
    # Domain Rule Outliers
    df = df[df['total_sqft'] / df['bhk'] >= 300]
    df = df[df['bhk'] <= 6]
    df = df[df['bath'] < df['bhk'] + 2]
    
    # IQR Method Outlier Removal
    q1 = df['price_per_sqft'].quantile(0.25)
    q3 = df['price_per_sqft'].quantile(0.75)
    IQR = q3 - q1
    lower_bound = q1 - 0.5 * IQR
    upper_bound = q3 + 0.5 * IQR
    
    df_clean = df[(df['price_per_sqft'] >= lower_bound) & (df['price_per_sqft'] <= upper_bound)].copy()
    df_clean.reset_index(drop=True, inplace=True)
    
    print(f"Cleaned Dataset Shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    
    # 7. Save Lightweight Cleaned CSV for Web App
    clean_csv_cols = ['location', 'total_sqft', 'bath', 'bhk', 'price', 'price_per_sqft']
    df_clean[clean_csv_cols].to_csv("cleaned_df.csv", index=False)
    print("Saved lightweight cleaned_df.csv successfully!")
    
    # 8. Feature Engineering for ML Training
    df_clean['sqft_per_bhk'] = df_clean['total_sqft'] / df_clean['bhk']
    df_clean['bath_per_bhk'] = df_clean['bath'] / df_clean['bhk']
    
    # 9. One-Hot Encoding
    df_encoded = pd.get_dummies(df_clean.drop(columns=['size', 'price_per_sqft']), columns=['location'], drop_first=True, dtype=int)
    
    X = df_encoded.drop(columns=['price'])
    y = df_encoded['price']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training Features: {X_train.shape[1]} columns")
    print(f"Training Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples")
    
    # 10. Model Training (Weighted Voting Ensemble of RF + ET + CatBoost)
    rf = RandomForestRegressor(
        n_estimators=350,
        max_depth=24,
        min_samples_split=3,
        max_features=0.4,
        random_state=42,
        n_jobs=-1
    )
    
    et = ExtraTreesRegressor(
        n_estimators=350,
        max_depth=26,
        min_samples_split=2,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    )
    
    cb = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        random_seed=42,
        verbose=0
    )
    
    # Weighted Ensemble (giving higher weight to ET and RF for high-cardinality dummy features)
    model = VotingRegressor([('rf', rf), ('et', et), ('cb', cb)], weights=[3, 4, 1])
    
    print("Fitting Ensemble Model (Random Forest + Extra Trees + CatBoost)...")
    model.fit(X_train, y_train)
    
    # 11. Evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    
    print("\n================ EVALUATION METRICS ================")
    print(f"Training R2 Score: {r2_train:.4f} ({r2_train*100:.2f}%)")
    print(f"Testing  R2 Score: {r2_test:.4f} ({r2_test*100:.2f}%)")
    print(f"Mean Absolute Error (MAE): {mae_test:.4f} Lakhs (Rs {mae_test*100000:,.0f})")
    print(f"Root Mean Squared Error (RMSE): {rmse_test:.4f} Lakhs")
    print("====================================================\n")
    
    # 12. Save Trained Model & Feature Columns
    joblib.dump(model, "rf_model.joblib")
    joblib.dump(X_train.columns.tolist(), "model_columns.joblib")
    
    print("Training Complete! Saved rf_model.joblib and model_columns.joblib")

if __name__ == "__main__":
    train_house_price_model()
