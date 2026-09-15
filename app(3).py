import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

FOLDER_PATH = r"sample_data"

print(f"Scanning the '{FOLDER_PATH}' folder for clinical neurodegenerative assets...")

if not os.path.exists(FOLDER_PATH):
    print(f"Warning: Directory missing. Please upload your CSVs to the left sidebar.")
else:
    all_csvs = glob.glob(os.path.join(FOLDER_PATH, "*.csv"))
    model_suite = {}
    
    # Filter strictly to the PRNP and Motor Data CSV files
    target_files = [f for f in all_csvs if "variants" in os.path.basename(f).lower() or "prnp" in os.path.basename(f).lower() or "data.csv" in os.path.basename(f).lower()]
    
    print(f"Isolating Bimodal datasets for training: {[os.path.basename(f) for f in target_files]}")
    
    for csv_file in target_files:
        fname = os.path.basename(csv_file)
        
        print(f"\n==================================================")
        print(f"PIPELINE INITIALIZED FOR: {fname}")
        print(f"==================================================")
        
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig', low_memory=False)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '')
            
            # --- MODEL MODULE A: PRNP GENETIC VARIANT ANALYSIS ---
            if "variants" in fname or "prnp" in fname:
                print(" -> Processing text parameters using categorical one-hot encoding matrices...")
                
                # Convert descriptive text columns into numeric matrices (0s and 1s)
                categorical_cols = ['prnp_mutation', 'clinical_significance', 'condition', 'cdna_change']
                existing_cats = [c for c in categorical_cols if c in df.columns]
                
                df_encoded = pd.get_dummies(df, columns=existing_cats, drop_first=False)
                df_numeric = df_encoded.apply(pd.to_numeric, errors='coerce').dropna(how='all', axis=1)
                
                # Define target vector parameter
                target_col = 'vcv000013401' if 'vcv000013401' in df_numeric.columns else df_numeric.columns[-1]
            
            # --- MODEL MODULE B: MOTOR REGISTRY ANALYSIS ---
            else:
                print(" -> Processing motor kinetics parameters...")
                df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(how='all', axis=1)
                target_col = 'total_time25' if 'total_time25' in df_numeric.columns else df_numeric.columns[-1]
            
            df_clean = df_numeric.dropna(how='all', axis=0)
            
            if df_clean.shape[0] < 5:
                print(f"Warning: Skipping {fname}: Not enough dense rows to compute dependencies.")
                continue
                
            feature_cols = [c for c in df_clean.columns if c != target_col]
            
            X = df_clean[feature_cols].fillna(0).to_numpy()
            y = df_clean[target_col].fillna(df_clean[target_col].mean()).to_numpy()
            
            print(f"Extracted Matrix Shape: {X.shape[0]} samples, {X.shape[1]} clinical biological features.")
            print(f"Target Prediction Variable: '{target_col}'")
            
            # Split: 80% to train structural connections, 20% to validate performance metrics
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the ensemble decision network
            ai_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            ai_model.fit(X_train, y_train)
            
            # Cross-Validation Analysis
            predictions = ai_model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            
            try:
                r2 = r2_score(y_test, predictions)
                r2_str = f"{r2:.4f}"
            except:
                r2_str = "N/A (Validation variance too low)"
            
            print(f"\nEvaluation Metrics Summary for {fname}:")
            print(f"   -> Mean Absolute Error (MAE): {mae:.4f}")
            print(f"   -> R2 Architecture Performance Score: {r2_str}")
            
            # Save elements to model directory dictionary using unified clean strings
            suite_key = 'real_prnp_variants_dataset.csv' if ("variants" in fname or "prnp" in fname) else 'data.csv'
            model_suite[suite_key] = {
                'model': ai_model,
                'features': feature_cols,
                'target': target_col,
                'mae': float(mae)
            }
            
        except Exception as e:
            print(f"Error: Failed to parse file structures due to anomaly: {e}")
            
    # =====================================================================
    # FREEZING THE DATA CORE
    # =====================================================================
    print(f"\n==================================================")
    if len(model_suite) > 0:
        # Save as standard backup asset and clear duplicate file variant
        joblib.dump(model_suite, 'prion_clinical_suite.pkl')
        joblib.dump(model_suite, 'prion_clinical_suite(2).pkl')
        print(f"Success: Successfully trained Bimodal clinical network model variants.")
        print(f"Diagnostic structures are unified and frozen into: 'prion_clinical_suite(2).pkl'")
