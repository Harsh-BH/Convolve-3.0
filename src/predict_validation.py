import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load

# Import preprocessing function
from data_preprocessing import preprocess_data

def predict_bad_flag(input_data_path):

    try:
        raw_data = pd.read_csv(input_data_path)
        print(f"Input data loaded successfully with shape: {raw_data.shape}")
    except FileNotFoundError:
        print(f"File not found: {input_data_path}")
        return None

    # Preprocess the data
    preprocessed_data = preprocess_data(raw_data)

    print(preprocessed_data)

    identifier_cols = ['account_number']
    features = preprocessed_data.drop(columns=identifier_cols)

    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Load pre-trained models
    try:
        model_xgb = load("models/xgb_model.joblib")
        model_cat = load("models/catboost_model.joblib")
        print("Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return None

    # Predict with both models
    predictions_xgb = model_xgb.predict(features_scaled)
    predictions_cat = model_cat.predict(features_scaled)

    # Add predictions to the DataFrame
    preprocessed_data['predictions_xgb'] = predictions_xgb
    preprocessed_data['predictions_cat'] = predictions_cat

    # Consensus prediction (e.g., majority voting)
    preprocessed_data['bad_flag'] = np.round((predictions_xgb + predictions_cat) / 2).astype(int)

    print("Predictions completed.")

    return preprocessed_data

result = predict_bad_flag("/home/harsh/Hackathons/Convolve/data/validation/validation_data_to_be_shared.csv")
result.to_csv("predictions_output.csv", index=False)
print("Predictions saved to predictions_output.csv.")
