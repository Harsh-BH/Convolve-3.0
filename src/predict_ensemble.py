import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

def predict_bad_flag_ensemble(
    df_new,
    model_paths,
    transaction_cols,
    bureau_cols,
    bureau_enquiry_cols,
    onus_cols
):
    """
    1) Loads four saved CatBoost models (transaction, bureau, bureau_enquiry, onus).
    2) Applies minimal preprocessing (fillna, scaling).
    3) Averages the predicted probabilities from all four models to get an ensemble output.
    4) Returns (y_pred_ensemble_prob, y_pred_ensemble_bin).

    You do NOT need a 'bad_flag' column in df_new for this function.
    """

    # 1. Load models
    model_transaction = joblib.load(model_paths['transaction'])
    model_bureau = joblib.load(model_paths['bureau'])
    model_bureau_enq = joblib.load(model_paths['bureau_enquiry'])
    model_onus = joblib.load(model_paths['onus'])

    # 2. Preprocess new data: fill missing values, scale
    df_new = df_new.copy()
    df_new = df_new.fillna(0)

    # In production, you would load the SAME scaler used during training
    # For simplicity, we'll just fit a new scaler here:
    scaler = MinMaxScaler()
    X_new_scaled = scaler.fit_transform(df_new)
    X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=df_new.columns, index=df_new.index)

    # 3. Subset columns for each model
    X_new_trans = X_new_scaled_df[transaction_cols]
    X_new_bur = X_new_scaled_df[bureau_cols]
    X_new_bur_enq = X_new_scaled_df[bureau_enquiry_cols]
    X_new_onus = X_new_scaled_df[onus_cols]

    # 4. Predict probabilities from each model and average
    y_pred_trans_prob = model_transaction.predict_proba(X_new_trans)[:, 1]
    y_pred_bur_prob = model_bureau.predict_proba(X_new_bur)[:, 1]
    y_pred_bur_enq_prob = model_bureau_enq.predict_proba(X_new_bur_enq)[:, 1]
    y_pred_onus_prob = model_onus.predict_proba(X_new_onus)[:, 1]

    y_pred_ensemble_prob = (
        y_pred_trans_prob
        + y_pred_bur_prob
        + y_pred_bur_enq_prob
        + y_pred_onus_prob
    ) / 4.0

    # # Binary prediction at 0.5 threshold
    # y_pred_ensemble_bin = (y_pred_ensemble_prob > 0.5).astype(int)

    return y_pred_ensemble_prob


if __name__ == "__main__":

    # Define the file paths to your saved models
    model_paths = {
        'transaction': "/home/harsh/Hackathons/Convolve/models/model_transaction.joblib",
        'bureau': "/home/harsh/Hackathons/Convolve/models/model_bureau.joblib",
        'bureau_enquiry': "/home/harsh/Hackathons/Convolve/models/model_bureau_enq.joblib",
        'onus': "/home/harsh/Hackathons/Convolve/models/model_onus.joblib"
    }

    # Load new data (MUST have the same columns you used during training)
    new_df = pd.read_csv("/home/harsh/Hackathons/Convolve/data/validation/validation_data_to_be_shared.csv")

    # Identify columns for each subset
    transaction_cols = [col for col in new_df.columns if col.startswith("transaction_attribute")]
    bureau_cols = [col for col in new_df.columns if col.startswith("bureau")]
    bureau_enquiry_cols = [col for col in new_df.columns if col.startswith("bureau_enquiry")]
    onus_cols = [col for col in new_df.columns if col.startswith("onus")]

    # Predict using the ensemble
    y_pred_probs = predict_bad_flag_ensemble(
        df_new=new_df,
        model_paths=model_paths,
        transaction_cols=transaction_cols,
        bureau_cols=bureau_cols,
        bureau_enquiry_cols=bureau_enquiry_cols,
        onus_cols=onus_cols
    )

    # Show results
    print("Predicted Probabilities (first 10):", y_pred_probs[:10])


    # Optionally, add predicted classes as a column in your data
    new_df["bad_flag"] = y_pred_probs
    result_df = new_df[["account_number", "bad_flag"]]
    result_df.to_csv("ensemble_predictions.csv", index=False)
    print("\nPredictions saved to ensemble_predictions.csv")
