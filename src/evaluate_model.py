from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, mean_absolute_error
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from joblib import dump

# Define the evaluate_model function
def evaluate_model(df):
    # Define target column and features
    identifier_cols = ['account_number']
    target_col = 'bad_flag'
    features = df.drop(columns=[target_col] + identifier_cols, errors='ignore')
    target = df[target_col]

    # # Handle class imbalance using SMOTE
    # smote = SMOTE(random_state=42)
    # features, target = smote.fit_resample(features, target)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42
    )

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance (for demonstration)
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    print("scale_pos_weight:", scale_pos_weight)

    # XGBoost Model
    best_params_xgb = {
        'colsample_bytree': 1.0,
        'learning_rate': 0.1,
        'max_depth': 3,
        'n_estimators': 100,
        'reg_alpha': 0,
        'reg_lambda': 10,
        'subsample': 1.0
    }

    model_xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42,
        **best_params_xgb
    )

    model_xgb.fit(X_train_scaled, y_train)

    # Predict probabilities and labels on the validation set
    y_val_pred_prob_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]
    y_val_pred_xgb = model_xgb.predict(X_test_scaled)

    # Evaluate XGBoost
    auc_xgb = roc_auc_score(y_test, y_val_pred_prob_xgb)
    mae_xgb = mean_absolute_error(y_test, y_val_pred_prob_xgb) * 100

    print("===== XGBoost Results =====")
    print(f"AUC-ROC: {auc_xgb}")
    print(f"MAE: {mae_xgb}")
    print("\nClassification Report (XGBoost):")
    print(classification_report(y_test, y_val_pred_xgb))

    conf_matrix_xgb = confusion_matrix(y_test, y_val_pred_xgb)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_xgb, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Bad", "Bad"], yticklabels=["No Bad", "Bad"])
    plt.title("Confusion Matrix - XGBoost")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # CatBoost Model
    best_params_cat = {
        'iterations': 100,
        'depth': 4,
        'learning_rate': 0.1,
        'random_seed': 42,
        'l2_leaf_reg': 3,
        'thread_count': -1,
        'eval_metric': 'AUC'
    }

    model_cat = CatBoostClassifier(
        **best_params_cat,
        class_weights=[1, scale_pos_weight]
    )

    model_cat.fit(
        X_train_scaled,
        y_train,
        eval_set=(X_test_scaled, y_test),
        verbose=False
    )

    y_val_pred_prob_cat = model_cat.predict_proba(X_test_scaled)[:, 1]
    y_val_pred_cat = model_cat.predict(X_test_scaled)

    auc_cat = roc_auc_score(y_test, y_val_pred_prob_cat)
    mae_cat = mean_absolute_error(y_test, y_val_pred_prob_cat) * 100

    print("\n===== CatBoost Results =====")
    print(f"AUC-ROC: {auc_cat}")
    print(f"MAE: {mae_cat}")
    print("\nClassification Report (CatBoost):")
    print(classification_report(y_test, y_val_pred_cat))

    conf_matrix_cat = confusion_matrix(y_test, y_val_pred_cat)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_cat, annot=True, fmt="d", cmap="Purples",
                xticklabels=["No Bad", "Bad"], yticklabels=["No Bad", "Bad"])
    plt.title("Confusion Matrix - CatBoost")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Save models (optional)
    dump(model_xgb, "models/xgb_model.joblib")
    dump(model_cat, "models/catboost_model.joblib")

    return model_xgb, model_cat

# Example usage
# model_xgb, model_cat = evaluate_model(preprocessed_data)
