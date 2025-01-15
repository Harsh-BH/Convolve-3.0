from data_preprocessing import preprocess_data , load_data
from evaluate_model import evaluate_model

import pandas as pd



def main():
    # Load data
    raw_data = load_data("/home/harsh/Hackathons/Convolve/data/dev/Dev_data_to_be_shared.csv")

    # Preprocess data
    preprocessed_data = preprocess_data(raw_data)
    print(preprocessed_data)

    # Train and evaluate model
    model_xgb, model_cat = evaluate_model(preprocessed_data)

    print(model_xgb)
    print(model_cat)

if __name__ == "__main__":
    main()
