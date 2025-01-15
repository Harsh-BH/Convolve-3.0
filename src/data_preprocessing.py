import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data(dev_data_path):
    try:
        df = pd.read_csv(dev_data_path)
        print(f"Development data loaded successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found at {dev_data_path}. Please check the path and try again.")
        return None

def preprocess_data(raw_data):
    # Load the development data
    def handle_missing_values(df):
       
        df = df.fillna(0)  # Fill remaining NaN values with 0
        return df

    # Impute numerical columns
    def impute_numerical_columns(df, numerical_cols):
        # No longer using SimpleImputer, as NaN values are filled with 0 in handle_missing_values
        return df

    # Encode categorical variables
    def encode_categorical_columns(df, categorical_cols):
        binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
        for col in binary_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        multi_cat_cols = [col for col in categorical_cols if df[col].nunique() > 2]
        df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
        return df

    def cap_outliers(df, numerical_cols, lower_q=0.01, upper_q=0.99):
        for col in numerical_cols:
            lower_val = df[col].quantile(lower_q)
            upper_val = df[col].quantile(upper_q)
            df[col] = np.clip(df[col], lower_val, upper_val)
        return df

    def aggregate_prefix_columns(df, prefix_cols, prefix_name):
        if not prefix_cols:
            return pd.DataFrame(index=df.index)

        subset = df[prefix_cols].astype(float)
        agg_dict = {
            'sum': subset.sum(axis=1),
            'mean': subset.mean(axis=1),
            'max': subset.max(axis=1),
            'min': subset.min(axis=1),
        }
        agg_df = pd.DataFrame(agg_dict)
        agg_df.columns = [f'{prefix_name}_{agg_col}' for agg_col in agg_df.columns]

        return agg_df

    def create_prefix_features(df, onus_cols, txn_cols, bureau_cols, bureau_enquiry_cols):
        onus_agg = aggregate_prefix_columns(df, onus_cols, 'onus')
        txn_agg = aggregate_prefix_columns(df, txn_cols, 'txn')
        bureau_agg = aggregate_prefix_columns(df, bureau_cols, 'bureau')
        bureau_enquiry_agg = aggregate_prefix_columns(df, bureau_enquiry_cols, 'bureau_enquiry')

        all_agg = pd.concat([onus_agg, txn_agg, bureau_agg, bureau_enquiry_agg], axis=1)
        df = pd.concat([df, all_agg], axis=1)
        return df

    # Create cross-prefix features
    def create_cross_prefix_features(df):
        if 'onus_sum' in df.columns and 'txn_sum' in df.columns:
            df['feat_ratio_txn_onus_sum'] = np.where(
                df['onus_sum'] == 0, 0, df['txn_sum'] / (df['onus_sum'] + 1e-9)
            )

        if 'bureau_mean' in df.columns and 'bureau_enquiry_mean' in df.columns:
            df['feat_diff_bureau_vs_enquiry_mean'] = df['bureau_mean'] - df['bureau_enquiry_mean']

        if 'bureau_sum' in df.columns and 'bureau_enquiry_sum' in df.columns:
            df['feat_ratio_bureau_enquiry_sum'] = np.where(
                df['bureau_enquiry_sum'] == 0, 0, df['bureau_sum'] / (df['bureau_enquiry_sum'] + 1e-9)
            )
        return df

    def apply_log_transform(df):
        log_cols = [c for c in df.columns if any(x in c for x in ['_sum', '_mean', '_max'])]
        for col in log_cols:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
        return df

    # Apply binning
    def apply_binning(df):
        if 'txn_sum' in df.columns:
            df['bin_txn_sum_quartile'] = pd.qcut(df['txn_sum'], q=4, labels=False, duplicates='drop')
        return df

    # Begin preprocessing
    df = raw_data.copy()
    df = handle_missing_values(df)

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'bad_flag' in numerical_cols:
        numerical_cols.remove('bad_flag')

    df = impute_numerical_columns(df, numerical_cols)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'bad_flag' in categorical_cols:
        categorical_cols.remove('bad_flag')

    df = encode_categorical_columns(df, categorical_cols)
    df = cap_outliers(df, numerical_cols)

    onus_cols = [c for c in df.columns if c.startswith('onus_attributes_')]
    txn_cols = [c for c in df.columns if c.startswith('transaction_attribute_')]
    bureau_cols = [c for c in df.columns if c.startswith('bureau_') and not c.startswith('bureau_enquiry_')]
    bureau_enquiry_cols = [c for c in df.columns if c.startswith('bureau_enquiry_')]

    df = create_prefix_features(df, onus_cols, txn_cols, bureau_cols, bureau_enquiry_cols)
    df = create_cross_prefix_features(df)
    df = apply_log_transform(df)
    df = apply_binning(df)

    return df
