import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW_DIR,
    DROP_COLS,
    TARGET,
    NUM_FEATURES,
    CAT_FEATURES,
    RANDOM_STATE,
    TEST_SIZE
)

# Load data
def load_frame():
    file_path = DATA_RAW_DIR / "data_D (1).csv"
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("Dataset is empty")
    df = clean_data(df)
    df = feature_engineering(df)
    return df

def clean_data(df):
    df = df.copy()

    # fix numerical data
    numeric_fix_cols = [
        'Age',
        'Annual_Income',
        'Num_of_Loan',
        'Num_of_Delayed_Payment',
        'Changed_Credit_Limit',
        'Outstanding_Debt',
        'Amount_invested_monthly',
        'Monthly_Balance'
    ]

    for col in numeric_fix_cols:
        df[col] = (df[col].astype(str).str.replace('_', '', regex=False))
        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    # drop useless features
    cols_to_drop = [
        col
        for col in DROP_COLS
        if col in df.columns
    ]

    df = df.drop(columns=cols_to_drop)

    # handle inconsistent data
    df['Occupation'] = (
        df['Occupation']
        .replace('_______', np.nan)
    )

    df['Credit_Mix'] = (
        df['Credit_Mix']
        .replace('_', np.nan)
    )

    df['Payment_of_Min_Amount'] = (
        df['Payment_of_Min_Amount']
        .replace('NM', np.nan)
    )

    df['Payment_Behaviour'] = (
        df['Payment_Behaviour']
        .replace('!@9#%8', np.nan)
    )

    # handle impossible value
    df.loc[
        (df['Age'] < 18) |
        (df['Age'] > 100),
        'Age'
    ] = np.nan

    df.loc[
        (df['Num_Bank_Accounts'] < 0) |
        (df['Num_Bank_Accounts'] > 20),
        'Num_Bank_Accounts'
    ] = np.nan

    df.loc[
        (df['Num_Credit_Card'] < 0) |
        (df['Num_Credit_Card'] > 20),
        'Num_Credit_Card'
    ] = np.nan

    df.loc[
        (df['Interest_Rate'] < 0) |
        (df['Interest_Rate'] > 100),
        'Interest_Rate'
    ] = np.nan

    df.loc[
        (df['Num_of_Loan'] < 0) |
        (df['Num_of_Loan'] > 20),
        'Num_of_Loan'
    ] = np.nan

    df.loc[
        df['Delay_from_due_date'] < 0,
        'Delay_from_due_date'
    ] = np.nan

    df.loc[
        (df['Num_of_Delayed_Payment'] < 0) |
        (df['Num_of_Delayed_Payment'] > 100),
        'Num_of_Delayed_Payment'
    ] = np.nan

    df.loc[
        df['Num_Credit_Inquiries'] > 100,
        'Num_Credit_Inquiries'
    ] = np.nan

    df.loc[
        df['Monthly_Balance'] < 0,
        'Monthly_Balance'
    ] = np.nan
    return df

# feature engineering
def convert_credit_history(age_text):
    if pd.isna(age_text):
        return np.nan
    match = re.search(
        r'(\d+)\s+Years?\s+and\s+(\d+)\s+Months?',
        str(age_text)
    )
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years * 12 + months
    return np.nan

def count_loans(x):
    loans = [loan.strip() for loan in str(x).split(',')]
    loans = [
        loan
        for loan in loans
        if loan != 'Not Specified'
    ]
    return len(loans)

def feature_engineering(df):
    df = df.copy()
    # Credit History Age → Months
    df['Credit_History_Age_Months'] = (df['Credit_History_Age'].apply(convert_credit_history))
    # Loan Count
    df['Type_of_Loan'] = (df['Type_of_Loan'].fillna('Not Specified'))
    df['Loan_Count'] = (df['Type_of_Loan'].apply(count_loans))
    # drop original columns
    df = df.drop(columns=['Credit_History_Age','Type_of_Loan'])
    return df

# split features target
def split_features_target(df):
    X = df[NUM_FEATURES+CAT_FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET])
    return X, y

# train test split
def split_train_test(X, y):
    return train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)