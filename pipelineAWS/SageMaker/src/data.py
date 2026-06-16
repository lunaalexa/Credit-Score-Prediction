"""Load and split the Credit Score dataset."""
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    DATA_RAW_DIR,
    DROP_COLS,
    TARGET,
    NUM_FEATURES,
    CAT_FEATURES,
    RANDOM_STATE,
    TEST_SIZE
)

CLASS_NAMES = ["Good","Poor","Standard"]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataframe: fix types, drop useless cols, handle inconsistencies."""
    df = df.copy()

    numeric_fix_cols = [
        'Age', 'Annual_Income', 'Num_of_Loan',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
        'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance'
    ]
    for col in numeric_fix_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace('_', '', regex=False),
            errors='coerce'
        )

    cols_to_drop = [col for col in DROP_COLS if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    df['Occupation'] = df['Occupation'].replace('_______', np.nan)
    df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', np.nan)
    df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan)
    df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = np.nan
    df.loc[(df['Num_Bank_Accounts'] < 0) | (df['Num_Bank_Accounts'] > 20), 'Num_Bank_Accounts'] = np.nan
    df.loc[(df['Num_Credit_Card'] < 0) | (df['Num_Credit_Card'] > 20), 'Num_Credit_Card'] = np.nan
    df.loc[(df['Interest_Rate'] < 0) | (df['Interest_Rate'] > 100), 'Interest_Rate'] = np.nan
    df.loc[(df['Num_of_Loan'] < 0) | (df['Num_of_Loan'] > 20), 'Num_of_Loan'] = np.nan
    df.loc[df['Delay_from_due_date'] < 0, 'Delay_from_due_date'] = np.nan
    df.loc[(df['Num_of_Delayed_Payment'] < 0) | (df['Num_of_Delayed_Payment'] > 100), 'Num_of_Delayed_Payment'] = np.nan
    df.loc[df['Num_Credit_Inquiries'] > 100, 'Num_Credit_Inquiries'] = np.nan
    df.loc[df['Monthly_Balance'] < 0, 'Monthly_Balance'] = np.nan
    return df


def _convert_credit_history(age_text) -> float:
    if pd.isna(age_text):
        return np.nan
    match = re.search(r'(\d+)\s+Years?\s+and\s+(\d+)\s+Months?', str(age_text))
    if match:
        return int(match.group(1))*12+int(match.group(2))
    return np.nan


def _count_loans(x) -> int:
    loans = [loan.strip() for loan in str(x).split(',')]
    return len([loan for loan in loans if loan != 'Not Specified'])


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Credit_History_Age_Months'] = df['Credit_History_Age'].apply(_convert_credit_history)
    df['Type_of_Loan'] = df['Type_of_Loan'].fillna('Not Specified')
    df['Loan_Count'] = df['Type_of_Loan'].apply(_count_loans)
    df = df.drop(columns=['Credit_History_Age', 'Type_of_Loan'])
    return df


def load_dataset() -> pd.DataFrame:
    file_path = DATA_RAW_DIR / "data_D (1).csv"
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("Dataset is empty")
    df = clean_data(df)
    df = feature_engineering(df)
    return df


def split_data(df: pd.DataFrame):
    # encode + split
    X = df[NUM_FEATURES + CAT_FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_train, X_test, y_train, y_test, encoder
