from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_ING_DIR = BASE_DIR / "data" / "ingested"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_BEST_MODEL = (ARTIFACTS_DIR / "creditScoreModel.pkl")
TARGET = "Credit_Score"

RANDOM_STATE = 42
TEST_SIZE = 0.2

DROP_COLS = [
    'Unnamed: 0',
    'ID',
    'Customer_ID',
    'Name',
    'SSN'
]

NUM_FEATURES = [
    'Age',
    'Annual_Income',
    'Monthly_Inhand_Salary',
    'Num_Bank_Accounts',
    'Num_Credit_Card',
    'Interest_Rate',
    'Num_of_Loan',
    'Delay_from_due_date',
    'Num_of_Delayed_Payment',
    'Changed_Credit_Limit',
    'Num_Credit_Inquiries',
    'Outstanding_Debt',
    'Credit_Utilization_Ratio',
    'Total_EMI_per_month',
    'Amount_invested_monthly',
    'Monthly_Balance',
    'Credit_History_Age_Months',
    'Loan_Count'
]

CAT_FEATURES = [
    'Month',
    'Occupation',
    'Credit_Mix',
    'Payment_of_Min_Amount',
    'Payment_Behaviour'
]

OPTUNA_LGBM_TRIALS = 15
CV_FOLDS = 5
