"""SageMaker inference entry point.

Generic sklearn Pipeline loader. Works for any pipeline saved as model.joblib,
regardless of which classifier is inside.

Four functions form the SageMaker contract:
    model_fn   - load model from disk (called once per container)
    input_fn   - parse request body (called per request)
    predict_fn - run inference (called per request)
    output_fn  - serialize response (called per request)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd


JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE  = "text/csv"

CLASS_NAMES = ["Good", "Poor", "Standard"]

FEATURE_NAMES = [
    # numeric
    'Age', 'Annual_Income', 'Monthly_Inhand_Salary',
    'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
    'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
    'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt',
    'Credit_Utilization_Ratio', 'Total_EMI_per_month',
    'Amount_invested_monthly', 'Monthly_Balance',
    'Credit_History_Age_Months', 'Loan_Count',
    # categorical
    'Month', 'Occupation', 'Credit_Mix',
    'Payment_of_Min_Amount', 'Payment_Behaviour',
]


def model_fn(model_dir: str):
    """Load the pickled sklearn Pipeline from model.joblib."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    """Parse incoming request body into a DataFrame.

    Accepts JSON: {"instances": [{"Age": 28, "Annual_Income": 50000, ...}, ...]}
    Or CSV: one row per instance, values in FEATURE_NAMES order.
    """
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        if instances and isinstance(instances[0], dict):
            return pd.DataFrame(instances)
        return pd.DataFrame(instances, columns=FEATURE_NAMES)

    if request_content_type == CSV_CONTENT_TYPE:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
        rows = [
            line.split(",")
            for line in request_body.strip().splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows, columns=FEATURE_NAMES)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: pd.DataFrame, pipeline) -> dict:
    """Run inference. Returns probabilities, predicted class IDs, and labels."""
    probs    = pipeline.predict_proba(input_data)
    class_ids = np.argmax(probs, axis=1)
    labels   = [CLASS_NAMES[int(i)] for i in class_ids]
    return {
        "probabilities": probs.tolist(),
        "predictions": class_ids.tolist(),
        "labels": labels,
    }


def output_fn(prediction: dict, accept_content_type: str):
    """Serialize the prediction dict for the response body."""
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")
