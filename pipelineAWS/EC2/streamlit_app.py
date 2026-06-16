"""
Streamlit UI for the Credit Score classifier hosted on SageMaker.

Reads endpoint name and region from environment variables.
boto3 picks up AWS credentials from:
  - the EC2 instance profile (when running on EC2 with LabInstanceProfile), OR
  - ~/.aws/credentials (when running locally)
"""

import json
import os
import re

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

CLASS_NAMES   = ["Good","Poor","Standard"]


def _convert_credit_history(age_text: str) -> float:
    """Convert 'X Years and Y Months' """
    if not age_text or age_text.strip()=="":
        return 0.0
    match = re.search(r'(\d+)\s+Years?\s+and\s+(\d+)\s+Months?', str(age_text))
    if match:
        return int(match.group(1))*12+int(match.group(2))
    return 0.0


def _count_loans(type_of_loan: str) -> int:
    """Count distinct loan types, excluding 'Not Specified'"""
    if not type_of_loan or type_of_loan.strip() == "":
        return 0
    loans = [loan.strip() for loan in type_of_loan.split(",")]
    return len([loan for loan in loans if loan != "Not Specified"])


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: list) -> dict:
    runtime  = get_runtime_client()
    payload  = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


# config
st.set_page_config(page_title="Credit Score Prediction")
st.title("Credit Score Prediction")

st.sidebar.header("Information")
st.sidebar.write(
    "This system predicts customer credit score using the best trained machine learning model."
)

# form
with st.form("credit_score_form"):
    st.subheader("Customer Data")
    col1, col2 = st.columns(2)

    with col1:
        customer_code = st.text_input("ID",  value="0x12x34")
        customer_id = st.text_input("Customer ID",value="CUS_0x1234")
        customer_name = st.text_input("Customer Name",value="John Doe")
        ssn = st.text_input("SSN",value="123-45-6789")
        month = st.selectbox(
            "Month",
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]
        )

        age        = st.number_input("Age", min_value=18, max_value=100, value=30)
        occupation = st.text_input( "Occupation",value="Engineer")

        annual_income= st.number_input("Annual Income",min_value=0.0, value=50000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary",min_value=0.0,value=4000.0)
        num_bank_accounts = st.number_input("Number of Bank Accounts",min_value=0, value=3)
        num_credit_card = st.number_input("Number of Credit Cards",min_value=0,value=2)
        interest_rate = st.number_input("Interest Rate",min_value=0.0, value=5.0)
        monthly_balance = st.number_input("Monthly Balance",value=1000.0)

    with col2:
        num_of_loan = st.number_input("Number of Loans",min_value=0, value=1)
        type_of_loan = st.text_input( "Type of Loan",value="Auto Loan")

        delay_due= st.number_input("Delay From Due Date",min_value=0, value=5)
        delayed_payment= st.number_input("Number of Delayed Payments", min_value=0, value=2)
        changed_limit= st.number_input("Changed Credit Limit", value=5.0)
        credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=2)

        credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])

        outstanding_debt = st.number_input("Outstanding Debt",min_value=0.0, value=1000.0)
        credit_utilization = st.number_input("Credit Utilization Ratio",min_value=0.0, value=30.0)
        credit_history_age = st.text_input( "Credit History Age", value="10 Years and 5 Months")

        payment_min_amount = st.selectbox("Payment of Minimum Amount", ["No", "Yes"])

        total_emi= st.number_input("Total EMI per Month",min_value=0.0, value=200.0)
        amount_invested = st.number_input("Amount Invested Monthly",min_value=0.0, value=500.0)

        payment_behaviour = st.selectbox(
            "Payment Behaviour",
            [
                "Low_spent_Small_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Large_value_payments",
                "High_spent_Small_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Large_value_payments",
            ]
        )

    submit = st.form_submit_button("Predict Credit Score")


# predict
if submit:
    credit_history_months = _convert_credit_history(credit_history_age)
    loan_count = _count_loans(type_of_loan)

    features = [
        # Numeric (18)
        age, annual_income, monthly_salary,
        num_bank_accounts, num_credit_card, interest_rate,
        num_of_loan, delay_due, delayed_payment,
        changed_limit, credit_inquiries, outstanding_debt,
        credit_utilization, total_emi, amount_invested,
        monthly_balance, credit_history_months, loan_count,
        # Categorical (5)
        month, occupation, credit_mix, payment_min_amount, payment_behaviour,
    ]

    try:
        result = invoke_endpoint(features)
    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label = result["labels"][0]

        st.divider()
        st.subheader("Prediction Result")

        if label == "Good":
            st.success(f"Credit Score: {label}")
        elif label == "Standard":
            st.warning(f"Credit Score: {label}")
        else:
            st.error(f"Credit Score: {label}")