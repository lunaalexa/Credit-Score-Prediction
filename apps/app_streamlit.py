import sys
import pandas as pd
import streamlit as st
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent

if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from config.config import ARTIFACT_BEST_MODEL
from src.utils.io import load_artifact
from src.data.loader import clean_data, feature_engineering


@st.cache_resource
def load_model():
    return load_artifact(ARTIFACT_BEST_MODEL)


model = load_model()


def main():

    st.title("Credit Score Prediction")

    st.sidebar.header("Information")
    st.sidebar.write(
        "This system predicts customer credit score using the best trained machine learning model."
    )

    with st.form("credit_score_form"):

        st.subheader("Customer Data")

        col1, col2 = st.columns(2)

        with col1:

            customer_code = st.text_input(
                "ID",
                value="0x12x34"
            )

            customer_id = st.text_input(
                "Customer ID",
                value="CUS_0x1234"
            )

            customer_name = st.text_input(
                "Customer Name",
                value="John Doe"
            )

            ssn = st.text_input(
                "SSN",
                value="123-45-6789"
            )

            month = st.selectbox(
                "Month",
                [
                    "January", "February", "March",
                    "April", "May", "June",
                    "July", "August", "September",
                    "October", "November", "December"
                ]
            )

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30
            )

            occupation = st.text_input(
                "Occupation",
                value="Engineer"
            )

            annual_income = st.number_input(
                "Annual Income",
                min_value=0.0,
                value=50000.0
            )

            monthly_salary = st.number_input(
                "Monthly Inhand Salary",
                min_value=0.0,
                value=4000.0
            )

            num_bank_accounts = st.number_input(
                "Number of Bank Accounts",
                min_value=0,
                value=3
            )

            num_credit_card = st.number_input(
                "Number of Credit Cards",
                min_value=0,
                value=2
            )

            interest_rate = st.number_input(
                "Interest Rate",
                min_value=0.0,
                value=5.0
            )

            monthly_balance = st.number_input(
                "Monthly Balance",
                value=1000.0
            )

        with col2:

            num_of_loan = st.number_input(
                "Number of Loans",
                min_value=0,
                value=1
            )

            type_of_loan = st.text_input(
                "Type of Loan",
                value="Auto Loan"
            )

            delay_due = st.number_input(
                "Delay From Due Date",
                min_value=0,
                value=5
            )

            delayed_payment = st.number_input(
                "Number of Delayed Payments",
                min_value=0,
                value=2
            )

            changed_limit = st.number_input(
                "Changed Credit Limit",
                value=5.0
            )

            credit_inquiries = st.number_input(
                "Number of Credit Inquiries",
                min_value=0,
                value=2
            )

            credit_mix = st.selectbox(
                "Credit Mix",
                ["Bad", "Standard", "Good"]
            )

            outstanding_debt = st.number_input(
                "Outstanding Debt",
                min_value=0.0,
                value=1000.0
            )

            credit_utilization = st.number_input(
                "Credit Utilization Ratio",
                min_value=0.0,
                value=30.0
            )

            credit_history_age = st.text_input(
                "Credit History Age",
                value="10 Years and 5 Months"
            )

            payment_min_amount = st.selectbox(
                "Payment of Minimum Amount",
                ["No", "Yes"]
            )

            total_emi = st.number_input(
                "Total EMI per Month",
                min_value=0.0,
                value=200.0
            )

            amount_invested = st.number_input(
                "Amount Invested Monthly",
                min_value=0.0,
                value=500.0
            )

            payment_behaviour = st.selectbox(
                "Payment Behaviour",
                [
                    "Low_spent_Small_value_payments",
                    "Low_spent_Medium_value_payments",
                    "Low_spent_Large_value_payments",
                    "High_spent_Small_value_payments",
                    "High_spent_Medium_value_payments",
                    "High_spent_Large_value_payments"
                ]
            )


        submit = st.form_submit_button(
            "Predict Credit Score"
        )

    if submit:

        features = pd.DataFrame([{

            "ID": customer_code,
            "Customer_ID": customer_id,
            "Name": customer_name,
            "SSN": ssn,

            "Month": month,
            "Age": age,
            "Occupation": occupation,

            "Annual_Income": annual_income,
            "Monthly_Inhand_Salary": monthly_salary,

            "Num_Bank_Accounts": num_bank_accounts,
            "Num_Credit_Card": num_credit_card,

            "Interest_Rate": interest_rate,
            "Num_of_Loan": num_of_loan,

            "Type_of_Loan": type_of_loan,

            "Delay_from_due_date": delay_due,
            "Num_of_Delayed_Payment": delayed_payment,

            "Changed_Credit_Limit": changed_limit,
            "Num_Credit_Inquiries": credit_inquiries,

            "Credit_Mix": credit_mix,

            "Outstanding_Debt": outstanding_debt,
            "Credit_Utilization_Ratio": credit_utilization,

            "Credit_History_Age": credit_history_age,

            "Payment_of_Min_Amount": payment_min_amount,

            "Total_EMI_per_month": total_emi,
            "Amount_invested_monthly": amount_invested,

            "Payment_Behaviour": payment_behaviour,

            "Monthly_Balance": monthly_balance

        }])

        features = clean_data(features)
        features = feature_engineering(features)

        prediction = model.predict(features)[0]

        st.divider()

        label_map = {
            0: "Good",
            1: "Poor",
            2: "Standard"
        }

        result = label_map[prediction]

        st.subheader("Prediction Result")

        if result == "Good":
            st.success(f"Credit Score: {result}")

        elif result == "Standard":
            st.warning(f"Credit Score: {result}")

        else:
            st.error(f"Credit Score: {result}")


if __name__ == "__main__":
    main()