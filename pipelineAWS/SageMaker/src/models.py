"""Build the sklearn Pipeline for the Credit Score LGBM model.
Pipeline shape: ColumnTransformer (preprocessing) -> LGBMClassifier
Preprocessing:
  - Numeric features : median imputation
  - Ordinal features : most_frequent imputation + OrdinalEncoder
  - Nominal features : most_frequent imputation + OneHotEncoder(drop='first')
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from lightgbm import LGBMClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import NUM_FEATURES, RANDOM_STATE


ORDINAL_FEATURES = ['Credit_Mix', 'Payment_of_Min_Amount']
ONEHOT_FEATURES  = ['Month', 'Occupation', 'Payment_Behaviour']


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("num_imputer", SimpleImputer(strategy="median"))
    ])

    ordinal_pipe = Pipeline([
        ("ord_imputer", SimpleImputer(strategy="most_frequent")),
        ("ord_encoder", OrdinalEncoder(categories=[['Bad','Standard','Good'], ['No','Yes']]))
    ])

    onehot_pipe = Pipeline([
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUM_FEATURES),
            ("ordinal", ordinal_pipe, ORDINAL_FEATURES),
            ("onehot", onehot_pipe, ONEHOT_FEATURES),
        ],
        remainder='drop'
    )


def build_pipeline(lgbm_params: dict = None) -> Pipeline:
    if lgbm_params is None:
        lgbm_params = {}

    default_params = {
        "class_weight":"balanced",
        "random_state":RANDOM_STATE,
        "verbose":-1,
    }
    default_params.update(lgbm_params)

    return Pipeline([
        ("preprocessor",build_preprocessor()),
        ("classifier",LGBMClassifier(**default_params)),
    ])
