from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from config.config import NUM_FEATURES, CAT_FEATURES


class DataPreprocessor:

    def build_preprocessor(self):

        numeric_preprocessor = Pipeline([
            ("num_imputer", SimpleImputer(strategy="median"))
        ])

        ordinal_preprocessor = Pipeline([
            ("ord_imputer", SimpleImputer(strategy="most_frequent")),
            ("ord_encoder",
                OrdinalEncoder(
                    categories=[
                        ['Bad', 'Standard', 'Good'],
                        ['No', 'Yes']
                    ]
                )
            )
        ])

        onehot_preprocessor = Pipeline([
            ("cat_imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",
                OneHotEncoder(
                    drop='first',
                    handle_unknown='ignore'
                )
            )
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_preprocessor, NUM_FEATURES),
                (
                    "ordinal",
                    ordinal_preprocessor,
                    ['Credit_Mix', 'Payment_of_Min_Amount']
                ),
                (
                    "onehot",
                    onehot_preprocessor,
                    ['Month', 'Occupation', 'Payment_Behaviour']
                )
            ],
            remainder='drop'
        )

        return preprocessor