from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


from src.utils.io import save_artifact
import optuna

from sklearn.model_selection import (StratifiedKFold,cross_val_score)

from config.config import (
    RANDOM_STATE,
    ARTIFACT_BEST_MODEL,
    OPTUNA_RF_TRIALS,
    OPTUNA_LGBM_TRIALS,
    CV_FOLDS
)

from src.pipelines.sklearn_pipeline import (PipelineFactory)

class ModelTrainer:

    def train_baseline_models(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        factory = PipelineFactory()

        results = {}

        # RF Baseline
        print("\nTraining RF Baseline...")

        rf_pipeline = factory.create_pipeline(
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced"
            )
        )

        rf_pipeline.fit(X_train, y_train)

        rf_pred = rf_pipeline.predict(X_test)

        rf_train_pred = rf_pipeline.predict(X_train)

        rf_train_acc = accuracy_score(
            y_train,
            rf_train_pred
        )


        rf_f1 = f1_score(
            y_test,
            rf_pred,
            average="macro"
        )

        rf_acc = accuracy_score(
            y_test,
            rf_pred
        )

        results["RF Baseline"] = {
            "model": rf_pipeline,
            "train_accuracy": rf_train_acc,
            "accuracy": rf_acc,
            "macro_f1": rf_f1
        }

    

        # LGBM Baseline
        print("\nTraining LGBM Baseline...")

        lgb_pipeline = factory.create_pipeline(
            LGBMClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                verbose=-1
            )
        )

        lgb_pipeline.fit(X_train, y_train)

        lgb_pred = lgb_pipeline.predict(X_test)

        lgb_train_pred = lgb_pipeline.predict(X_train)

        lgb_train_acc = accuracy_score(
            y_train,
            lgb_train_pred
        )

        lgb_f1 = f1_score(
            y_test,
            lgb_pred,
            average="macro"
        )

        lgb_acc = accuracy_score(
            y_test,
            lgb_pred
        )

        results["LGBM Baseline"] = {
            "model": lgb_pipeline,
            "train_accuracy": lgb_train_acc,
            "accuracy": lgb_acc,
            "macro_f1": lgb_f1
        }
        return results

    def optimize_rf(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        print("\nOptimizing RF...")

        cv_clf = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        factory = PipelineFactory()

        def objective_rf(trial):

            params = {
                'n_estimators': trial.suggest_int(
                    'n_estimators',
                    100,
                    300
                ),
                'max_depth': trial.suggest_int(
                    'max_depth',
                    5,
                    20
                ),
                'min_samples_split': trial.suggest_int(
                    'min_samples_split',
                    2,
                    10
                ),
                'min_samples_leaf': trial.suggest_int(
                    'min_samples_leaf',
                    1,
                    8
                ),
                'class_weight': 'balanced',
                'random_state': RANDOM_STATE
            }

            pipeline = factory.create_pipeline(
                RandomForestClassifier(**params)
            )

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv_clf,
                scoring='f1_macro'
            )

            return scores.mean()

        study_rf = optuna.create_study(
            direction='maximize'
        )

        study_rf.optimize(
            objective_rf,
            n_trials=OPTUNA_RF_TRIALS
        )

        print(
            f"Best RF Macro F1 (CV): "
            f"{study_rf.best_value:.4f}"
        )

        print(
            f"Best RF Params: "
            f"{study_rf.best_params}"
        )

        best_pipeline = factory.create_pipeline(
            RandomForestClassifier(
                **study_rf.best_params,
                class_weight='balanced',
                random_state=RANDOM_STATE
            )
        )

        best_pipeline.fit(
            X_train,
            y_train
        )

        train_pred = best_pipeline.predict(X_train)

        train_acc = accuracy_score(
            y_train,
            train_pred
        )

        pred = best_pipeline.predict(X_test)

        return {
            "model": best_pipeline,
            "train_accuracy": train_acc,
            "accuracy": accuracy_score(
                y_test,
                pred
            ),
            "macro_f1": f1_score(
                y_test,
                pred,
                average='macro'
            ),
            "best_params": study_rf.best_params
        }

    def optimize_lgbm(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        print("\nOptimizing LGBM...")

        cv_clf = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        factory = PipelineFactory()

        def objective_lgb(trial):

            params = {
                'n_estimators': trial.suggest_int(
                    'n_estimators',
                    100,
                    500
                ),
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    0.01,
                    0.3,
                    log=True
                ),
                'num_leaves': trial.suggest_int(
                    'num_leaves',
                    15,
                    100
                ),
                'max_depth': trial.suggest_int(
                    'max_depth',
                    3,
                    12
                ),
                'min_child_samples': trial.suggest_int(
                    'min_child_samples',
                    5,
                    50
                ),
                'subsample': trial.suggest_float(
                    'subsample',
                    0.6,
                    1.0
                ),
                'colsample_bytree': trial.suggest_float(
                    'colsample_bytree',
                    0.6,
                    1.0
                ),
                'class_weight': 'balanced',
                'random_state': RANDOM_STATE,
                'verbose': -1
            }

            pipeline = factory.create_pipeline(LGBMClassifier(**params))

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv_clf,
                scoring='f1_macro'
            )

            return scores.mean()

        study_lgb = optuna.create_study(
            direction='maximize'
        )

        study_lgb.optimize(
            objective_lgb,
            n_trials=OPTUNA_LGBM_TRIALS
        )

        print(
            f"Best LGBM Macro F1 (CV): "
            f"{study_lgb.best_value:.4f}"
        )

        print(
            f"Best LGBM Params: "
            f"{study_lgb.best_params}"
        )


        best_pipeline = factory.create_pipeline(LGBMClassifier(
            **study_lgb.best_params,
            class_weight='balanced',
            random_state=RANDOM_STATE, verbose=-1))

        best_pipeline.fit(
            X_train,
            y_train
        )

        train_pred = best_pipeline.predict(X_train)

        train_acc = accuracy_score(
            y_train,
            train_pred
        )        

        pred = best_pipeline.predict(X_test)

        return {
            "model": best_pipeline,
            "train_accuracy": train_acc,
            "accuracy": accuracy_score(
                y_test,
                pred
            ),
            "macro_f1": f1_score(
                y_test,
                pred,
                average='macro'
            ),
            "best_params": study_lgb.best_params
        }






    def select_best_model(
        self,
        results
    ):

        best_model_name = max(
            results,
            key=lambda x:
                results[x]["macro_f1"]
        )

        best_model = (
            results[best_model_name]
            ["model"]
        )

        print(
            f"\nBest Model: "
            f"{best_model_name}"
        )

        print(
            f"Best Macro F1: "
            f"{results[best_model_name]['macro_f1']:.4f}"
        )

        save_artifact(
            best_model,
            ARTIFACT_BEST_MODEL
        )

        return (
            best_model,
            best_model_name,
            results
        )