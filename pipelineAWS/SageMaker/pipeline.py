"""Train LGBM Tuned model, and package.

Saves the winning sklearn Pipeline as model_artifact/model.tar.gz, ready to
upload to S3 for SageMaker deployment.

Run from a SageMaker Notebook Instance:
    pip install lightgbm scikit-learn pandas joblib optuna
    python pipeline.py
"""

import os
import sys
import tarfile

import joblib
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from src.data import CLASS_NAMES, load_dataset, split_data
from src.evaluate import evaluate_pipeline, print_classification_report, print_comparison
from src.models import build_pipeline

from config.config import RANDOM_STATE, OPTUNA_LGBM_TRIALS, CV_FOLDS


ARTIFACT_DIR   = "model_artifact"
MODEL_FILENAME = "model.joblib"
TARBALL_PATH   = os.path.join(ARTIFACT_DIR, "model.tar.gz")


def optimize_lgbm(X_train, y_train) -> dict:
    """Run Optuna hyperparameter search for LGBMClassifier. Returns best params."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 100),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        pipeline = build_pipeline(params)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_LGBM_TRIALS)

    print(f"Best LGBM Macro F1 (CV): {study.best_value:.4f}")
    print(f"Best LGBM Params: {study.best_params}")

    return study.best_params


def main() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")

    X_train, X_test, y_train, y_test, encoder = split_data(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    label_counts = {name: int((y_train == i).sum()) for i, name in enumerate(encoder.classes_)}
    print(f"Class distribution (train): {label_counts}")

    # Tune LGBM with Optuna
    print("\nOptimizing LGBM with Optuna...")
    best_params = optimize_lgbm(X_train, y_train)

    # Train final pipeline with best params
    print("\nTraining final LGBM Tuned pipeline...")
    pipeline = build_pipeline(best_params)
    pipeline.fit(X_train, y_train)

    results = {"LGBM Tuned": evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test)}
    print_comparison(results)

    print(f"\nDetailed report for LGBM Tuned:")
    print_classification_report(pipeline, X_test, y_test, list(encoder.classes_))

    # Save pipeline
    model_path = os.path.join(ARTIFACT_DIR, MODEL_FILENAME)
    joblib.dump(pipeline, model_path)
    print(f"\nSaved: {model_path}")

    with tarfile.open(TARBALL_PATH, "w:gz") as tar:
        tar.add(model_path, arcname=MODEL_FILENAME)
    print(f"Packaged: {TARBALL_PATH}")

    print("\nNext steps:")
    print("  Make the S3 bucket with:\n")
    print("  aws s3 mb s3://your-bucket-name --region us-east-1\n")
    print("  Then upload with:")
    print(f"\n  aws s3 cp {TARBALL_PATH} s3://your-bucket-name/credit-score/model.tar.gz\n")


if __name__ == "__main__":
    main()
