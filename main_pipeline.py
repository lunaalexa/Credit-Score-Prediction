from src.data.loader import (load_frame,split_features_target,split_train_test)
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
import mlflow
import mlflow.sklearn

def main():

    print("=" * 50)
    print("Credit Score Classification Pipeline")
    print("=" * 50)

    print("\nStep 1: Load Data")
    df = load_frame()

    print("\nStep 2: Split Features & Target")
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = (
        split_train_test(X, y)
    )

    trainer = ModelTrainer()

    with mlflow.start_run(
        run_name="CreditScorePipeline"
    ):

        print("\nStep 3: Train Baseline Models")

        results = trainer.train_baseline_models(
            X_train,
            y_train,
            X_test,
            y_test
        )

        print("\nStep 4: Tune RF")

        results["RF Tuned"] = (
            trainer.optimize_rf(
                X_train,
                y_train,
                X_test,
                y_test
            )
        )

        print("\nStep 5: Tune LGBM")

        results["LGBM Tuned"] = (
            trainer.optimize_lgbm(
                X_train,
                y_train,
                X_test,
                y_test
            )
        )
        for name, result in results.items():

            mlflow.log_metric(
                f"{name}_accuracy",
                result["accuracy"]
            )

            mlflow.log_metric(
                f"{name}_macro_f1",
                result["macro_f1"]
            )
        print("\n=== MODEL COMPARISON ===")

        for name, result in results.items():

            print(
                f"{name}"
                f" | Train Acc: {result['train_accuracy']:.4f}"
                f" | Test Acc: {result['accuracy']:.4f}"
                f" | Macro F1: {result['macro_f1']:.4f}"
            )

        best_model, best_model_name, results = (
            trainer.select_best_model(
                results
            )
        )

        print("\nStep 6: Evaluate Best Model")

        evaluator = ModelEvaluator()

        metrics = evaluator.evaluate(
            best_model,
            X_train,
            y_train,
            X_test,
            y_test
        )

        mlflow.log_param(
            "best_model",
            best_model_name
        )

        mlflow.log_metric(
            "train_accuracy",
            metrics["train_acc"]
        )

        mlflow.log_metric(
            "test_accuracy",
            metrics["test_acc"]
        )

        mlflow.log_metric(
            "macro_f1",
            metrics["macro_f1"]
        )

        mlflow.sklearn.log_model(
            best_model,
            "credit_score_model"
        )

    print("\nPipeline Finished")

    print(
        f"\nBest Model : {best_model_name}"
    )

    print(
        f"Test Accuracy : {metrics['test_acc']:.4f}"
    )

    print(
        f"Macro F1 : {metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()