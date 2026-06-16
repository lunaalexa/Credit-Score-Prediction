from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class ModelEvaluator:

    def evaluate(
        self,
        model,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(
            y_train,
            train_pred
        )

        test_acc = accuracy_score(
            y_test,
            test_pred
        )

        test_f1 = f1_score(
            y_test,
            test_pred,
            average='macro'
        )

        print(
            f"Train Accuracy: "
            f"{train_acc:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{test_acc:.4f}"
        )

        print(
            f"Macro F1: "
            f"{test_f1:.4f}"
        )

        print("\nClassification Report")

        print(
            classification_report(
                y_test,
                test_pred
            )
        )

        cm = confusion_matrix(
            y_test,
            test_pred
        )

        print("\nConfusion Matrix")
        print(cm)

        return {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "macro_f1": test_f1
        }