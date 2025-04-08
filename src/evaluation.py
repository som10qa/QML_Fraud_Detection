# src/evaluation.py

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns

def print_classification_report(y_true, y_pred):
    print("Classification Report:\n")
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_roc_curves(y_test, X_test, classical_model, vqc_raw_outputs, qnn_raw_outputs, eqnn_raw_outputs, sqnn_raw_outputs):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # Obtain probabilities from the classical model.
    y_prob_classical = classical_model.predict_proba(X_test)[:, 1]
    fpr_clf, tpr_clf, _ = roc_curve(y_test, y_prob_classical)
    auc_clf = auc(fpr_clf, tpr_clf)

    # Compute ROC curves and AUC for VQC.
    fpr_vqc, tpr_vqc, _ = roc_curve(y_test, vqc_raw_outputs)
    auc_vqc = auc(fpr_vqc, tpr_vqc)

    # Compute ROC curves and AUC for QNN.
    fpr_qnn, tpr_qnn, _ = roc_curve(y_test, qnn_raw_outputs)
    auc_qnn = auc(fpr_qnn, tpr_qnn)

    # Compute ROC curves and AUC for EQNN.
    fpr_eqnn, tpr_eqnn, _ = roc_curve(y_test, eqnn_raw_outputs)
    auc_eqnn = auc(fpr_eqnn, tpr_eqnn)

    # Compute ROC curves and AUC for SQNN.
    fpr_sqnn, tpr_sqnn, _ = roc_curve(y_test, sqnn_raw_outputs)
    auc_sqnn = auc(fpr_sqnn, tpr_sqnn)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_clf, tpr_clf, label=f"Random Forest (AUC = {auc_clf:.2f})")
    plt.plot(fpr_qnn, tpr_qnn, linestyle="--", label=f"VQC (AUC = {auc_vqc:.2f})")
    plt.plot(fpr_qnn, tpr_qnn, linestyle="--", label=f"QNN (AUC = {auc_qnn:.2f})")
    plt.plot(fpr_eqnn, tpr_eqnn, linestyle="-.", label=f"EQNN (AUC = {auc_eqnn:.2f})")
    plt.plot(fpr_sqnn, tpr_sqnn, linestyle=":", label=f"SQNN (AUC = {auc_sqnn:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()