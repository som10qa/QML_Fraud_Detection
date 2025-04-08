# #!/usr/bin/env python
# import sys
# import os

# # Append the project root to sys.path so that "src" can be found
# project_root = os.path.join(os.path.dirname(__file__), '..')
# sys.path.append(project_root)

# from src import data_preprocessing, classical_models, quantum_models, evaluation

# def main():
#     # Load and preprocess the dataset
#     X_train, X_test, y_train, y_test = data_preprocessing.load_and_preprocess_data("data/creditcard.csv")
    
#     # Train and evaluate the classical baseline model (Random Forest)
#     rf_model, rf_pred = classical_models.train_random_forest(X_train, X_test, y_train, y_test)
#     print("\n--- Random Forest Results ---")
#     evaluation.print_classification_report(y_test, rf_pred)
    
#     # # Train and evaluate the QSVM model
#     # print("\n--- Training QSVM ---")
#     # qsvm_pred = quantum_models.train_qsvm(X_train, X_test, y_train, y_test)
#     # print("\n--- QSVM Results ---")
#     # evaluation.print_classification_report(y_test, qsvm_pred)
    
#     # Train and evaluate the VQC model
#     print("\n--- Training VQC ---")
#     vqc_result = quantum_models.train_vqc(X_train, X_test, y_train, y_test)
#     print("\n--- VQC Accuracy ---")
#     print(vqc_result)
    
#     # Train and evaluate the QNN model
#     print("\n--- Training QNN ---")
#     qnn_results = quantum_models.train_qnn(X_train, X_test, y_train, y_test)
#     print("\n--- QNN Results ---")
#     evaluation.print_classification_report(y_test, qnn_results['predictions'])
    
#     # Plot ROC curves for comparison (example: Random Forest vs QNN)
#     evaluation.plot_roc_curves(y_test, X_test, rf_model, qnn_results['raw_outputs'])

    
# if __name__ == "__main__":
#     main()


#!/usr/bin/env python
import sys
import os

# Append project root so that "src" can be imported
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root)

from src import data_preprocessing, classical_models, quantum_models_angleEmbedding_standard, evaluation
#from src.quantum_models_angleEmbedding_standard import plot_combined_roc_curves

def main():
    # 1) Load & preprocess
    X_train, X_test, y_train, y_test = data_preprocessing.load_and_preprocess_data("data/creditcard.csv")
    
    # 2) Classical baseline
    rf_model, rf_pred = classical_models.train_random_forest(X_train, X_test, y_train, y_test)
    print("\n--- Random Forest Results ---")
    evaluation.print_classification_report(y_test, rf_pred)
    
    # 3) QSVM
    # qsvm_res = quantum_models_angleEmbedding_standard.train_qsvm(X_train, X_test, y_train, y_test)
    # print("\n--- QSVM Results ---")
    # evaluation.print_classification_report(y_test, qsvm_res["predictions"])
    
    # 4) VQC
    vqc_res = quantum_models_angleEmbedding_standard.train_vqc(X_train, X_test, y_train, y_test)
    print("\n--- VQC Results ---")
    evaluation.print_classification_report(y_test, vqc_res["predictions"])
    
    # 5) QNN
    qnn_res = quantum_models_angleEmbedding_standard.train_qnn(X_train, X_test, y_train, y_test)
    print("\n--- QNN Results ---")
    evaluation.print_classification_report(y_test, qnn_res["predictions"])

    # 6) EQNN
    eqnn_res = quantum_models_angleEmbedding_standard.train_eqnn(X_train, X_test, y_train, y_test)
    print("\n--- EQNN Results ---")
    evaluation.print_classification_report(y_test, eqnn_res["predictions"])

    # 7) SQNN
    sqnn_res = quantum_models_angleEmbedding_standard.train_sqnn(X_train, X_test, y_train, y_test)
    print("\n--- SQNN Results ---")
    evaluation.print_classification_report(y_test, sqnn_res["predictions"])
    
    # 8) Combined ROC
    print("\n--- Combined ROC Curves ---")
    evaluation.plot_combined_roc_curves(
        y_test,
        rf_model.predict_proba(X_test)[:, 1],
        #qsvm_res["scores"],
        vqc_res["scores"],
        qnn_res["scores"],
        eqnn_res["scores"],
        sqnn_res["scores"]
    )

if __name__ == "__main__":
    main()