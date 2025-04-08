# QML Fraud Detection

This project demonstrates a quantum machine learning (QML) approach for credit card fraud detection. It combines classical data preprocessing, baseline machine learning models, and multiple QML models (QSVM, VQC, and QNN) using Qiskit and PyTorch.

## Folder Structure

- **data/**: Contains the dataset (`creditcard.csv`).
- **docs/**: Final report and presentation.
- **notebooks/**: Jupyter Notebook for interactive exploration.
- **scripts/**: Main script (`run_project.py`) to run the pipeline.
- **src/**: Python modules for configuration, preprocessing, models, and evaluation.
- **requirements.txt**: Python dependencies.
- **README.md**: Project overview and instructions.
- **.gitignore**: Files to be ignored by Git.

## Setup Instructions

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv qml_env
   source qml_env/bin/activate
   pip install -r requirements.txt
