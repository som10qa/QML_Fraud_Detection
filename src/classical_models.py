# src/classical_models.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from src import config

def train_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=config.RF_N_ESTIMATORS, random_state=config.RANDOM_SEED)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    return rf, predictions

def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))
