import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from src import config

def load_and_preprocess_data(filepath):
    # 1) Load all features (except label)
    df = pd.read_csv(filepath)
    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]

    # 2) Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3) PCA
    pca = PCA(n_components=config.PCA_COMPONENTS)
    X_pca = pca.fit_transform(X_scaled)

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y,
        test_size=0.3,
        random_state=config.RANDOM_SEED,
        stratify=y
    )

    # 5) SMOTE *only* on the training set
    smote = SMOTE(random_state=config.RANDOM_SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Debug
    print("After SMOTE, training set shapes:", X_train_res.shape, y_train_res.shape)
    print("Test set shapes:", X_test.shape, y_test.shape)

    return X_train_res, X_test, y_train_res, y_test