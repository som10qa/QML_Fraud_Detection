import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
import pennylane as qml
from pennylane.qnn import TorchLayer
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from src import config
import autoray
import pennylane.math as pml_math

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = config.RANDOM_SEED
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── Device Setup ───────────────────────────────────────────────────────────
# For local Apple hardware, we use MPS if available; change to "cuda" on HPC.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", DEVICE)

# ─── Logging Setup ──────────────────────────────────────────────────────────
LOG_FILE = "training_log.txt"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)
def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# ─── MPS Float32 Patch ───────────────────────────────────────────────────────
def patched_asarray(x, like=None, **kwargs):
    if isinstance(x, torch.Tensor):
        return x.to(torch.device(DEVICE)).clone().detach().requires_grad_(True)
    return torch.tensor(x, dtype=torch.float32, device=torch.device(DEVICE), requires_grad=True)
autoray.numpy.asarray = patched_asarray
pml_math._asarray = patched_asarray

def safe_torch_as_tensor(data, **kwargs):
    if isinstance(data, torch.Tensor):
        return data.to(torch.device(DEVICE)).clone().detach().requires_grad_(True)
    return torch.tensor(data, dtype=torch.float32, device=torch.device(DEVICE), requires_grad=True)
torch.as_tensor = safe_torch_as_tensor

# =============================================================================
# Model Training Functions
# =============================================================================

def train_qsvm(X_train, X_test, y_train, y_test):
    """Quantum Support Vector Classifier using AmplitudeEmbedding."""
    log("\n--- Training QSVM ---")
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Normalize using L2 norm.
    X_train_res = X_train_res / np.linalg.norm(X_train_res, axis=1, keepdims=True)
    X_test_norm = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    dev = qml.device("default.qubit", wires=config.PCA_COMPONENTS)

    @qml.qnode(dev)
    def feature_map(x, y=None):
        # Convert tensors to numpy arrays if needed.
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if y is not None and isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        # Embed data via AmplitudeEmbedding.
        qml.templates.AmplitudeEmbedding(x, wires=range(config.PCA_COMPONENTS), pad_with=0.0, normalize=True)
        if y is not None:
            qml.adjoint(qml.templates.AmplitudeEmbedding)(y, wires=range(config.PCA_COMPONENTS), pad_with=0.0, normalize=True)
        return qml.probs(wires=0)

    def gram_matrix(A, B=None):
        B = A if B is None else B
        K = np.zeros((len(A), len(B)))
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                K[i, j] = feature_map(a, b)[0]
        return K

    K_train = gram_matrix(X_train_res)
    K_test = gram_matrix(X_test_norm, X_train_res)

    from sklearn.svm import SVC
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train_res)
    predictions = clf.predict(K_test)
    return predictions

def train_vqc(X_train, X_test, y_train, y_test, batch_size=64):
    """Variational Quantum Classifier (VQC) with a classical head for improved performance."""
    log("\n--- Training VQC ---")
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # L2 normalization
    X_train_res = X_train_res / np.linalg.norm(X_train_res, axis=1, keepdims=True)
    X_test_norm = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    dev = qml.device("default.qubit", wires=config.PCA_COMPONENTS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        AngleEmbedding(inputs, wires=range(config.PCA_COMPONENTS))
        StronglyEntanglingLayers(weights, wires=range(config.PCA_COMPONENTS))
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (config.VQC_LAYERS, config.PCA_COMPONENTS, 3)}
    qlayer = TorchLayer(circuit, weight_shapes)

    class VQCClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer
            self.fc1 = nn.Linear(1, 32)
            self.act1 = nn.ELU()
            self.fc2 = nn.Linear(32, 16)
            self.act2 = nn.ELU()
            self.fc3 = nn.Linear(16, 1)
        def forward(self, x):
            out = self.qlayer(x)
            if out.ndim > 1:
                out = out[:, 0]
            x_in = out + 1  # Shift from [-1,1] to [0,2]
            x_in = self.act1(self.fc1(x_in.unsqueeze(1)))
            x_in = self.act2(self.fc2(x_in))
            logits = self.fc3(x_in)
            return logits

    model = VQCClassifier().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=config.QNN_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    pos_weight = torch.tensor(float((y_train_res == 0).sum() / (y_train_res == 1).sum()),
                              dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr = torch.tensor(X_train_res, dtype=torch.float32, device=DEVICE)
    y_tr_arr = y_train_res.values if hasattr(y_train_res, "values") else np.array(y_train_res)
    y_tr = torch.tensor(y_tr_arr.astype(np.float32).reshape(-1, 1), device=DEVICE)
    X_te = torch.tensor(X_test_norm, dtype=torch.float32, device=DEVICE)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.QNN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        log(f"VQC Epoch {epoch+1}/{config.QNN_EPOCHS} - Loss: {loss.item():.4f}")
        print(f"VQC Epoch {epoch+1}/{config.QNN_EPOCHS} - Loss: {loss.item():.4f}")
        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                log(f"Early stopping triggered at epoch {epoch+1}")
                print("Early stopping triggered.")
                break

    with torch.no_grad():
        raw = torch.sigmoid(model(X_te)).cpu().numpy().flatten()
    preds = (raw >= 0.5).astype(int)
    return {"predictions": preds, "scores": raw}

def train_qnn(X_train, X_test, y_train, y_test, batch_size=64):
    """Quantum Neural Network (QNN) with a classical head including dropout for improved performance."""
    log("\n--- Training QNN ---")
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    X_train_res = X_train_res / np.linalg.norm(X_train_res, axis=1, keepdims=True)
    X_test_norm = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    dev = qml.device("default.qubit", wires=config.PCA_COMPONENTS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        AngleEmbedding(inputs, wires=range(config.PCA_COMPONENTS))
        StronglyEntanglingLayers(weights, wires=range(config.PCA_COMPONENTS))
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (config.VQC_LAYERS, config.PCA_COMPONENTS, 3)}
    qlayer = TorchLayer(circuit, weight_shapes)

    class QNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer
            self.fc1 = nn.Linear(1, 32)
            self.act1 = nn.ELU()
            self.drop1 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(32, 16)
            self.act2 = nn.ELU()
            self.drop2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(16, 1)
        def forward(self, x):
            out = self.qlayer(x)
            if out.ndim > 1:
                out = out[:, 0]
            x_in = out + 1
            x_in = self.fc1(x_in.unsqueeze(1))
            x_in = self.drop1(self.act1(x_in))
            x_in = self.fc2(x_in)
            x_in = self.drop2(self.act2(x_in))
            logits = self.fc3(x_in)
            return logits

    model = QNNClassifier().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=config.QNN_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    pos_weight = torch.tensor(float((y_train_res == 0).sum() / (y_train_res == 1).sum()),
                              dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr = torch.tensor(X_train_res, dtype=torch.float32, device=DEVICE)
    y_tr_arr = y_train_res.values if hasattr(y_train_res, "values") else np.array(y_train_res)
    y_tr = torch.tensor(y_tr_arr.astype(np.float32).reshape(-1, 1), device=DEVICE)
    X_te = torch.tensor(X_test_norm, dtype=torch.float32, device=DEVICE)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.QNN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        log(f"QNN Epoch {epoch+1}/{config.QNN_EPOCHS} - Loss: {loss.item():.4f}")
        print(f"QNN Epoch {epoch+1}/{config.QNN_EPOCHS} - Loss: {loss.item():.4f}")
        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                log(f"Early stopping triggered at epoch {epoch+1}")
                print("Early stopping triggered.")
                break

    model.eval()
    with torch.no_grad():
        raw_outputs = torch.sigmoid(model(X_te)).cpu().numpy().flatten()
        predictions = (raw_outputs >= 0.5).astype(int)
        auc_val = roc_auc_score(y_test, raw_outputs)
        log(f"\nQNN ROC AUC: {auc_val:.4f}")
        print(f"\nQNN ROC AUC: {auc_val:.4f}")

    return {
        "predictions": predictions,
        "raw_outputs": raw_outputs,
        "auc": auc_val
    }

def train_eqnn(X_train, X_test, y_train, y_test, batch_size=64):
    """
    Train an Estimator Quantum Neural Network (EQNN) model.
    The EQNN model uses a quantum feature map to extract expectation values from each qubit,
    then processes these quantum features with a classical neural network.
    """
    log("\n--- Training EQNN ---")
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    X_train_res = X_train_res / np.linalg.norm(X_train_res, axis=1, keepdims=True)
    X_test_norm  = X_test  / np.linalg.norm(X_test, axis=1, keepdims=True)
    
    dev = qml.device("default.qubit", wires=config.PCA_COMPONENTS)
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        AngleEmbedding(inputs, wires=range(config.PCA_COMPONENTS))
        StronglyEntanglingLayers(weights, wires=range(config.PCA_COMPONENTS))
        # Return one expectation per wire
        return [qml.expval(qml.PauliZ(i)) for i in range(config.PCA_COMPONENTS)]
    
    weight_shapes = {"weights": (config.VQC_LAYERS, config.PCA_COMPONENTS, 3)}
    qlayer = TorchLayer(circuit, weight_shapes)
    
    class EQNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer
            self.fc1 = nn.Linear(config.PCA_COMPONENTS, 32)
            self.act1 = nn.ELU()
            self.fc2 = nn.Linear(32, 16)
            self.act2 = nn.ELU()
            self.fc3 = nn.Linear(16, 1)
        def forward(self, x):
            quantum_features = self.qlayer(x)  # shape: (batch, PCA_COMPONENTS)
            out = self.fc1(quantum_features)
            out = self.act1(out)
            out = self.fc2(out)
            out = self.act2(out)
            logits = self.fc3(out)
            return logits
    
    model = EQNNClassifier().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=config.QNN_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config.LR_FACTOR, patience=config.LR_PATIENCE, verbose=True)
    pos_weight = torch.tensor(float((y_train_res == 0).sum() / (y_train_res == 1).sum()),
                              dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    X_tr = torch.tensor(X_train_res, dtype=torch.float32, device=DEVICE)
    y_tr_arr = y_train_res.values if hasattr(y_train_res, "values") else np.array(y_train_res)
    y_tr = torch.tensor(y_tr_arr.astype(np.float32).reshape(-1,1), device=DEVICE)
    X_te = torch.tensor(X_test_norm, dtype=torch.float32, device=DEVICE)
    
    best_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(config.QNN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        log(f"EQNN Epoch {epoch+1}/{config.QNN_EPOCHS} - Loss: {loss.item():.4f}")
        print(f"EQNN Epoch {epoch+1}/{config.QNN_EPOCHS} - Loss: {loss.item():.4f}")
        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                log(f"Early stopping triggered at epoch {epoch+1}")
                print("Early stopping triggered.")
                break

    model.eval()
    with torch.no_grad():
        raw_outputs = torch.sigmoid(model(X_te)).cpu().numpy().flatten()
        predictions = (raw_outputs >= 0.5).astype(int)
        auc_val = roc_auc_score(y_test, raw_outputs)
        log(f"\nEQNN ROC AUC: {auc_val:.4f}")
        print(f"\nEQNN ROC AUC: {auc_val:.4f}")
    
    return {"predictions": predictions, "raw_outputs": raw_outputs, "auc": auc_val}

def train_sqnn(X_train, X_test, y_train, y_test, batch_size=64, shots=1024):
    """
    Train a Sampler Quantum Neural Network (SQNN) model.
    The SQNN uses qml.sample to obtain raw samples, averages them to estimate expectation values,
    and then feeds these features into a classical network.
    """
    log("\n--- Training SQNN ---")
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    X_train_res = X_train_res / np.linalg.norm(X_train_res, axis=1, keepdims=True)
    X_test_norm  = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
    
    dev = qml.device("default.qubit", wires=config.PCA_COMPONENTS, shots=shots)
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        AngleEmbedding(inputs, wires=range(config.PCA_COMPONENTS))
        StronglyEntanglingLayers(weights, wires=range(config.PCA_COMPONENTS))
        return [qml.sample(qml.PauliZ(i)) for i in range(config.PCA_COMPONENTS)]
    
    weight_shapes = {"weights": (config.VQC_LAYERS, config.PCA_COMPONENTS, 3)}
    qlayer = TorchLayer(circuit, weight_shapes)
    
    class SQNNFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer
        def forward(self, x):
            samples = self.qlayer(x)
            if isinstance(samples, list):
                samples = torch.stack(samples, dim=1)  # shape: (batch, PCA_COMPONENTS, shots)
            quantum_features = torch.mean(samples, dim=2)  # average over shots
            return quantum_features
    
    class SQNNClassifier(nn.Module):
        def __init__(self, feature_extractor):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.fc1 = nn.Linear(config.PCA_COMPONENTS, 32)
            self.act1 = nn.ELU()
            self.fc2 = nn.Linear(32, 16)
            self.act2 = nn.ELU()
            self.fc3 = nn.Linear(16, 1)
        def forward(self, x):
            features = self.feature_extractor(x)
            out = self.fc1(features)
            out = self.act1(out)
            out = self.fc2(out)
            out = self.act2(out)
            logits = self.fc3(out)
            return logits

    feature_extractor = SQNNFeatureExtractor().to(DEVICE)
    model = SQNNClassifier(feature_extractor).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=config.QNN_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=config.LR_FACTOR, patience=config.LR_PATIENCE, verbose=True)
    pos_weight = torch.tensor(float((y_train_res == 0).sum() / (y_train_res == 1).sum()),
                              dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    X_tr = torch.tensor(X_train_res, dtype=torch.float32, device=DEVICE)
    y_tr_arr = y_train_res.values if hasattr(y_train_res, "values") else np.array(y_train_res)
    y_tr = torch.tensor(y_tr_arr.astype(np.float32).reshape(-1, 1), device=DEVICE)
    X_te = torch.tensor(X_test_norm, dtype=torch.float32, device=DEVICE)
    
    best_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(config.QNN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        log(f"SQNN Epoch {epoch+1}/{config.QNN_EPOCHS} - Loss: {loss.item():.4f}")
        print(f"SQNN Epoch {epoch+1}/{config.QNN_EPOCHS} - Loss: {loss.item():.4f}")
        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                log(f"Early stopping triggered at epoch {epoch+1}")
                print("Early stopping triggered.")
                break

    model.eval()
    with torch.no_grad():
        raw_outputs = torch.sigmoid(model(X_te)).cpu().numpy().flatten()
        predictions = (raw_outputs >= 0.5).astype(int)
        auc_value = roc_auc_score(y_test, raw_outputs)
        log(f"\nSQNN ROC AUC: {auc_value:.4f}")
        print(f"\nSQNN ROC AUC: {auc_value:.4f}")
    
    return {"predictions": predictions, "raw_outputs": raw_outputs, "auc": auc_value}
