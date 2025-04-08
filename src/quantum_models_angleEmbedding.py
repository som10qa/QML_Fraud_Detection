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

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = config.SEED
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── Device Setup ───────────────────────────────────────────────────────────
# For local machine using Apple hardware, you can use MPS if available:
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

# ─── Patch for Apple M1/M2/M3 (MPS compatibility) ──────────────────────────
import autoray
import pennylane.math as pml_math
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

# ─────────────────────────────────────────────────────────────────────────────
# Updated VQC Function with additional classical layers for improved performance.
def train_vqc(X_train, X_test, y_train, y_test, batch_size=64):
    """
    Train a Variational Quantum Classifier (VQC) model using a quantum circuit 
    wrapped with a classical postprocessing head.
    """
    log("\n--- Training VQC ---")
    # Apply SMOTE on the training set
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Normalize features using L2 normalization
    X_train_res = X_train_res / np.linalg.norm(X_train_res, axis=1, keepdims=True)
    X_test_norm = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    import pennylane as qml
    from pennylane.qnn import TorchLayer
    from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers

    # Configure quantum device
    dev = qml.device("default.qubit", wires=config.PCA_COMPONENTS)
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        AngleEmbedding(inputs, wires=range(config.PCA_COMPONENTS))
        StronglyEntanglingLayers(weights, wires=range(config.PCA_COMPONENTS))
        return qml.expval(qml.PauliZ(0))
    
    weight_shapes = {"weights": (config.VQC_LAYERS, config.PCA_COMPONENTS, 3)}
    qlayer = TorchLayer(circuit, weight_shapes)
    
    # VQC model with an added classical head for improved decision boundaries.
    class VQCClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer
            self.fc1 = nn.Linear(1, 32)
            self.act1 = nn.ELU()
            self.fc2 = nn.Linear(32, 16)
            self.act2 = nn.ELU()
            self.fc3 = nn.Linear(16, 1)  # final output (logits)
        def forward(self, x):
            out = self.qlayer(x)
            # If the quantum layer returns multiple outputs, take the first element per sample.
            if out.ndim > 1:
                out = out[:, 0]
            # Shift quantum output (for example, mapping [-1,1] to [0,2])
            x_in = out + 1  
            # Pass through classical layers
            x_in = self.act1(self.fc1(x_in.unsqueeze(1)))  # unsqueeze to shape (batch, 1)
            x_in = self.act2(self.fc2(x_in))
            logits = self.fc3(x_in)  # raw logits output
            return logits

    model = VQCClassifier().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=config.QNN_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    # Using BCEWithLogitsLoss incorporates the sigmoid internally
    pos_weight = torch.tensor(float((y_train_res == 0).sum() / (y_train_res == 1).sum()),
                              dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Convert training and test data to tensors.
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
        # Apply sigmoid on logits to obtain probabilities
        raw = torch.sigmoid(model(X_te)).cpu().numpy().flatten()
    preds = (raw >= 0.5).astype(int)
    return {"predictions": preds, "scores": raw}

# ─────────────────────────────────────────────────────────────────────────────
# Updated QNN Function with additional classical layers.
def train_qnn(X_train, X_test, y_train, y_test, batch_size=64):
    log("\n--- Training QNN ---")
    # Apply SMOTE on training set to balance classes
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Normalize features using L2 normalization
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
    
    # QNN model with an enhanced classical head for improved learning.
    class QNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer
            self.fc1 = nn.Linear(1, 32)
            self.act1 = nn.ELU()
            self.fc2 = nn.Linear(32, 16)
            self.act2 = nn.ELU()
            self.fc3 = nn.Linear(16, 1)  # output raw logits
        def forward(self, x):
            out = self.qlayer(x)
            if out.ndim > 1:
                out = out[:, 0]
            x_in = out + 1  # shift output from quantum layer
            x_in = self.act1(self.fc1(x_in.unsqueeze(1)))  # unsqueeze to add dimension
            x_in = self.act2(self.fc2(x_in))
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
        auc_value = roc_auc_score(y_test, raw_outputs)
        log(f"\nQNN ROC AUC: {auc_value:.4f}")
        print(f"\nQNN ROC AUC: {auc_value:.4f}")

    return {
        "predictions": predictions,
        "raw_outputs": raw_outputs,
        "auc": auc_value
    }