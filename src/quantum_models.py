import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from sklearn.svm import SVC

import pennylane as qml
from pennylane.templates import AmplitudeEmbedding, StronglyEntanglingLayers
from pennylane.qnn import TorchLayer
from src import config

# ─── Device detection ───────────────────────────────────────────────────────

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# ─── Patch Pennylane Tensor Behavior for MPS ────────────────────────────────

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

# ─── QSVM ────────────────────────────────────────────────────────────────────

def train_qsvm(X_train, X_test, y_train, y_test):
    dev = qml.device("default.qubit", wires=config.PCA_COMPONENTS)

    @qml.qnode(dev)
    def feature_map(x, y=None):
        AmplitudeEmbedding(x, wires=range(config.PCA_COMPONENTS), pad_with=0.0, normalize=True)
        if y is not None:
            qml.adjoint(AmplitudeEmbedding)(y, wires=range(config.PCA_COMPONENTS), pad_with=0.0, normalize=True)
        return qml.probs(wires=0)

    def gram_matrix(A, B=None):
        B = A if B is None else B
        K = np.zeros((len(A), len(B)))
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                K[i, j] = feature_map(a, b)[0]
        return K

    print("\n--- Training QSVM ---")
    K_train = gram_matrix(X_train)
    K_test = gram_matrix(X_test, X_train)

    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    return clf.predict(K_test)

# ─── VQC ─────────────────────────────────────────────────────────────────────

def train_vqc(X_train, X_test, y_train, y_test):
    dev = qml.device("default.qubit", wires=config.PCA_COMPONENTS)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        inputs = inputs.to(torch.device(DEVICE))
        AmplitudeEmbedding(inputs, wires=range(config.PCA_COMPONENTS), pad_with=0.0, normalize=True)
        StronglyEntanglingLayers(weights, wires=range(config.PCA_COMPONENTS))
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (config.VQC_LAYERS, config.PCA_COMPONENTS, 3)}
    qlayer = TorchLayer(circuit, weight_shapes)

    class VQCClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer

        def forward(self, x):
            return (self.qlayer(x) + 1) / 2

    model = VQCClassifier().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=config.QNN_LEARNING_RATE)
    loss_fn = nn.BCELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=DEVICE).requires_grad_()
    y_tr = torch.tensor(y_train.values.reshape(-1), dtype=torch.float32, device=DEVICE)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

    for epoch in range(config.QNN_EPOCHS):
        optimizer.zero_grad()
        preds = model(X_tr)
        loss = loss_fn(preds, y_tr)
        loss.backward()
        optimizer.step()
        print(f"VQC Epoch {epoch+1}/{config.QNN_EPOCHS} - Loss: {loss.item():.4f}")

    with torch.no_grad():
        raw = model(X_te).cpu().numpy().flatten()
    return (raw >= 0.5).astype(int)

# ─── QNN ─────────────────────────────────────────────────────────────────────

def train_qnn(X_train, X_test, y_train, y_test):
    dev = qml.device("default.qubit", wires=config.PCA_COMPONENTS)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        inputs = inputs.to(torch.device(DEVICE))
        AmplitudeEmbedding(inputs, wires=range(config.PCA_COMPONENTS), pad_with=0.0, normalize=True)
        StronglyEntanglingLayers(weights, wires=range(config.PCA_COMPONENTS))
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (config.VQC_LAYERS, config.PCA_COMPONENTS, 3)}
    qlayer = TorchLayer(circuit, weight_shapes)

    class QNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer
            self.head = nn.Linear(1, 1)

        def forward(self, x):
            z = self.qlayer(x).unsqueeze(1)
            return torch.sigmoid(self.head(z)).squeeze(1)

    model = QNNClassifier().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=config.QNN_LEARNING_RATE)
    loss_fn = nn.BCELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=DEVICE).requires_grad_()
    y_tr = torch.tensor(y_train.values, dtype=torch.float32, device=DEVICE)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

    for epoch in range(config.QNN_EPOCHS):
        optimizer.zero_grad()
        preds = model(X_tr)
        loss = loss_fn(preds, y_tr)
        loss.backward()
        optimizer.step()
        print(f"QNN Epoch {epoch+1}/{config.QNN_EPOCHS} - Loss: {loss.item():.4f}")

    with torch.no_grad():
        raw = model(X_te).cpu().numpy().flatten()
    return {
        "predictions": (raw >= 0.5).astype(int),
        "raw_outputs": raw
    }


# import numpy as np
# import torch
# from torch import nn
# from torch.optim import Adam
# from sklearn.svm import SVC

# import pennylane as qml
# from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
# from pennylane.qnn import TorchLayer
# from src import config
# def train_vqc(X_train, X_test, y_train, y_test):
#     """VQC with AngleEmbedding and StronglyEntanglingLayers."""
#     print("\n--- Training VQC ---")
#     dev = qml.device("default.qubit", wires=config.PCA_COMPONENTS)

#     @qml.qnode(dev, interface="torch", diff_method="backprop")
#     def circuit(inputs, weights):
#         AngleEmbedding(inputs, wires=range(config.PCA_COMPONENTS))
#         StronglyEntanglingLayers(weights, wires=range(config.PCA_COMPONENTS))
#         return qml.expval(qml.PauliZ(0))

#     weight_shapes = {"weights": (config.VQC_LAYERS, config.PCA_COMPONENTS, 3)}
#     qlayer = TorchLayer(circuit, weight_shapes)

#     class VQCClassifier(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.qlayer = qlayer

#         def forward(self, x):
#             return (self.qlayer(x) + 1) / 2  # ensure output in [0, 1]

#     model = VQCClassifier().to(DEVICE)
#     optimizer = Adam(model.parameters(), lr=config.QNN_LEARNING_RATE)
#     loss_fn = nn.BCELoss()

#     # Normalize input
#     X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
#     X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

#     # Ensure grad-tracking tensor
#     X_tr = torch.tensor(X_train, dtype=torch.float32, device=DEVICE, requires_grad=True)
#     y_tr = torch.tensor(y_train.values.reshape(-1), dtype=torch.float32, device=DEVICE)
#     X_te = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

#     for epoch in range(config.QNN_EPOCHS):
#         optimizer.zero_grad()

#         preds = model(X_tr)
#         if not preds.requires_grad:
#             print("⚠️ Warning: Predictions do not require grad!")
#         loss = loss_fn(preds, y_tr)

#         try:
#             loss.backward()
#         except RuntimeError as e:
#             print(f"🔥 Backward failed: {e}")
#             print(f"Requires grad? {preds.requires_grad}")
#             print(f"Preds type: {type(preds)}")
#             print(f"Preds sample: {preds[:5]}")
#             raise e

#         optimizer.step()
#         print(f"VQC Epoch {epoch+1}/{config.QNN_EPOCHS} - Loss: {loss.item():.4f}")

#     with torch.no_grad():
#         raw = model(X_te).cpu().numpy().flatten()
#     return (raw >= 0.5).astype(int)
