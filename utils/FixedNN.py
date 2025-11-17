import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.stats import norm
from scipy.spatial import ConvexHull
from utils.SVMFilterStrategy import SVMFilterStrategy

# -------------------- Fixed Surrogate --------------------
class FixedNN(nn.Module):
    def __init__(self, input_dim):
        super(FixedNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)

class FixedPredictor:
    def __init__(self, input_dim):
        self.model = FixedNN(input_dim)
        self.scaler_x = StandardScaler()

    def fit(self, X):  # Only fit scaler
        self.scaler_x.fit(X)

    def predict(self, X):
        X_scaled = self.scaler_x.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_tensor).numpy().ravel()