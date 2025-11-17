import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.stats import norm
from scipy.spatial import ConvexHull
from utils.SVMFilterStrategy import SVMFilterStrategy

class SurrogateNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SurrogateNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)

class SurrogateTrainer:
    def __init__(self, input_dim):
        self.model = SurrogateNN(input_dim)        
    def __init__(self, input_dim, lr=1e-3, epochs=500):
        self.model = SurrogateNN(input_dim)
        self.lr = lr
        self.epochs = epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        losses = []

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = loss_fn(output, y_tensor)
            loss.backward()
            optimizer.step()

            # Track loss
            losses.append(loss.item())

            # Print progress every 50 epochs
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")

    def predict(self, X):
        X_scaled = self.scaler_x.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor).numpy()
        return self.scaler_y.inverse_transform(y_pred).ravel()
    
    def compute_input_influence(self, X, mode='global'):
        """
        Computes input feature influence on surrogate predictions via gradient sensitivity.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to evaluate gradients on.

        mode : str
            'local' → use only the first point in X
            'global' → average gradients across all points

        Returns
        -------
        influence : np.ndarray of shape (n_features,)
            Per-feature sensitivity scores (absolute gradient magnitudes).
        """
        X_scaled = self.scaler_x.transform(X)
        self.model.eval()

        if mode == 'local':
            x0 = torch.tensor(X_scaled[0], dtype=torch.float32, requires_grad=True)
            y_pred = self.model(x0)
            y_pred.backward()
            grad = x0.grad.detach().numpy()
            return np.abs(grad)

        elif mode == 'global':
            grads = []
            for x in X_scaled:
                xt = torch.tensor(x, dtype=torch.float32, requires_grad=True)
                y_pred = self.model(xt)
                y_pred.backward()
                grads.append(xt.grad.detach().numpy())
            grads = np.abs(np.stack(grads))
            return grads.mean(axis=0)

        else:
            raise ValueError("mode must be 'local' or 'global'")
