import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler


class SurrogateCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        """
        A simple 1D CNN surrogate model for tabular inputs.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dim : int
            Size of hidden fully connected layer.
        """
        super(SurrogateCNN, self).__init__()
        # Convolutions over feature dimension (treat features as sequence)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, input_dim)
        x = x.unsqueeze(1)  # add channel dimension → (batch, 1, input_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler


class SurrogateCNNTrainer:
    def __init__(self, input_dim, lr=1e-3, epochs=200, patience=20, min_delta=1e-4):
        """
        Trainer for SurrogateCNN with early stopping.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        lr : float
            Learning rate.
        epochs : int
            Maximum number of epochs.
        patience : int
            How many epochs to wait for improvement before stopping.
        min_delta : float
            Minimum improvement in validation loss to reset patience.
        """
        self.model = SurrogateCNN(input_dim)
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X, y, log_weights=False, val_split=0.2):
        """
        Train the surrogate CNN with early stopping and optional weight logging.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        log_weights : bool
            If True, log mean/std of weights per epoch.
        val_split : float
            Fraction of data to use for validation.

        Returns
        -------
        history : dict
            {
            "train_losses": [...],
            "val_losses": [...],
            "weights": [...]  # only if log_weights=True
            }
        """
        # Scale data
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Train/val split
        n_val = int(len(X_scaled) * val_split)
        X_train, X_val = X_scaled[:-n_val], X_scaled[-n_val:]
        y_train, y_val = y_scaled[:-n_val], y_scaled[-n_val:]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        train_losses, val_losses, weights_log = [], [], []
        best_val_loss = np.inf
        patience_counter = 0

        for epoch in range(self.epochs):
            # ---- Training ----
            self.model.train()
            optimizer.zero_grad()
            output = self.model(X_train_tensor)
            loss = loss_fn(output, y_train_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # ---- Validation ----
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val_tensor)
                val_loss = loss_fn(val_output, y_val_tensor).item()
            val_losses.append(val_loss)

            # ---- Log weights ----
            if log_weights:
                epoch_stats = {}
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        epoch_stats[name] = {
                            "mean": param.data.mean().item(),
                            "std": param.data.std().item()
                        }
                weights_log.append(epoch_stats)

            # ---- Early stopping ----
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_surrogate_cnn.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}, best val_loss={best_val_loss:.6f}")
                    self.model.load_state_dict(torch.load("best_surrogate_cnn.pt"))
                    break

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "weights": weights_log if log_weights else None
        }

    def predict(self, X):
        """
        Predict outputs for new inputs using trained surrogate CNN.
        """
        X_scaled = self.scaler_x.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor).numpy()
        return self.scaler_y.inverse_transform(y_pred).ravel()