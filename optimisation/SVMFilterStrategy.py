from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class SVMFilterStrategy:
    """
    A unified strategy for training and applying SVM-based filtering
    with flexible exploration/exploitation modes.
    """

    def __init__(self, mode='median', percentile=90, threshold=0.2,
                 knn_k=3, kernel=None, filter_training_mode='exploration',
                 filter_strategy='good'):
        """
        Parameters:
        - mode: 'median', 'gp', or 'knn'
        - percentile: for 'gp' or 'knn', top percentile to label as 'good'
        - threshold: decision boundary margin for filtering
        - knn_k: number of neighbors for KNN
        - kernel: optional GP kernel (used in 'gp' mode)
        - filter_training_mode: 'exploration', 'exploitation', or 'hybrid'
        - filter_strategy: 'boundary' (uncertain points) or 'good' (positively classified)
        """
        self.mode = mode
        self.percentile = percentile
        self.threshold = threshold
        self.knn_k = knn_k
        self.kernel = kernel or RBF(length_scale=0.1, length_scale_bounds='fixed')
        self.filter_training_mode = filter_training_mode
        self.filter_strategy = filter_strategy
        self.svm_clf = None
        self.scaler = None

    def fit(self, X, y, X_grid=None):
        """
        Train the SVM classifier based on the selected labeling strategy
        and training mode.
        """
        self.X_train = X
        self.y_train = y
        
        if self.mode == 'median':
            labels = (y >= np.median(y)).astype(int)
            train_X, train_labels = X, labels

        elif self.mode in ['gp', 'knn']:
            if X_grid is None:
                raise ValueError("X_grid must be provided for 'gp' or 'knn' mode.")

            if self.mode == 'gp':
                model = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10)
                model.fit(X, y)
                mean, _ = model.predict(X_grid, return_std=True)
                cutoff = np.percentile(mean, self.percentile)
                grid_labels = (mean >= cutoff).astype(int)

            elif self.mode == 'knn':
                model = KNeighborsRegressor(n_neighbors=self.knn_k)
                model.fit(X, y)
                y_grid = model.predict(X_grid)
                cutoff = np.percentile(y_grid, self.percentile)
                grid_labels = (y_grid >= cutoff).astype(int)

            if self.filter_training_mode == 'exploration':
                train_X, train_labels = X_grid, grid_labels
            elif self.filter_training_mode == 'exploitation':
                y_obs_pred = model.predict(X)
                cutoff_obs = np.percentile(y_obs_pred, self.percentile)
                obs_labels = (y_obs_pred >= cutoff_obs).astype(int)
                train_X, train_labels = X, obs_labels
            elif self.filter_training_mode == 'hybrid':
                y_obs_pred = model.predict(X)
                cutoff_obs = np.percentile(y_obs_pred, self.percentile)
                obs_labels = (y_obs_pred >= cutoff_obs).astype(int)
                train_X = np.vstack([X, X_grid])
                train_labels = np.concatenate([obs_labels, grid_labels])
            else:
                raise ValueError(f"Unsupported filter_training_mode: {self.filter_training_mode}")

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(train_X)
        self.svm_clf = SVC(kernel='rbf', C=1.0)
        self.svm_clf.fit(X_scaled, train_labels)

    def filter(self, candidates_raw, plotter=None):
        """
        Apply the trained SVM to filter candidate points.

        Returns:
        - filtered_candidates
        """
        if self.svm_clf is None or self.scaler is None:
            raise RuntimeError("SVMFilterStrategy must be fit before calling filter().")

        # Scale candidates
        candidates_n = self.scaler.transform(candidates_raw)

        # -----------------------------
        # Boundary strategy (adaptive auto‑tuned)
        # -----------------------------
        if self.filter_strategy == 'boundary':
            decision_values = self.svm_clf.decision_function(candidates_n)
            abs_vals = np.abs(decision_values)
            # Spread of decision values tells us how sharp or noisy the boundary is
            spread = abs_vals.max() - abs_vals.min()
            # Adaptive percentile selection
            if spread < 2:
                pct = 10      # clean, sharp boundary → thin band
            elif spread < 10:
                pct = 15      # moderate spread → medium band
            else:
                pct = 20      # noisy or wide spread → thicker band
            # Auto‑tune threshold based on chosen percentile
            self.threshold = np.percentile(abs_vals, pct)
            print(f"Auto‑tuned percentile: {pct}%, threshold: {self.threshold:.4f}")
            mask = abs_vals < self.threshold

        # -----------------------------
        # Good strategy (class == 1)
        # -----------------------------
        elif self.filter_strategy == 'good':
            mask = self.svm_clf.predict(candidates_n) == 1

        else:
            raise ValueError(f"Unsupported filter_strategy: {self.filter_strategy}")

        # Apply mask
        filtered = candidates_raw[mask]
        print(f"Filtered {len(filtered)} candidates using filter_strategy='{self.filter_strategy}'")

        # Optional plotting
        if plotter is not None:
            plotter.plot_svm_with_candidates(
                self.svm_clf,
                self.scaler,
                self.X_train,
                self.y_train,
                candidates_raw,
                filtered
            )

        return filtered