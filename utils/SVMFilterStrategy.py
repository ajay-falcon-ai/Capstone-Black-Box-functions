from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class SVMFilterStrategy:
    """
    A unified strategy for training and applying SVM-based filtering
    using different labeling modes: 'median', 'gp', or 'knn'.
    """

    def __init__(self, mode='median', percentile=90, threshold=0.2, knn_k=3, kernel=None):
        """
        Parameters:
        - mode: 'median', 'gp', or 'knn'
        - percentile: for 'gp' or 'knn', top percentile to label as 'good'
        - threshold: decision boundary margin for filtering
        - knn_k: number of neighbors for KNN
        - kernel: optional GP kernel (used in 'gp' mode)
        """
        self.mode = mode
        self.percentile = percentile
        self.threshold = threshold
        self.knn_k = knn_k
        self.kernel = kernel or RBF(length_scale=0.1, length_scale_bounds='fixed')
        self.svm_clf = None
        self.scaler = None

    def fit(self, X, y, X_grid=None):
        """
        Train the SVM classifier based on the selected labeling strategy.

        Parameters:
        - X, y: known data
        - X_grid: required for 'gp' and 'knn' modes (grid to label)
        """
        if self.mode == 'median':
            labels = (y >= np.median(y)).astype(int)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.svm_clf = SVC(kernel='rbf', C=1.0)
            self.svm_clf.fit(X_scaled, labels)

        elif self.mode == 'gp':
            if X_grid is None:
                raise ValueError("X_grid must be provided for 'gp' mode.")
            gp = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10)
            gp.fit(X, y)
            mean, _ = gp.predict(X_grid, return_std=True)
            cutoff = np.percentile(mean, self.percentile)
            labels = (mean >= cutoff).astype(int)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_grid)
            self.svm_clf = SVC(kernel='rbf', C=1.0)
            self.svm_clf.fit(X_scaled, labels)

        elif self.mode == 'knn':
            if X_grid is None:
                raise ValueError("X_grid must be provided for 'knn' mode.")
            knn = KNeighborsRegressor(n_neighbors=self.knn_k)
            knn.fit(X, y)
            y_grid = knn.predict(X_grid)
            cutoff = np.percentile(y_grid, self.percentile)
            labels = (y_grid >= cutoff).astype(int)
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_grid)
            self.svm_clf = SVC(kernel='rbf', C=1.0)
            self.svm_clf.fit(X_scaled, labels)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def filter(self, candidates_raw):
        """
        Apply the trained SVM to filter candidate points.

        Parameters:
        - candidates_raw: grid or candidate points to filter

        Returns:
        - filtered_candidates: subset near decision boundary
        """
        if self.svm_clf is None or self.scaler is None:
            raise RuntimeError("SVMFilterStrategy must be fit before calling filter().")

        candidates_n = self.scaler.transform(candidates_raw)
        decision_values = self.svm_clf.decision_function(candidates_n)
        mask = np.abs(decision_values) < self.threshold
        filtered = candidates_raw[mask]

        num_total = len(candidates_raw)
        num_filtered = len(filtered)
        num_excluded = num_total - num_filtered

        print(f"Filtered {num_filtered} candidates near decision boundary (|margin| < {self.threshold})")
        print(f"Excluded {num_excluded} candidates from consideration")
        print(f"Final grid shape: {filtered.shape}")

        return filtered