# bayes_opt_utils.py
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import pairwise_distances
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.spatial import ConvexHull
from optimisation.SVMFilterStrategy import SVMFilterStrategy
from plotting.PlotUtils import PlotUtils
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from scipy.stats import qmc   # for Sobol sequences


class BayesOptUtils:
    # --- Preprocessing & bounds ---
    def preprocess_y(self, y, apply_scaling=False, objective_mode="raw"):
        """
        Preprocess target values according to objective_mode and scaling.

        Parameters
        ----------
        y : np.ndarray
            Raw outputs from the black-box function.
        apply_scaling : bool
            Whether to standardize targets.
        objective_mode : str
            "raw"         → use y directly
            "zero_target" → use y**2 (distance to zero, always ≥ 0)
            "negated"     → use -y (flip for maximization)
        """
        # Transform first
        if objective_mode == "zero_target":
            y_proc = y**2
        elif objective_mode == "negated":
            y_proc = -y
        else:
            y_proc = y

        # Then scale if requested
        if apply_scaling:
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y_proc.reshape(-1, 1)).ravel()
            return y_scaled, scaler

        return y_proc, None

    def compute_bounds(self, X, margin=0.1, min_value=1e-2):
        bounds = []
        for i in range(X.shape[1]):
            # Clamp at min_value instead of 0.0
            low = max(min_value, np.min(X[:, i]) - margin)
            high = np.max(X[:, i]) + margin
            bounds.append((low, high))
        return bounds

    def filter_grid(self, X, y, X_grid, filter_mode='gp', percentile=90, threshold=0.2, training_mode='exploitation', filter_strategy='good'):
        strategy = SVMFilterStrategy(filter_mode, percentile, threshold, training_mode=training_mode, filter_strategy=filter_strategy)
        strategy.fit(X, y, X_grid=X_grid)
        return strategy.filter(X_grid)

    # --- Acquisition functions ---
    def compute_acquisition(self, method, mean, std, y, xi=0.1, kappa=2.0, direction="max"):
        """
        Compute acquisition values for PI, EI, UCB.

        Parameters
        ----------
        method : str
            Acquisition method: 'PI', 'EI', or 'UCB'.
        mean : np.ndarray
            Surrogate mean predictions.
        std : np.ndarray
            Surrogate uncertainty estimates.
        y : np.ndarray
            Observed target values (same space as mean).
        xi : float
            Exploration parameter for PI/EI.
        kappa : float
            Exploration parameter for UCB.
        direction : str
            'max' → standard maximization formulas
            'min' → minimization variants (drive toward smallest values)
        """
        eps = 1e-12

        if direction == "max":
            y_best = np.max(y)
            if method == 'PI':
                z = (mean - y_best - xi) / (std + eps)
                return norm.cdf(z)
            elif method == 'EI':
                z = (mean - y_best - xi) / (std + eps)
                return (mean - y_best - xi) * norm.cdf(z) + std * norm.pdf(z)
            elif method == 'UCB':
                return mean + kappa * std

        elif direction == "min":
            y_best = np.min(y)
            if method == 'PI':
                z = (y_best - mean - xi) / (std + eps)
                return norm.cdf(z)
            elif method == 'EI':
                z = (y_best - mean - xi) / (std + eps)
                improvement = (y_best - mean - xi)
                return improvement * norm.cdf(z) + std * norm.pdf(z)
            elif method == 'UCB':
                # Lower Confidence Bound analogue
                return -mean + kappa * std

        else:
            raise ValueError("Unsupported method or direction")
    
    # --- Candidate management ---
    def log_query_candidate(self, results, method, param, x, acq_score, predicted_y=None):
        """Stores a candidate query point and its acquisition score."""
        # Format inputs
        x_formatted = np.array([float(f"{val:.6f}") for val in x])

        # Format predicted output
        pred_val = None
        if predicted_y is not None:
            pred_val = float(f"{predicted_y:.6f}")

        # Format acquisition score
        score_val = float(f"{acq_score:.6f}")
    
        results.append({
            'method': method,
            'xi': param if method in ['PI', 'EI'] else None,
            'kappa': param if method == 'UCB' else None,
            'x': np.array(x),
            'score': acq_score,
            'dim': len(x),
            'predicted_y': predicted_y
        })

#    def find_best_candidate(self, results):
#
#        return max(results, key=lambda r: r['score'])

    def add_points_to_df(self, df, results):
        for result in results:
            point = list(result['x'])
            y_val = result['predicted_y']
            predicted_y = [y_val] if np.isscalar(y_val) else list(y_val)
            row = point + predicted_y + ['week-x']
            df.loc[len(df)] = row

    def train_svm_classifier_on_median(self, X, y):
        """
        Train an SVM classifier on input-output data using median split.
        
        Returns:
        - svm_clf: trained SVM classifier
        - x_scaler: fitted StandardScaler
        """
        median_y = np.median(y)
        labels = (y >= median_y).astype(int)

        x_scaler = StandardScaler()
        X_scaled = x_scaler.fit_transform(X)

        svm_clf = SVC(kernel='rbf', C=1.0)
        svm_clf.fit(X_scaled, labels)

        return svm_clf, x_scaler

    def create_nd_grid(self, bounds, grid_size=50, downsample=True, stride=2):
        """
        Create an N-dimensional grid over the given bounds, with optional downsampling.

        Parameters
        ----------
        bounds : list of tuple
            List of (low, high) tuples specifying the range for each dimension.
            Example: [(0, 1), (0, 1)] for a 2D unit square.
        
        grid_size : int, default=50
            Number of points per axis before downsampling.
        
        downsample : bool, default=False
            Whether to downsample the grid along each axis.
        
        stride : int, default=2
            Downsampling stride. Only used if `downsample=True`.
            Example: stride=2 → take every 2nd point along each axis.

        Returns
        -------
        grid : ndarray of shape (M, D)
            Flattened grid of points, where:
            - D = number of dimensions (len(bounds))
            - M = number of points (depends on downsampling)

        Notes
        -----
        - Full grid size = grid_size^D
        - Downsampled grid size ≈ (grid_size / stride)^D
        - Uses `np.meshgrid` with 'ij' indexing for consistency.
        """
        dim = len(bounds)
        print(f"Using {dim} dimensions, grid size per axis: {grid_size}, downsample: {downsample}, stride: {stride}")

        # Build axes
        axes = []
        for low, high in bounds:
            axis = np.linspace(low, high, grid_size)
            if downsample:
                axis = axis[::stride]
            axes.append(axis)

        # Report sizes
        total_pre = grid_size ** dim
        total_post = len(axes[0]) ** dim
        print(f"Points before downsampling: {total_pre}")
        print(f"Points after downsampling: {total_post}")

        # Meshgrid and flatten
        mesh = np.meshgrid(*axes, indexing='ij')
        grid = np.stack([m.ravel() for m in mesh], axis=-1)

        print(f"Final grid shape: {grid.shape}")
        return grid

    def latin_hypercube_sampling(self, n_samples, n_dim, bounds, random_state=None):
        """
        Generate Latin Hypercube Samples (LHS) for an n-dimensional space.

        Parameters
        ----------
        n_samples : int
            Number of sample points to generate.
        n_dim : int
            Number of dimensions.
        bounds : list of tuple
            List of (low, high) tuples for each dimension.
        random_state : int, optional
            Seed for reproducibility.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_dim)
            Candidate points.
        """
        rng = np.random.default_rng(random_state)

        # Step 1: Divide [0,1] into n_samples intervals
        cut = np.linspace(0, 1, n_samples + 1)

        # Step 2: For each dimension, sample one point from each interval
        u = np.zeros((n_samples, n_dim))
        for j in range(n_dim):
            # draw n_samples values, one from each interval
            u[:, j] = rng.uniform(low=cut[:-1], high=cut[1:], size=n_samples)
            # permute them to break correlation
            rng.shuffle(u[:, j])

        # Step 3: Scale to bounds
        samples = np.zeros_like(u)
        for j, (low, high) in enumerate(bounds):
            samples[:, j] = low + u[:, j] * (high - low)

        return samples

    def create_sample_points(self, bounds, grid_size=50,
                            downsample_grid=True, downsample_stride=2, n_samples=None,
                            random_state=None, sample_strategy="cartesian"):
        """
        Create candidate sample points using different strategies.

        Parameters
        ----------
        bounds : list of tuple
            List of (low, high) tuples specifying the range for each dimension.
        sample_strategy : str, default="cartesian"
            Candidate generation strategy: "cartesian", "lhs", or "sobol".
        grid_size : int, default=50
            Number of points per axis (used for cartesian).
        downsample : bool, default=True
            Whether to downsample the grid (cartesian only).
        stride : int, default=2
            Downsampling stride (cartesian only).
        n_samples : int, optional
            Number of points to sample (lhs/sobol).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        samples : ndarray of shape (M, D)
            Candidate points, where:
            - D = number of dimensions (len(bounds))
            - M = number of points (depends on strategy and parameters).
        """
        dim = len(bounds)

        if sample_strategy == "cartesian":
            # Call your existing Cartesian grid builder
            samples = self.create_nd_grid(bounds, grid_size=grid_size,
                                    downsample=downsample, stride=stride)

        elif sample_strategy == "lhs":
            if n_samples is None:
                n_samples = grid_size
            samples = self.latin_hypercube_sampling(n_samples, dim, bounds, random_state)

        elif sample_strategy == "sobol":
            if n_samples is None:
                n_samples = grid_size
            sampler = qmc.Sobol(d=dim, scramble=True, seed=random_state)
            unit_samples = sampler.random(n_samples)
            samples = np.zeros_like(unit_samples)
            for j, (low, high) in enumerate(bounds):
                samples[:, j] = low + unit_samples[:, j] * (high - low)

        else:
            raise ValueError(f"Unsupported sample_strategy: {sample_strategy}")

        print(f"sample_strategy={sample_strategy}, Final sample shape: {samples.shape}")
        return samples

    def run_flow(self, trainer, X, y,
                xi_values,
                kappa_values,
                grid_sizes,
                methods,
                apply_scaling,
                filter_mode,
                downsample_grid,
                downsample_stride,
                sample_strategy="cartesian",
                optimization_direction="max",
                objective_mode="raw",
                training_mode="exploitation",
                filter_strategy="good",
                out_dir=None,
                ):
        """
        End-to-end pipeline for Bayesian optimisation with a surrogate NN.

        Parameters
        ----------
        trainer : SurrogateTrainer
            Surrogate model trainer instance (MLP, CNN, etc.).
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.
        y : np.ndarray of shape (n_samples,)
            Output target values (radiation readings).
        xi_values : list of float
            Exploration parameters for PI/EI acquisition functions.
        kappa_values : list of float
            Exploration parameters for UCB acquisition function.
        grid_sizes : list of int
            Resolutions of the search grid to sweep.
        methods : list of str
            Acquisition methods to sweep (e.g. ['PI','EI','UCB']).
        apply_scaling : bool
            Whether to scale y before training (StandardScaler).
        filter_mode : str
            Grid filtering strategy (e.g. 'gp', 'svm').
        downsample_grid : bool
            Whether to downsample the grid along each axis.
        downsample_stride : int
            Stride for downsampling (e.g. stride=2 → every 2nd point).
        out_dir : str or Path, optional
            Directory to save plots and CSV outputs.

        Returns
        -------
        results : dict
            Dictionary keyed by grid_size with values:
            (X_grid_filtered, mean_raw, bounds, best, query_results, acq_maps)

            - X_grid_filtered : np.ndarray
                Candidate grid points after filtering.
            - mean_raw : np.ndarray
                Surrogate predictions in **raw (de-scaled)** units.
            - bounds : list of tuple
                Bounds for each input dimension.
            - best : dict
                Best candidate info (method, params, score, x, predicted_y).
            - query_results : list of dict
                All candidate query results with acquisition scores.
            - acq_maps : dict
                Acquisition maps keyed by (method, param).
        """

        # Step 1: preprocess targets (scale if requested)
        # Returns both scaled y and the fitted scaler for later inverse-transform
        y_proc, y_scaler = self.preprocess_y(y, apply_scaling=apply_scaling, objective_mode=objective_mode)

        # Step 2: train surrogate model on scaled targets
        history = trainer.fit(X, y_proc, log_weights=True)
        plotter = PlotUtils(out_dir)

        # Ensure output directory exists
        from pathlib import Path
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        # Plot training diagnostics
        plotter.plot_loss_curve_and_weigth_evolution(history)

        # Step 3: loop over grid sizes
        results = {}
        acq_maps = {}   # store full acquisition maps per method/param
        for grid_size in grid_sizes:
            # Compute bounds from input data
            bounds = self.compute_bounds(X, margin=0.1)

            # Build full grid and downsample if requested
            #X_grid_full = self.create_nd_grid(bounds, int(grid_size),
            #                                downsample_grid, downsample_stride)
            
            # Filter grid using chosen strategy (GP, SVM, etc.)
            X_grid_full = self.create_sample_points(
                            bounds=bounds,
                            grid_size=grid_size,
                            downsample_grid=downsample_grid,
                            downsample_stride=downsample_stride,
                            n_samples=None,
                            random_state=None,
                            sample_strategy=sample_strategy
                    )

            # Filter grid using chosen strategy (GP, SVM, etc.)
            X_grid_filtered = self.filter_grid(X, y_proc, X_grid_full,
                                            filter_mode=filter_mode)

            if X_grid_filtered.shape[0] == 0:
                print(f"⚠️ Skipping grid_size={grid_size}, filter removed all points")
                continue

            # Surrogate predictions (scaled space)
            mean_scaled = trainer.predict(X_grid_filtered)
            print(f"Predictions range (scaled): {np.min(mean_scaled):.4f} to {np.max(mean_scaled):.4f}")
            # 🔧 Inverse-transform predictions back to raw units if scaler exists
            mean_raw = mean_scaled
            if y_scaler is not None:
                print("Inverse-transforming predictions to raw units")
                mean_raw = y_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
                print(f"Predictions range (raw): {np.min(mean_raw):.4f} to {np.max(mean_raw):.4f}")

            # Uncertainty heuristic: distance to nearest training point

            dist = pairwise_distances(X_grid_filtered, X)
            std = np.min(dist, axis=1)

            # Sweep acquisition functions
            query_results = []
            for method in methods:
                if method in ['PI', 'EI']:
                    for xi in xi_values:
                        # Acquisition must be computed in scaled space
                        acq_map = self.compute_acquisition(method, mean_scaled, std, y_proc, xi=xi)
                        best_index = np.argmax(acq_map)
                        next_query = X_grid_filtered[best_index]
                        score = acq_map[best_index]

                        # Candidate prediction logged in raw units
                        pred_val = mean_raw[best_index]

                        self.log_query_candidate(query_results, method, xi,
                                                next_query, score, pred_val)
                        acq_maps[(method, xi)] = acq_map

                elif method == 'UCB':
                    for kappa in kappa_values:
                        # Acquisition must be computed in scaled space
                        acq_map = self.compute_acquisition(method, mean_scaled, std, y_proc, kappa=kappa)
                        best_index = np.argmax(acq_map)
                        next_query = X_grid_filtered[best_index]
                        score = acq_map[best_index]

                        # Candidate prediction logged in raw units
                        pred_val = mean_raw[best_index]

                        self.log_query_candidate(query_results, method, kappa,
                                                next_query, score, pred_val)
                        acq_maps[(method, kappa)] = acq_map

            if not query_results:
                print(f"⚠️ No valid candidates found for grid_size={grid_size}")
                continue

            # Select best candidate
            best = self.find_best_candidate(query_results)

            # Store results for this grid size
            results[grid_size] = (X_grid_filtered, mean_raw, bounds, best,
                                query_results, acq_maps)

        return results

    def train_svm_on_gp_mean(self, X, y, X_grid, percentile=90, kernel=None):
        """
        Train an SVM using GP mean predictions on the grid to label points.
        
        Parameters:
        - X, y: Known data
        - X_grid: Grid points to label and train on
        - percentile: Percentile threshold for 'good' labeling
        - kernel: Optional GP kernel
        
        Returns:
        - svm_clf: Trained SVM
        - x_scaler: Scaler fitted on grid
        """
        if kernel is None:
            kernel = RBF(length_scale=0.1, length_scale_bounds='fixed')

        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
        gp.fit(X, y)

        mean, _ = gp.predict(X_grid, return_std=True)
        threshold = np.percentile(mean, percentile)
        labels = (mean >= threshold).astype(int)

        x_scaler = StandardScaler()
        X_scaled = x_scaler.fit_transform(X_grid)

        svm_clf = SVC(kernel='rbf', C=1.0)
        svm_clf.fit(X_scaled, labels)

        return svm_clf, x_scaler
    

    def train_svm_on_knn_labels(self, X, y, X_grid, percentile=90, knn_k=3):
        """
        Train an SVM using KNN-predicted labels on the grid.
        
        Parameters:
        - X, y: Known data
        - X_grid: Grid points to label and train on
        - percentile: Percentile threshold for 'good' labeling
        - knn_k: Number of neighbors for KNN
        
        Returns:
        - svm_clf: Trained SVM classifier
        - x_scaler: Scaler fitted on grid
        """
        knn = KNeighborsRegressor(n_neighbors=knn_k)
        knn.fit(X, y)
        y_grid = knn.predict(X_grid)

        threshold = np.percentile(y_grid, percentile)
        labels = (y_grid >= threshold).astype(int)

        x_scaler = StandardScaler()
        X_scaled = x_scaler.fit_transform(X_grid)

        svm_clf = SVC(kernel='rbf', C=1.0)
        svm_clf.fit(X_scaled, labels)

        return svm_clf, x_scaler

    def filter_candidates_with_svm(self, candidates_raw, svm_clf, x_scaler):
        """
        Filters candidate points using a trained SVM classifier.

        This function applies a fitted StandardScaler to the candidate points,
        then uses the trained SVM to predict binary labels. It retains only those
        candidates classified as 'good' (label = 1), based on the SVM's decision.

        Parameters:
        ----------
        candidates_raw : np.ndarray of shape (M, N)
            The raw grid or candidate points to be filtered. Each row is a point in N-dimensional space.

        svm_clf : sklearn.svm.SVC
            A trained SVM classifier that was fit on scaled data with binary labels.

        x_scaler : sklearn.preprocessing.StandardScaler
            A fitted scaler used to transform candidate points before SVM prediction.
            Must match the scaler used during SVM training.

        Returns:
        -------
        good_candidates_raw : np.ndarray of shape (K, N)
            A subset of candidates_raw that were predicted as 'good' by the SVM (label = 1).
            If no points are classified as good, the full grid is returned with a warning.
        """
        if svm_clf is None:
            print("No SVM classifier provided. Returning all candidates.")
            return candidates_raw

        candidates_n = x_scaler.transform(candidates_raw)
        predicted_labels = svm_clf.predict(candidates_n)
        good_idx = np.where(predicted_labels == 1)[0]

        if len(good_idx) > 0:
            good_candidates_raw = candidates_raw[good_idx]
            print(f"Reduced candidates to {len(good_candidates_raw)} promising ones using SVM.")
        else:
            good_candidates_raw = candidates_raw
            print("No promising candidates found. Returning full grid.")

        print("Reduced candidates =", good_candidates_raw.shape)
        return good_candidates_raw

    
    def select_next_query_multi(self, X, y, method='PI', xi=0.1, kappa=2.0, grid_size=100, dimension=None,
                                apply_scaling=False, title="Acquisition Function", kernel=None, filter_mode='gp'):
        """
        Selects the next query point using a specified acquisition strategy.

        Parameters:
        - X: np.ndarray of shape (n_samples, n_features), input features
        - y: np.ndarray of shape (n_samples,), output values
        - method: str, one of 'PI', 'EI', 'UCB'
        - xi: float, exploration parameter for PI/EI
        - kappa: float, exploration parameter for UCB
        - grid_size: int, resolution of the search grid
        - apply_scaling: bool, whether to standardize y
        - title: str, plot title
        - kernel: sklearn.gaussian_process.kernels object, optional custom kernel

        Returns:
        - next_query: np.ndarray of shape (n_features,), the selected input point
        - acquisition_map: np.ndarray of shape (grid_size**n,), acquisition values
        - X_grid: np.ndarray of shape (grid_size**n, n_features), grid points evaluated
        - mean: np.ndarray of shape (grid_size**n,), GP mean predictions
        - std: np.ndarray of shape (grid_size**n,), GP std predictions
        """
        if apply_scaling:
            scaler = StandardScaler()
            y = scaler.fit_transform(y.reshape(-1, 1)).ravel()

        n_features = X.shape[1]
        print("Number of features: ", n_features)
        # original bounds
        #bounds = [(0, 1)] * n_features
        #week-3 - changing bounds
        margin = 0.1
        bounds = [(np.min(X[:, i]) - margin, np.max(X[:, i]) + margin) for i in range(X.shape[1])]
        #end week-3 - changing bounds
        X_grid = create_nd_grid(bounds, grid_size, dimension)
        
        # Train strategy using GP mean labeling
        strategy = SVMFilterStrategy(filter_mode, percentile=90, threshold=0.2)
        strategy.fit(X, y, X_grid=X_grid)
        # Filter grid
        X_grid = strategy.filter(X_grid)

        # Use provided kernel or default to fixed RBF
        if kernel is None:
            kernel = RBF(length_scale=0.1, length_scale_bounds='fixed')

        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
        gp.fit(X, y)
        
        mean, std = gp.predict(X_grid, return_std=True)
        print("Mean ", len(mean))
        print("Std  ", len(std))

        if method == 'PI':
            y_max = np.max(y)
            z = (mean - y_max - xi) / (std + 1e-12)
            acquisition_map = norm.cdf(z)

        elif method == 'EI':
            y_max = np.max(y)
            z = (mean - y_max - xi) / (std + 1e-12)
            acquisition_map = (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

        elif method == 'UCB':
            acquisition_map = mean + kappa * std

        else:
            raise ValueError("Unsupported method. Choose from 'PI', 'EI', or 'UCB'.")

        best_index = np.argmax(acquisition_map)
        next_query = X_grid[best_index]

        return next_query, acquisition_map, X_grid

    def select_next_query_multi_fixedNN(self, X, y, method='PI', xi=0.1, kappa=2.0, grid_size=100,
                                        apply_scaling=False, filter_mode='gp', dimension=None):
        """
        Selects the next query point using a fixed (untrained) neural network surrogate model
        and an acquisition function over a filtered input grid.

        Parameters:
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Observed input features.

        y : np.ndarray of shape (n_samples,)
            Observed target values.

        method : str
            Acquisition function to use: 'PI', 'EI', or 'UCB'.

        xi : float
            Exploration parameter for PI/EI.

        kappa : float
            Exploration parameter for UCB.

        grid_size : int
            Number of points per axis in the search grid.

        apply_scaling : bool
            Whether to standardize `y` before filtering.

        filter_mode : str
            SVM filtering mode: 'gp', 'nn', etc.

        dimension : int or None
            If set, downsample grid to use only the first `dimension` features.
            Useful for reducing grid size in high-dimensional spaces.

        Returns:
        -------
        next_query : np.ndarray
            Selected input point with highest acquisition score.

        acquisition_map : np.ndarray
            Acquisition values over filtered grid.

        X_grid_filtered : np.ndarray
            Filtered grid of candidate points.
        """
        n_features = X.shape[1]
        print(f"Input space dimension: {n_features}")

        # Step 1: Compute bounds and downsample dimensions if requested
        bounds = compute_bounds(X)
        if dimension is not None:
            if dimension > n_features:
                raise ValueError(f"Requested dimension {dimension} exceeds feature count {n_features}")
            bounds = bounds[:dimension]
            print(f"Downsampling grid to first {dimension} dimensions.")

        # Step 2: Create grid
        X_grid_full = create_nd_grid(bounds, grid_size)
        print(f"Points before downsampling: {X_grid_full.shape[0]}")

        # Step 3: Initialize and fit fixed predictor
        predictor = FixedPredictor(input_dim=n_features)
        predictor.fit(X)

        # Step 4: Predict surrogate outputs on full grid for filtering
        y_grid = predictor.predict(X_grid_full)

        # Step 5: Filter grid using SVM strategy
        X_grid_filtered = filter_grid(X, y, X_grid_full, filter_mode)
        print(f"Final grid shape: {X_grid_filtered.shape}")

        if X_grid_filtered.shape[0] == 0:
            print("⚠️ Warning: SVM filter removed all grid points. Skipping this configuration.")
            return None, None, None

        # Step 6: Predict again on filtered grid
        mean = predictor.predict(X_grid_filtered)
        std = np.ones_like(mean) * 0.1  # fixed uncertainty

        # Step 7: Compute acquisition and select best point
        acquisition_map = compute_acquisition(method, mean, std, y, xi, kappa)
        best_index = np.argmax(acquisition_map)
        next_query = X_grid_filtered[best_index]

        return next_query, acquisition_map, X_grid_filtered

    def select_next_query_multi_NNtrain(self, X, y, method='PI', xi=0.1, kappa=2.0, grid_size=100,
                                        apply_scaling=False, filter_mode='gp', dimension=None):
        """
        Selects the next query point using a trained neural network surrogate model
        and an acquisition function over a filtered input grid.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Observed input features.

        y : np.ndarray of shape (n_samples,)
            Observed target values.

        method : str
            Acquisition function to use: 'PI', 'EI', or 'UCB'.

        xi : float
            Exploration parameter for PI/EI.

        kappa : float
            Exploration parameter for UCB.

        grid_size : int
            Number of points per axis in the search grid.

        apply_scaling : bool
            Whether to standardize `y` before filtering.

        filter_mode : str
            SVM filtering mode: 'gp', 'nn', etc.

        dimension : int or None
            If set, downsample grid to use only the first `dimension` features.

        Returns
        -------
        next_query : np.ndarray
            Selected input point with highest acquisition score.

        acquisition_map : np.ndarray
            Acquisition values over filtered grid.

        X_grid_filtered : np.ndarray
            Filtered grid of candidate points.
        """
        n_features = X.shape[1]
        print(f"Input space dimension: {n_features}")

        # Step 1: Preprocess y
        y_proc = preprocess_y(y, apply_scaling=apply_scaling)

        # Step 2: Compute bounds and downsample dimensions if requested
        bounds = compute_bounds(X, margin=0.1)
        if dimension is not None:
            if dimension > n_features:
                raise ValueError(f"Requested dimension {dimension} exceeds feature count {n_features}")
            bounds = bounds[:dimension]
            print(f"Downsampling grid to first {dimension} dimensions.")

        # Step 3: Create grid
        X_grid_full = create_nd_grid(bounds, grid_size)
        print(f"Points before filtering: {X_grid_full.shape[0]}")

        # Step 4: Filter grid using SVM strategy
        X_grid_filtered = filter_grid(X, y_proc, X_grid_full, filter_mode=filter_mode)
        print(f"Final grid shape: {X_grid_filtered.shape}")

        if X_grid_filtered.shape[0] == 0:
            print("⚠️ Warning: SVM filter removed all grid points. Skipping this configuration.")
            return None, None, None

        # Step 5: Train surrogate (NN) on observed data
        trainer = SurrogateTrainer(input_dim=n_features)
        trainer.fit(X, y_proc)

        # Step 6: Predict surrogate outputs on filtered grid
        mean = trainer.predict(X_grid_filtered)

        # Step 7: Estimate uncertainty (distance-based heuristic)
        dist = pairwise_distances(X_grid_filtered, X)
        std = np.min(dist, axis=1)

        # Step 8: Compute acquisition via shared utility
        acquisition_map = compute_acquisition(method, mean, std, y_proc, xi=xi, kappa=kappa)

        # Step 9: Select best point
        best_index = np.argmax(acquisition_map)
        next_query = X_grid_filtered[best_index]

        return next_query, acquisition_map, X_grid_filtered

    def find_best_candidate(self, results):
        if not results:
            raise ValueError("No candidates to evaluate.")
        print("Total number of points to evaluate:", len(results))
        for item in results:
            print(item)
        """Returns the best candidate point based on acquisition score."""

        best = max(results, key=lambda r: r['score'])

        print("🔍 Best candidate (highest acquisition score):")
        print(f"Method: {best['method']}")
        if best['method'] in ['PI', 'EI']:
            print(f"xi: {best['xi']}")
        else:
            print(f"kappa: {best['kappa']}")
        print(f"Dimensions: {best['dim']}")
        print(f"x: {np.array2string(best['x'], precision=6, separator=', ')}")
        print(f"Score: {best['score']:.6f}")

        return best

    # Example DataFrame
    def add_points_to_df(df, results):
        """
        Appends rows to df using:
        - x values from each result
        - an empty string for 'yield'
        - the method name as 'source'
        """
        for result in results:
            point = list(result['x'])
            y_val = result['predicted_y']
            # If it's a scalar, wrap it in a list
            if np.isscalar(y_val):
                predicted_y = [y_val]
            else:
                predicted_y = list(y_val)
            row = point + predicted_y + ['week-x']
            df.loc[len(df)] = row