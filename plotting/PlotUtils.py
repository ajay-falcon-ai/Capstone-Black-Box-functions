import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os




class PlotUtils:
    
    def __init__(self, output_dir=".", cfg=None):
        """
        output_dir: directory where plots will be saved
        """
        self.config = cfg
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_acquisition_heatmap(
        self, 
        X,                   # shape (n_samples, 2): queried points
        acquisition_map,     # shape (grid_size**2,): acquisition values
        X_grid,              # shape (grid_size**2, 2): grid points
        next_query=None,     # shape (2,), optional: next query point
        title='Acquisition Heatmap',
        cmap='viridis',
        color_range=(-3, 1), # default color scale for consistency
        save_path=None       # optional: path to save plot
    ):
        grid_size = int(np.sqrt(len(acquisition_map)))
        acquisition_2d = acquisition_map.reshape(grid_size, grid_size)

        # Compute extent from grid
        x_min, x_max = X_grid[:, 0].min(), X_grid[:, 0].max()
        y_min, y_max = X_grid[:, 1].min(), X_grid[:, 1].max()
        extent = [x_min, x_max, y_min, y_max]

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            acquisition_2d,
            origin='lower',
            extent=extent,
            cmap=cmap,
            aspect='auto',
            vmin=color_range[0],
            vmax=color_range[1]
        )
        plt.colorbar(im, label='Acquisition Value')

        # Queried points
        plt.scatter(X[:, 0], X[:, 1], c='white', edgecolors='black', label='Queried Points')

        # Next query
        if next_query is not None:
            plt.plot(next_query[0], next_query[1], 'r*', markersize=14, label='Next Query')

        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_output_points(
        self,
        df,
        scale_factor=1.0,
        yield_scale_factor=1.0,
        nan_marker='o',
        debug=False,
        plot_candidates=True,   # toggle candidate plotting
        grid_size=None
    ):
        """
        Plot inputs and outputs with binary colours:
        - Inputs: all columns except the last two, each in a distinct colour
        - Real outputs: source != 'week-x' (black, from penultimate column)
        - Candidate outputs: source == 'week-x' (red, from penultimate column)
            Can be toggled with plot_candidates flag.
        """

        fig, ax = plt.subplots()
        df = df.copy()
        columns = df.columns

        if len(columns) < 2:
            raise ValueError("DataFrame must have at least 2 columns: output and source.")

        # Identify columns
        input_cols = list(columns[:-2])              # all inputs
        output_col = columns[-2]                     # output
        source_col = columns[-1]                     # source

        # Convert numerics
        df[output_col] = pd.to_numeric(df[output_col], errors='coerce')
        for col in input_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Masks
        real_mask = df[source_col] != "week-x"
        cand_mask = df[source_col] == "week-x"
        missing_mask = df[output_col].isna()

        # Debug
        if debug:
            print(f"Total rows: {len(df)}")
            print(f"Real rows: {real_mask.sum()}, Candidate rows: {cand_mask.sum()}")
            print(f"NaNs in output: {missing_mask.sum()}")
            print(f"Input columns ({len(input_cols)}): {input_cols}")

        # --- Plot inputs (each input in a different colour) ---
        if input_cols:
            cmap = cm.get_cmap('tab20', len(input_cols))
            for i, col in enumerate(input_cols):
                ax.scatter(
                    df.index,
                    (df[col] * scale_factor).values,
                    marker='.',
                    color=cmap(i),
                    alpha=0.75,
                    label=f"Input: {col}"
                )

        # --- Plot outputs ---
        if real_mask.any():
            ax.scatter(
                df.index[real_mask],
                (df.loc[real_mask, output_col] * yield_scale_factor).values,
                marker='x',
                color='black',
                label='Real outputs (file + weekly)'
            )

        if plot_candidates and cand_mask.any():
            ax.scatter(
                df.index[cand_mask],
                (df.loc[cand_mask, output_col] * yield_scale_factor).values,
                marker='x',
                color='red',
                s=80,
                label='Candidates (week-x)'
            )

        # Optional: mark rows with missing output
        if missing_mask.any():
            ax.scatter(
                df.index[missing_mask],
                np.zeros(missing_mask.sum()),
                marker=nan_marker,
                color='gray',
                label='Missing output'
            )

        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title(f'Output Curve with Inputs Grid={grid_size}')
        ax.grid(True)
        ax.legend(ncols=2, fontsize=9)
        # --- Always save and show ---
        full_path = os.path.join(self.output_dir, f"output_plot{grid_size}.png")
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        plt.show()

        return None

    def plot_surrogate_generic(self, X_grid, y_pred,
                                X_filtered=None,
                                bounds=None,
                                candidates=None,
                                candidate_labels=None,
                                title="NN Surrogate",
                                initial_inputs=None,
                                initial_outputs=None,
                                weekly_points=None,
                                grid_size=None):
        """
        Plot surrogate predictions with optional filtered points, candidate queries,
        and overlays for initial_inputs, initial_outputs and weekly_points.

        Parameters
        ----------
        X_grid : array of shape (N, d)
            Grid points (can be full or filtered)
        y_pred : array of shape (N,)
            Surrogate predictions
        X_filtered : array of shape (M, d), optional
            Points actually evaluated (highlighted separately)
        bounds : list of (min, max) for each dimension, optional
            Defines the bounding box of the unfiltered grid
        candidates : array of shape (K, d), optional
            Candidate query points selected via Bayesian optimisation
        candidate_labels : list of str, optional
            Labels for candidate points (e.g., iteration index or acquisition score)
        title : str
            Plot title
        initial_inputs : array-like (n_init, d) or None
            Initial training input points to overlay
        initial_outputs : array-like (n_init,) or None
            Corresponding outputs for initial_inputs
        weekly_points : array-like (n_weekly, d) or None
            Weekly points to highlight on the plot
        """
        import numpy as np
        import matplotlib.pyplot as plt

        X_grid = np.asarray(X_grid)
        y_pred = np.asarray(y_pred).ravel()
        d = X_grid.shape[1]

        fig, ax = plt.subplots()

        # Helper to plot bounds rectangle for two dims
        def _plot_bounds(ax, b, dims=(0, 1)):
            if b is None:
                return
            x_min, x_max = b[dims[0]]
            y_min, y_max = b[dims[1]]
            rect_x = [x_min, x_max, x_max, x_min, x_min]
            rect_y = [y_min, y_min, y_max, y_max, y_min]
            ax.plot(rect_x, rect_y, color="black", linewidth=2, label="Bounds")

        # 1D plotting
        if d == 1:
            ax.plot(X_grid[:, 0], y_pred, label="NN surrogate", zorder=1)

            if X_filtered is not None:
                Xf = np.asarray(X_filtered)
                yf = np.interp(Xf[:, 0], X_grid[:, 0], y_pred)
                ax.scatter(Xf[:, 0], yf, color="red", label="Filtered points",
                        s=25, zorder=4, edgecolor="k")

            if candidates is not None:
                cand = np.asarray(candidates)
                yc = np.interp(cand[:, 0], X_grid[:, 0], y_pred)
                ax.scatter(cand[:, 0], yc, color="blue", marker="*",
                        s=60, label="Candidate queries", zorder=5, edgecolor="k")
                if candidate_labels is not None:
                    for i, pt in enumerate(cand):
                        ax.text(pt[0], np.interp(pt[0], X_grid[:, 0], y_pred),
                                str(candidate_labels[i]), color="blue",
                                fontsize=8, ha="left")

            if initial_inputs is not None:
                xi = np.asarray(initial_inputs).ravel()
                if initial_outputs is not None:
                    io = np.asarray(initial_outputs).ravel()
                    ax.scatter(xi, io, marker="D", s=40, color="red",
                            edgecolor="k", label="Initial outputs", zorder=6)
                else:
                    yi = np.interp(xi, X_grid[:, 0], y_pred)
                    ax.scatter(xi, yi, marker="o", s=40, color="tab:blue",
                            edgecolor="w", label="Initial inputs", zorder=6)

            if weekly_points is not None:
                xw = np.asarray(weekly_points).ravel()
                yw = np.interp(xw, X_grid[:, 0], y_pred)
                ax.scatter(xw, yw, marker="^", s=40, color="tab:green",
                        edgecolor="k", label="Weekly points", zorder=7)

            if bounds is not None:
                x_min, x_max = bounds[0]
                ax.axvline(x_min, color="black", linestyle="--", label="Bounds")
                ax.axvline(x_max, color="black", linestyle="--")

            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()
            
            # --- Always save and show ---
            # take gridsize from config
            fname = f"surrogate_generic_plot_grid{grid_size}.png" if grid_size else "surrogate_generic_plot.png"
            full_path = os.path.join(self.output_dir, fname)
            plt.savefig(full_path, dpi=300, bbox_inches="tight")
            plt.show()
            return None

        # 2D plotting
        if d == 2:
            scatter = ax.scatter(X_grid[:, 0], X_grid[:, 1],
                                c=y_pred, cmap="viridis", s=20, zorder=1)
            fig.colorbar(scatter, ax=ax, label="Model prediction")

            if X_filtered is not None:
                Xf = np.asarray(X_filtered)
                ax.scatter(Xf[:, 0], Xf[:, 1], c="red", edgecolor="black",
                        s=25, label="Filtered points", zorder=4)

            if candidates is not None:
                cand = np.asarray(candidates)
                ax.scatter(cand[:, 0], cand[:, 1], c="blue", marker="*",
                        s=60, label="Candidate queries", zorder=5, edgecolor="k")
                if candidate_labels is not None:
                    for i, pt in enumerate(cand):
                        ax.text(pt[0], pt[1], str(candidate_labels[i]),
                                color="blue", fontsize=8, ha="left")

            if initial_inputs is not None:
                init_in = np.asarray(initial_inputs)
                if init_in.ndim == 2 and init_in.shape[1] >= 2:
                    ax.scatter(init_in[:, 0], init_in[:, 1], marker="o", s=40,
                            color="tab:blue", edgecolor="white", linewidth=0.6,
                            label="Initial inputs", zorder=6)
                    if initial_outputs is not None:
                        ax.scatter(init_in[:, 0], init_in[:, 1], marker="D", s=40,
                                color="red", edgecolor="k",
                                label="Initial outputs", zorder=7)

            if weekly_points is not None:
                weekly = np.asarray(weekly_points)
                if weekly.ndim == 2 and weekly.shape[1] >= 2:
                    ax.scatter(weekly[:, 0], weekly[:, 1], marker="^", s=40,
                            color="tab:green", edgecolor="k",
                            label="Weekly points", zorder=8)

            if bounds is not None:
                _plot_bounds(ax, bounds, dims=(0, 1))

            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_title(title)
            ax.legend(loc="best", fontsize="small")
            fig.tight_layout()
            
            # --- Always save and show ---
            # take gridsize from config
            fname = f"surrogate_generic_plot_grid{grid_size}.png" if grid_size else "surrogate_generic_plot.png"
            full_path = os.path.join(self.output_dir, fname)
            plt.savefig(full_path, dpi=300, bbox_inches="tight")
            plt.show()
            return None

        # d > 2: project to first two dims (0,1)
        i, j = 0, 1
        scatter = ax.scatter(X_grid[:, i], X_grid[:, j],
                            c=y_pred, cmap="viridis", s=20, zorder=1)
        fig.colorbar(scatter, ax=ax, label="Model prediction")

        if X_filtered is not None:
            Xf = np.asarray(X_filtered)
            ax.scatter(Xf[:, i], Xf[:, j], c="red", edgecolor="black",
                    s=25, label="Filtered points", zorder=4)

        if candidates is not None:
            cand = np.asarray(candidates)
            ax.scatter(cand[:, i], cand[:, j], c="blue", marker="*",
                    s=60, label="Candidate queries", zorder=5, edgecolor="k")
            if candidate_labels is not None:
                for idx, pt in enumerate(cand):
                    ax.text(pt[i], pt[j], str(candidate_labels[idx]),
                            color="blue", fontsize=8, ha="left")

        if initial_inputs is not None:
            init_in = np.asarray(initial_inputs)
            if init_in.ndim == 2 and init_in.shape[1] > max(i, j):
                ax.scatter(init_in[:, i], init_in[:, j], marker="o", s=40,
                        color="tab:blue", edgecolor="white", linewidth=0.6,
                        label="Initial inputs", zorder=6)
                if initial_outputs is not None:
                    ax.scatter(init_in[:, i], init_in[:, j], marker="D", s=40,
                            color="red", edgecolor="k",
                            label="Initial outputs", zorder=7)

        if weekly_points is not None:
            weekly = np.asarray(weekly_points)
            if weekly.ndim == 2 and weekly.shape[1] > max(i, j):
                ax.scatter(weekly[:, i], weekly[:, j], marker="^", s=40,
                        color="tab:green", edgecolor="k",
                        label="Weekly points", zorder=8)
        if bounds is not None:
            _plot_bounds(ax, bounds, dims=(i, j))

        ax.set_xlabel(f"x{i}")
        ax.set_ylabel(f"x{j}")
        ax.set_title(f"{title} (dims {i},{j})")
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        # --- Always save and show ---
        # take gridsize from config
        fname = f"surrogate_generic_plot_grid{grid_size}.png" if grid_size else "surrogate_generic_plot.png"
        full_path = os.path.join(self.output_dir, fname)
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        plt.show()
        return None

    def plot_loss_curve(self, history, title="Training Loss Curve"):
        """
        Plot training and validation loss from a history dict and show the figure.
        Expects history to contain keys "train_losses" and "val_losses".
        """
        plt.plot(history["train_losses"], label="Training loss")
        plt.plot(history["val_losses"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.show()
        return None

    def plot_weight_evolution(self, history, layer_name="model.0.weight", stat="mean", title="Weight Evolution"):
        """
        Plot weight evolution after training using the history dict.

        Parameters
        ----------
        history : dict returned by SurrogateTrainer.fit
        layer_name : str, name of the layer to track (e.g. 'model.0.weight')
        stat : str, which statistic to plot ('mean' or 'std')
        title : str, plot title
        """
        if history.get("weights") is None:
            raise ValueError("No weight log found. Run fit with log_weights=True.")

        values = [epoch_stats[layer_name][stat] for epoch_stats in history["weights"]]

        plt.plot(values)
        plt.xlabel("Epoch")
        plt.ylabel(f"{stat} value")
        plt.title(title)
        plt.show()
        return None
        
    def plot_loss_curve_and_weigth_evolution(self, history, title="Training Evolution"):
        """
        Plot training & validation loss together with weight mean evolution.

        Parameters
        ----------
        history : dict
            Expected keys: "train_losses", "val_losses", "weights".
        title : str
            Plot title.
        """
        plt.plot(history["train_losses"], label="Training loss")
        plt.plot(history["val_losses"], label="Values loss")

        values = [epoch_stats["model.0.weight"]["mean"] for epoch_stats in history["weights"]]
        plt.plot(values, label="Weight mean")

        plt.xlabel("Epoch")
        plt.ylabel("Training Loss, Values Loss, Weight mean")
        plt.title(title)
        plt.legend()
        
        # --- Always save and show ---
        fname = "plot_loss_curve_and_weigth_evolution.png"
        full_path = os.path.join(self.output_dir, fname)
        plt.savefig(full_path, dpi=300, bbox_inches="tight")

        plt.show()
        return None

    def plot_weight_evolution_all(self, history, layer_name="model.0.weight", stat="mean",
                                title="Weight Evolution"):
        """
        Plot weight evolution after training using the history dict.

        Plots:
        - mean line for the requested layer(s)
        - shaded band mean +/- std
        - bias mean (if available) as dashed line
        - per-epoch change of the mean on a secondary y-axis

        Parameters
        ----------
        history : dict returned by SurrogateTrainer.fit
        layer_name : str or list of str, name(s) of the layer(s) to track (e.g. 'model.0.weight')
        stat : str, unused for selection here but kept for compatibility ('mean' or 'std')
        title : str, plot title
        """
        if history.get("weights") is None:
            raise ValueError("No weight log found. Run fit with log_weights=True.")

        # allow single name or list of names
        if isinstance(layer_name, (list, tuple)):
            layers = list(layer_name)
        else:
            layers = [layer_name]

        epochs = np.arange(len(history["weights"]))
        fig, ax = plt.subplots()

        for lname in layers:
            # extract mean and std for each epoch (if missing, raise)
            try:
                means = np.array([epoch[lname]["mean"] for epoch in history["weights"]], dtype=float)
                stds  = np.array([epoch[lname]["std"]  for epoch in history["weights"]], dtype=float)
            except KeyError:
                raise KeyError(f"Layer '{lname}' not found in history['weights'] entries.")

            # plot mean and shaded std band
            ax.plot(epochs, means, label=f"{lname} mean")
            ax.fill_between(epochs, means - stds, means + stds, alpha=0.2, label=f"{lname} ± std")

            # try to plot corresponding bias if present
            bias_name = lname.replace("weight", "bias")
            if any(bias_name in epoch for epoch in history["weights"]):
                bias_means = np.array([epoch[bias_name]["mean"] for epoch in history["weights"]], dtype=float)
                ax.plot(epochs, bias_means, linestyle="--", label=f"{bias_name} mean")

        # secondary axis: per-epoch change of the first requested layer's mean
        primary_means = np.array([epoch[layers[0]]["mean"] for epoch in history["weights"]], dtype=float)
        changes = np.concatenate([[0.0], np.diff(primary_means)])  # first epoch change = 0
        ax2 = ax.twinx()
        ax2.plot(epochs, changes, color="tab:gray", alpha=0.9, linewidth=1, label=f"{layers[0]} mean Δ")
        ax2.set_ylabel("Per-epoch change", color="tab:gray")
        ax2.tick_params(axis="y", labelcolor="tab:gray")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.grid(alpha=0.3)

        # combine legends from both axes
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize="small")

        fig.tight_layout()
        plt.show()
        return None

    def plot_training_summary_simple(
        self,
        history,
        layer_name="model.0.weight",
        stat="mean",                # "mean" or "std"
        title="Training: Losses and Weight Trend",
    ):
        """
        Simple combined plot for train/val losses and one layer weight stat.
        Expects history = {"train_losses": [...], "val_losses": [...], "weights": [epoch_dicts] }.
        Each epoch_dict maps layer_name -> {"mean": float, "std": float}.
        """

        # --- losses
        train = np.asarray(history.get("train_losses", [])) if isinstance(history, dict) else np.asarray([])
        val = np.asarray(history.get("val_losses", [])) if isinstance(history, dict) else np.asarray([])

        # --- weight series (extract stat from per-epoch dicts)
        w_series = None
        weights_seq = history.get("weights") if isinstance(history, dict) else None
        if weights_seq:
            vals = []
            for epoch_stats in weights_seq:
                if not isinstance(epoch_stats, dict):
                    vals.append(np.nan)
                    continue
                entry = epoch_stats.get(layer_name)
                if entry is None:
                    vals.append(np.nan)
                    continue
                # entry expected to be {'mean':..., 'std':...}
                if isinstance(entry, dict):
                    v = entry.get(stat, entry.get("mean", np.nan))
                    try:
                        vals.append(float(v))
                    except Exception:
                        vals.append(np.nan)
                else:
                    # fallback: numeric scalar/array
                    try:
                        arr = np.asarray(entry)
                        vals.append(float(np.nanmean(arr)) if arr.size > 0 else np.nan)
                    except Exception:
                        vals.append(np.nan)
            w_series = np.asarray(vals, dtype=float)

        # --- build figure (use default matplotlib sizing)
        fig, ax1 = plt.subplots()
        plotted = False

        if train.size > 0:
            epochs = np.arange(len(train))
            ax1.plot(epochs, train, marker="o", color="C0", label="train loss")
            plotted = True

        if val.size > 0:
            epochs_val = np.arange(len(val))
            ax1.plot(epochs_val, val, marker="s", linestyle="--", color="C1", label="val loss")
            plotted = True

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)

        ax2 = ax1.twinx()
        if w_series is not None and w_series.size > 0 and not np.all(np.isnan(w_series)):
            e_w = np.arange(len(w_series))
            ax2.plot(e_w, w_series, marker="X", color="C3", linewidth=2, label=f"{layer_name} {stat}")
            ax2.set_ylabel(f"{layer_name} {stat}")
            plotted = True

            # if std available and stat == "mean", try to plot shaded band using stored std
            if stat == "mean" and weights_seq:
                std_vals = []
                for epoch_stats in weights_seq:
                    if isinstance(epoch_stats, dict) and layer_name in epoch_stats:
                        entry = epoch_stats[layer_name]
                        if isinstance(entry, dict) and "std" in entry:
                            try:
                                std_vals.append(float(entry["std"]))
                                continue
                            except Exception:
                                pass
                    std_vals.append(np.nan)
                std_arr = np.asarray(std_vals, dtype=float)
                if np.any(~np.isnan(std_arr)):
                    lower = w_series - std_arr
                    upper = w_series + std_arr
                    ax2.fill_between(e_w, lower, upper, color="C3", alpha=0.15)
        else:
            ax2.set_ylabel("")

        # combined legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize="small")

        plt.title(title)
        plt.tight_layout()

        if not plotted:
            ax1.text(0.5, 0.5, "No training or weight history available", ha="center", va="center")
            ax1.set_axis_off()

        plt.show()
        return None

    def extract_candidates_and_labels(self, query_results, include_score=False):
        """
        Convert query_results into candidate points and labels for plotting.
        
        Parameters
        ----------
        query_results : list of dict
            Logged candidates from log_query_candidate
        include_score : bool
            Whether to append acquisition score to labels
        
        Returns
        -------
        candidates : np.ndarray of shape (K, d)
        labels : list of str
        """
        candidates = np.array([entry['x'] for entry in query_results])
        labels = []
        for entry in query_results:
            if entry['method'] in ['PI', 'EI']:
                label = f"{entry['method']} (xi={entry['xi']})"
            else:  # UCB
                label = f"{entry['method']} (kappa={entry['kappa']})"
            if include_score:
                label += f", score={entry['score']:.3f}"
            labels.append(label)
        return candidates, labels


    def plot_acquisition_heatmap(
        X,                   # shape (n_samples, 2): queried points
        acquisition_map,     # shape (grid_size**2,): acquisition values
        X_grid,              # shape (grid_size**2, 2): grid points
        next_query=None,     # shape (2,), optional: next query point
        title='Acquisition Heatmap',
        cmap='viridis',
        color_range=(-3, 1), # default color scale for consistency
        save_path=None       # optional: path to save plot
    ):
        grid_size = int(np.sqrt(len(acquisition_map)))
        acquisition_2d = acquisition_map.reshape(grid_size, grid_size)

        # Compute extent from grid
        x_min, x_max = X_grid[:, 0].min(), X_grid[:, 0].max()
        y_min, y_max = X_grid[:, 1].min(), X_grid[:, 1].max()
        extent = [x_min, x_max, y_min, y_max]

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            acquisition_2d,
            origin='lower',
            extent=extent,
            cmap=cmap,
            aspect='auto',
            vmin=color_range[0],
            vmax=color_range[1]
        )
        plt.colorbar(im, label='Acquisition Value')

        # Queried points
        plt.scatter(X[:, 0], X[:, 1], c='white', edgecolors='black', label='Queried Points')

        # Next query
        if next_query is not None:
            plt.plot(next_query[0], next_query[1], 'r*', markersize=14, label='Next Query')

        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
