#!/usr/bin/env python3
"""
Generic runner for function flows.

Usage:
    python run_function.py --config path/to/config.yaml
    python run_function.py --config path/to/config.yaml --data-module Function1DataHandler --data-class Function1DataHandler
"""
import argparse
import importlib
import json
from logging import log
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


import numpy as np
import pandas as pd

def enable_full_df_print():
    """
    Monkey-patch pandas DataFrame __repr__ and __str__ so that
    print(df) always shows the full DataFrame without truncation.
    """
    pd.DataFrame.__repr__ = lambda self: self.to_string(max_rows=None, max_cols=None)
    pd.DataFrame.__str__  = lambda self: self.to_string(max_rows=None, max_cols=None)

def disable_full_df_print():
    """
    Restore pandas default repr/str behavior (truncated display).
    """
    pd.DataFrame.__repr__ = pd.core.frame.DataFrame.__repr__
    pd.DataFrame.__str__  = pd.core.frame.DataFrame.__str__

def force_text_repr(max_rows=None, max_cols=None):
    """
    Force pandas DataFrames to always display using the full text repr
    instead of the truncated HTML table in Jupyter notebooks.
    By default shows all rows/columns, but you can pass limits.
    """
    pd.options.display.notebook_repr_html = False
    pd.options.display.html.use_mathjax = False
    pd.DataFrame.__repr__ = lambda self: self.to_string(max_rows=max_rows, max_cols=max_cols)
    pd.DataFrame.__str__  = lambda self: self.to_string(max_rows=max_rows, max_cols=max_cols)

import yaml

# Project imports (adjust package paths if your repo layout differs)
from optimisation.BayesOptUtils import BayesOptUtils
from plotting.PlotUtils import PlotUtils
from models.SurrogateNN import SurrogateTrainer
from models.SurrogateCNN import SurrogateCNNTrainer


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg

def instantiate_data_handler(module_name, class_name):
    """
    Robustly import module and instantiate the data handler class.
    Tries:
      1) importlib.import_module(module_name)
      2) add cwd/repo root to sys.path and retry
      3) load module from a local file named <module_name>.py using SourceFileLoader
    Returns an instance of the class.
    """
    mod = None
    # 1) Try normal import
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        mod = None

    # 2) Try adding cwd / repo root to sys.path and retry
    if mod is None:
        cwd = Path.cwd()
        repo_root = cwd.parent.resolve()
        for p in (cwd, repo_root):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            mod = None

    # 3) Fallback: try to find a local file named <module_name>.py in cwd or repo_root
    if mod is None:
        try:
            from importlib.machinery import SourceFileLoader
            import importlib.util
            candidates = [
                Path.cwd() / f"{module_name}.py",
                Path.cwd() / module_name / "__init__.py",
                repo_root / f"{module_name}.py",
                repo_root / module_name / "__init__.py",
            ]
            found = None
            for c in candidates:
                if c.exists():
                    found = c
                    break
            if found:
                loader_name = f"local_{module_name}"
                loader = SourceFileLoader(loader_name, str(found))
                spec = importlib.util.spec_from_loader(loader_name, loader)
                mod = importlib.util.module_from_spec(spec)
                loader.exec_module(mod)
        except Exception:
            mod = None

    if mod is None:
        raise ImportError(
            f"Could not import module '{module_name}' (tried normal import, sys.path adjustments, and local file fallback)."
        )

    if not hasattr(mod, class_name):
        raise AttributeError(f"Module '{module_name}' does not define class '{class_name}'")

    cls = getattr(mod, class_name)
    return cls()


def safe_get(cfg, *keys, default=None):
    """Helper to get nested config values: safe_get(cfg, 'a', 'b', default=x)"""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

class Tee:
    """
    A drop-in replacement for sys.stdout that duplicates all printed output:
    - It still prints to the notebook
    - It also writes the same text to a log file
    """
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout  # original notebook stdout

    def write(self, message):
        self.stdout.write(message)   # print to notebook
        self.file.write(message)     # write to file

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def setup_results_logging(cfg):
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H-%M-%S")
    print(time_str)

    out_dir = Path(cfg.get("out_dir", "results")) / f"run_{time_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Outputs will be written to:", out_dir)

    # --- Save a copy of the config ---
    config_copy_path = out_dir / "config_copy.yaml"
    with open(config_copy_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Saved a copy of the config : {config_copy_path}")

    # --- Activate tee logging (print to notebook AND to file) ---
    log_path = out_dir / "notebook_output.txt"
    sys.stdout = Tee(log_path)      # writes to file + notebook
    sys.stderr = sys.stdout         # capture errors too

    return out_dir

def run_pipeline(cfg, dh, out_dir=None):
    enable_full_df_print()
    """
    Run the generic pipeline using the provided config and data handler instance.
    Returns the results dict from BayesOptUtils.run_flow.
    """

    plotter = PlotUtils(out_dir, cfg=cfg)
    utils = BayesOptUtils()

    # Build trainer if CNN surrogate is used
    print("Creating utilities and trainer...")
    trainer_cfg = cfg.get("trainer", {})
    trainer = build_model(trainer_cfg, input_dim=dh.inputs.shape[1])

    bo_cfg = cfg.get("bayesopt", {})
    mod_cfg = cfg.get("model", {})
    opt_direction = bo_cfg.get("optimization_direction", "max")
    print("Optimization direction:", opt_direction)
    objective_mode = mod_cfg.get("objective_mode", "raw")
    print("Objective mode:", objective_mode)

    print("Running Bayesian optimisation flow with config:", json.dumps(bo_cfg, default=str))
    results = utils.run_flow(
        trainer,
        dh.inputs,
        dh.outputs,
        xi_values=bo_cfg.get("xi_values"),
        kappa_values=bo_cfg.get("kappa_values"),
        grid_sizes=bo_cfg.get("grid_sizes"),
        methods=bo_cfg.get("methods"),
        apply_scaling=bo_cfg.get("apply_scaling", True),
        filter_mode=bo_cfg.get("filter_mode", None),
        sample_strategy=bo_cfg.get("sample_strategy", "cartesian"),
        downsample_grid=bo_cfg.get("downsample_grid", True),
        downsample_stride=bo_cfg.get("downsample_stride", 2),
        optimization_direction=opt_direction,
        objective_mode=objective_mode,
        filter_training_mode=bo_cfg.get("filter_training_mode", "exploitation"),
        filter_strategy=bo_cfg.get("filter_strategy", "good"),
        out_dir=out_dir,
        config=cfg
    )

    # Build base dataframe once (non-destructive)
    df_global = dh.build_dataframe()
    print("Base dataframe built with shape:", df_global.shape)

    # --- Branch depending on model type ---
    if mod_cfg.get("type") == "llm":
        # LLM results are keyed by "llm"
        X_grid_filtered, mean, bounds, best, query_results, acq_maps = results["llm"]

        print("\n=== Results from LLM candidate generation ===")
        print("Best candidate:", best)

        # Extract candidate arrays
        cand_array, y_pred_array = dh.extract_candidate_arrays(query_results)

        # Append candidates to dataframe
        df_copy, new_indices = dh.append_candidates_to_df_copy(
            candidates=cand_array,
            predicted_y=y_pred_array,
            source_label=cfg.get("source_label", "llm"),
            feature_cols=cfg.get("feature_cols", None),
            yield_col=cfg.get("yield_col", None),
            source_col=cfg.get("source_col", "source"),
        )

        print(df_copy)
        plotter.plot_output_points(df_copy, scale_factor=10)

        # Save CSV
        csv_path = out_dir / "df_llm_candidates.csv"
        df_copy.to_csv(csv_path, index=False, float_format="%.6f")
        print("Saved LLM candidates to:", csv_path)

    else:
        # CNN surrogate path: loop over grid sizes
        for grid_size, (X_grid_filtered, mean, bounds, best, query_results, acq_maps) in results.items():
            print(f"\n=== Results for grid_size={grid_size} ===")
            candidates, labels = plotter.extract_candidates_and_labels(query_results, include_score=True)

            # Surrogate plot
            plotter.plot_surrogate_generic(
                X_grid_filtered,
                mean,
                X_filtered=X_grid_filtered,
                bounds=bounds,
                candidates=candidates,
                candidate_labels=labels,
                title=f"Surrogate predictions (grid_size={grid_size})",
                initial_inputs=dh.inputs,
                initial_outputs=dh.outputs,
                weekly_points=None,
                grid_size=grid_size
            )

            # Extract candidate arrays
            cand_array, y_pred_array = dh.extract_candidate_arrays(query_results)

            # Append candidates to dataframe
            df_copy, new_indices = dh.append_candidates_to_df_copy(
                candidates=cand_array,
                predicted_y=y_pred_array,
                source_label=cfg.get("source_label", "week-x"),
                feature_cols=cfg.get("feature_cols", None),
                yield_col=cfg.get("yield_col", None),
                source_col=cfg.get("source_col", "source"),
            )

            # Sort and scale
            df_copy_sorted = df_copy.sort_values(by=df_copy.columns[-2], ascending=False, na_position="last").reset_index(drop=True)
            col = df_copy_sorted.columns[-2]
            max_val = df_copy_sorted[col].max(skipna=True)
            scaled_col = f"{col}-scaled"
            df_copy_sorted[scaled_col] = (df_copy_sorted[col].astype(float) * (10.0 / max_val)) if (np.isfinite(max_val) and max_val != 0) else np.nan

            # Save CSV
            csv_path = build_grid_filename(out_dir, cfg, grid_size=grid_size)
            df_copy_sorted.to_csv(csv_path, index=False) # store the values as they are, float_format="%.6f")
            print("Saved dataframe to:", csv_path)

            print("Best candidate:", best)
            print(df_copy_sorted)

    print("\nPipeline finished.")
    return results

def build_model(cfg, input_dim):
    """
    Factory to build surrogate model based on config.
    """
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "mlp").lower()
    params = model_cfg.get("params", {})

    if model_type == "mlp":
        return SurrogateTrainer(
            input_dim=input_dim,
            lr=params.get("lr", 1e-3),
            epochs=params.get("epochs", 500),
            patience=params.get("patience", 50),
            min_delta=params.get("min_delta", 1e-4)
        )
    elif model_type == "cnn":
        return SurrogateCNNTrainer(
            input_dim=input_dim,
            lr=params.get("lr", 1e-3),
            epochs=params.get("epochs", 200),
            patience=params.get("patience", 20),
            min_delta=params.get("min_delta", 1e-4)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def build_grid_filename(out_dir, config, grid_size=None):
    """
    Construct a filename for grid results that encodes key config attributes,
    excluding acquisition method.
    """
    bo_cfg = config.get("bayesopt", {})
    mod_cfg = config.get("model", {})
    mod_cfg_params = mod_cfg.get("params", {})
    
    stride = config["bayesopt"]["downsample_stride"]
    filter_mode = config["bayesopt"]["filter_mode"]
    opt_dir = bo_cfg.get("optimization_direction", "max")
    objtve_mode = mod_cfg.get("objective_mode", "raw")
    lr = mod_cfg_params.get("lr", 1e-3)
    epochs = mod_cfg_params.get("epochs", 200)

    filename = (
        f"df_grid_{grid_size}_stride{stride}_"
        f"{filter_mode}_{opt_dir}_{objtve_mode}_"
        f"lr{lr}_ep{epochs}.csv"
    )
    return Path(out_dir) / filename
    
def main():
    parser = argparse.ArgumentParser(description="Run generic function pipeline using a YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--data-module", help="Optional override module path for data handler.")
    parser.add_argument("--data-class", help="Optional override class name for data handler.")
    parser.add_argument("--out-dir", help="Optional output directory (overrides config).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # Apply display settings from config (insert after cfg is loaded)
    display_cfg = cfg.get("display", {})

    # Matplotlib settings
    #mpl_cfg = display_cfg.get("matplotlib", {})
    #if mpl_cfg:
    #    import matplotlib as mpl
    #    mpl.rcParams['figure.figsize'] = tuple(mpl_cfg.get("figsize", mpl.rcParams.get('figure.figsize')))
    #    mpl.rcParams['figure.dpi'] = mpl_cfg.get("dpi", mpl.rcParams.get('figure.dpi'))
    #    mpl.rcParams['font.size'] = mpl_cfg.get("font_size", mpl.rcParams.get('font.size'))

    # pandas display settings
    #pd_cfg = display_cfg.get("pandas", {})
    #if pd_cfg:
    #    import pandas as pd
    #    if "display_max_rows" in pd_cfg:
    #        pd.set_option('display.max_rows', pd_cfg["display_max_rows"])
    #    if "display_max_colwidth" in pd_cfg:
    #        pd.set_option('display.max_colwidth', pd_cfg["display_max_colwidth"])
    #    if "display_width" in pd_cfg:
    #        pd.set_option('display.width', pd_cfg["display_width"])
    #    if "display_expand_frame_repr" in pd_cfg:
    #        pd.set_option('display.expand_frame_repr', pd_cfg["display_expand_frame_repr"])

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 0)                  # Use unlimited width (auto-detect terminal size)
    pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping to multiple lines

    # allow CLI overrides
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.data_module:
        cfg["data_module"] = args.data_module
    if args.data_class:
        cfg["data_class"] = args.data_class

    # Validate required config entries
    if "data_module" not in cfg or "data_class" not in cfg:
        raise ValueError("Config must include 'data_module' and 'data_class' (or pass via CLI).")

    print("Instantiating data handler:", cfg["data_module"], cfg["data_class"])
    dh = instantiate_data_handler(cfg["data_module"], cfg["data_class"])

    # Optional: call add_weekly_updates if present and enabled in config
    if cfg.get("apply_weekly_updates", True) and hasattr(dh, "add_weekly_updates"):
        print("Applying weekly updates via data handler...")
        dh.add_weekly_updates()

    run_pipeline(cfg, dh)


if __name__ == "__main__":
    main()