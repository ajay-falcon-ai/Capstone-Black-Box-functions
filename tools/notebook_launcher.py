# tools/notebook_launcher.py
import sys
import subprocess
import shlex
from pathlib import Path
import yaml
import tempfile
import re

DEFAULT_CONFIG_NAMES = ["config.yaml", "config.yml"]
RUNNER_FILENAME = "run_function.py"
MAX_UP = 6

def find_local_config(notebook_dir: Path, names=DEFAULT_CONFIG_NAMES):
    for name in names:
        p = notebook_dir / name
        if p.exists():
            return p
    return None

def infer_data_handler_from_folder(folder_name: str):
    """
    Infer module/class names from folder name.
    Examples:
      function_1 -> Function1DataHandler
      function-2 -> Function2DataHandler
      function1  -> Function1DataHandler
      my_func   -> MyFuncDataHandler
    """
    m = re.search(r'function[_-]?(\d+)', folder_name, flags=re.IGNORECASE)
    if m:
        idx = m.group(1)
        cls = f"Function{idx}DataHandler"
        module = cls
    else:
        parts = re.split(r'[_\-\s]+', folder_name)
        title = ''.join(p.capitalize() for p in parts if p)
        cls = f"{title}DataHandler"
        module = cls
    return module, cls

def merge_and_write_temp_config(original_config_path: Path, inferred_module: str, inferred_class: str):
    """
    Read the local YAML, set data_module/data_class if missing, write merged config to a temp file,
    and return the temp file Path.
    """
    with open(original_config_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("data_module", inferred_module)
    cfg.setdefault("data_class", inferred_class)
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    tmp_path = Path(tmp.name)
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp_path

def find_runner(start_dir: Path, runner_filename=RUNNER_FILENAME, max_up=MAX_UP):
    p = start_dir
    for _ in range(max_up + 1):
        candidate = p / runner_filename
        if candidate.exists():
            return candidate
        p = p.parent
    return None

def run_runner_with_config(runner_path: Path, config_path: Path, python_exe: str = sys.executable, extra_args=None):
    cmd = [python_exe, str(runner_path), "--config", str(config_path)]
    if extra_args:
        cmd += list(extra_args)
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode

def launch_from_notebook(notebook_dir: str = None, config_names=None, runner_filename=None, max_up=None):
    """
    High-level helper to be called from a tiny notebook cell.
    Returns (return_code, tmp_config_path, runner_path).
    """
    notebook_dir = Path(notebook_dir or Path.cwd())
    config_names = config_names or DEFAULT_CONFIG_NAMES
    runner_filename = runner_filename or RUNNER_FILENAME
    max_up = max_up if max_up is not None else MAX_UP

    config_path = find_local_config(notebook_dir, config_names)
    if config_path is None:
        raise FileNotFoundError(f"No config file found in {notebook_dir}. Create one of: {config_names}")

    inferred_module, inferred_class = infer_data_handler_from_folder(notebook_dir.name)
    tmp_config = merge_and_write_temp_config(config_path, inferred_module, inferred_class)

    runner_path = find_runner(notebook_dir, runner_filename, max_up)
    if runner_path is None:
        raise FileNotFoundError(f"Could not find {runner_filename} within {max_up} levels above {notebook_dir}")

    rc = run_runner_with_config(runner_path, tmp_config)
    return rc, tmp_config, runner_path