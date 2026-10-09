"""Kaggle kernel entry point (pushed as a GPU script kernel; runs as a background batch job).

Installs the pinned fine-tuning environment, then runs finetune_qlora.py on the attached private
dataset. Results (LoRA adapters, GGUF + Modelfile, run_log.json) land in /kaggle/working/out and
become the kernel's output, downloadable with `kaggle kernels output`.
"""

import json
import subprocess
import sys
from pathlib import Path

src = next(Path("/kaggle/input").rglob("finetune_qlora.py")).parent
pip = [sys.executable, "-m", "pip", "install", "-q", "-r", str(src / "requirements-finetune.txt")]
subprocess.run(pip, check=True)
config = src / "run_config.json"  # which tasks this dataset version trains (default: all its files)
tasks = json.loads(config.read_text())["tasks"] if config.exists() else []
cmd = [sys.executable, str(src / "finetune_qlora.py"), "--data", str(src)]
subprocess.run(cmd + (["--tasks", *tasks] if tasks else []), check=True)
