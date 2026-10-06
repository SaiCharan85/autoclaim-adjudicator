"""Kaggle kernel entry point: export a finished LoRA adapter to GGUF (no training).

The private dataset holds the adapter files flattened as `<task>_lora__<file>` (Kaggle datasets
have no folders); this rebuilds `/kaggle/working/<task>_lora/` and runs
`finetune_qlora.py --export-only`. Output: /kaggle/working/out/<task>/gguf_gguf/*.gguf + Modelfile.
"""

import shutil
import subprocess
import sys
from pathlib import Path

TASK = "intake"
src = next(Path("/kaggle/input").rglob("finetune_qlora.py")).parent
work = Path("/kaggle/working")
lora = work / f"{TASK}_lora"
lora.mkdir(parents=True, exist_ok=True)
for f in src.glob(f"{TASK}_lora__*"):
    shutil.copy2(f, lora / f.name.split("__", 1)[1])
if not (lora / "adapter_config.json").exists():
    raise SystemExit(f"no {TASK} adapter in the dataset")
pip = [sys.executable, "-m", "pip", "install", "-q", "-r", str(src / "requirements-finetune.txt")]
subprocess.run(pip, check=True)
cmd = [sys.executable, str(src / "finetune_qlora.py"), "--data", str(work), "--tasks", TASK,
       "--export-only"]  # fmt: skip
subprocess.run(cmd, check=True)
