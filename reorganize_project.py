import os
import shutil
from pathlib import Path

# Project root
project_root = Path(__file__).parent
llp_dir = project_root / 'llp'

# Files to move to llp directory
files_to_move = [
    'cifar_core.py',
    'cifar_meta_model.py',
    'cifar_reporting.py',
    'cifar_training.py',
    'unified_cifar_training.py',
    'app.py',
    'server.py',
    'simple_docs.py'
]

# Create __init__.py if it doesn't exist
init_file = llp_dir / '__init__.py'
if not init_file.exists():
    with open(init_file, 'w') as f:
        f.write('')

# Move files to llp directory
for file in files_to_move:
    src = project_root / file
    if src.exists():
        dst = llp_dir / file
        print(f"Moving {src} to {dst}")
        shutil.move(str(src), str(dst))

print("\nProject reorganization complete!")
