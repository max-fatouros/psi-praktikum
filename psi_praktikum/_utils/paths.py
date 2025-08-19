from pathlib import Path

path = Path(__file__)
while not Path.is_dir(path / 'psi_praktikum' ):
    path = path.parent

PROJECT_ROOT = path

UTIL_DIR = Path(__file__).parent

LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)


DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)