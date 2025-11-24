import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from leadlag.kaggle.notebook_entrypoint import *  # noqa: F401,F403


if __name__ == "__main__":
    from leadlag.kaggle.notebook_entrypoint import main

    main()
