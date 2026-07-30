import sys
from pathlib import Path

SUBMISSION_SRC = Path(__file__).resolve().parents[1] / "submission_src"
if str(SUBMISSION_SRC) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_SRC))
