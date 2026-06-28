import sys
from pathlib import Path

# Allow `pytest` to import the package from src/ without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
