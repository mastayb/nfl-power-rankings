"""Re-execute docs/methodology.ipynb in place against the latest outputs.

Run: python scripts/refresh_methodology.py
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path(__file__).resolve().parent.parent / "docs" / "methodology.ipynb"

nb = nbformat.read(NB, as_version=4)
NotebookClient(nb, timeout=900, kernel_name="nfl-market-ratings",
               resources={"metadata": {"path": str(NB.parent)}}).execute()
nbformat.write(nb, NB)
print(f"refreshed {NB}")
