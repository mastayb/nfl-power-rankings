"""Re-execute docs/methodology.ipynb in place and export the public web page.

Run: python scripts/refresh_methodology.py

Writes:
  docs/methodology.ipynb   notebook with fresh outputs (code + results)
  docs/methodology.html    the same, code cells hidden: the page linked from posts,
                           served by GitHub Pages from /docs on main
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

DOCS = Path(__file__).resolve().parent.parent / "docs"
NB = DOCS / "methodology.ipynb"

nb = nbformat.read(NB, as_version=4)
NotebookClient(nb, timeout=900, kernel_name="nfl-market-ratings",
               resources={"metadata": {"path": str(DOCS)}}).execute()
nbformat.write(nb, NB)

exporter = HTMLExporter(exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True)
html, _ = exporter.from_notebook_node(nb)
html = html.replace("<title>Notebook</title>", "<title>Market + Résumé Power Ratings: methodology</title>")
(DOCS / "methodology.html").write_text(html, encoding="utf-8")
print(f"refreshed {NB.name} and wrote methodology.html")
