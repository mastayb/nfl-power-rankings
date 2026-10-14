"""Re-execute the docs notebooks in place and export the public web pages.

Run: python scripts/refresh_methodology.py            (methodology only: the weekly refresh)
     python scripts/refresh_methodology.py xs-and-os  (or any page names; "all" for every page)

Writes, per page:
  docs/<page>.ipynb   notebook with fresh outputs (code + results)
  docs/<page>.html    the same, code cells hidden: the page linked from posts,
                      served by GitHub Pages from /docs on main
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

DOCS = Path(__file__).resolve().parent.parent / "docs"
PAGES = {
    "methodology": "Market + Résumé Power Ratings: methodology",
    "xs-and-os": "Xs and Os: what the data can and can't see",
}


def refresh(page: str) -> None:
    path = DOCS / f"{page}.ipynb"
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="nfl-market-ratings",
                   resources={"metadata": {"path": str(DOCS)}}).execute()
    nbformat.write(nb, path)
    exporter = HTMLExporter(exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True)
    html, _ = exporter.from_notebook_node(nb)
    html = html.replace("<title>Notebook</title>", f"<title>{PAGES[page]}</title>")
    (DOCS / f"{page}.html").write_text(html, encoding="utf-8")
    print(f"refreshed {path.name} and wrote {page}.html")


if __name__ == "__main__":
    args = sys.argv[1:] or ["methodology"]
    for page in (list(PAGES) if args == ["all"] else args):
        refresh(page)
