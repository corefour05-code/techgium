"""Execute notebook cells programmatically and save outputs."""

import os
from pathlib import Path

# Set cache dirs to local writable paths
os.environ["MPLCONFIGDIR"] = str(Path("data/interim/.matplotlib").resolve())
os.environ["IPYTHONDIR"] = str(Path("data/interim/.ipython").resolve())

import nbformat
from nbclient import NotebookClient

notebook_path = Path("notebooks/01_dataset_exploration.ipynb")

print(f"Reading {notebook_path}...")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="infusion-pump-ml")
print("Executing notebook cells...")
client.execute()

print("Saving executed notebook...")
with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook execution completed successfully.")

