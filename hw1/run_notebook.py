"""Execute experiments.ipynb from its directory and save every completed cell."""
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parent
os.chdir(root)
path=root/'experiments.ipynb'
notebook=nbformat.read(path,as_version=4)
client=NotebookClient(notebook,timeout=7200,kernel_name='python3',resources={'metadata':{'path':str(root)}})
try:
    client.execute()
finally:
    nbformat.write(notebook,path)
print('Executed:',path)
