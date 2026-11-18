"""Execute this package's Python notebook cells with an embedded IPython shell.

This avoids requiring a running Jupyter server. Rich display outputs and actual
cell errors are retained in the notebook. Only run notebooks you trust.
"""
import argparse
import os
from pathlib import Path
import time
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

def execute(path):
    path=Path(path).resolve();nb=nbformat.read(path,as_version=4)
    os.chdir(path.parent)
    shell=InteractiveShell.instance();count=0
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        count+=1;before=time.perf_counter()
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result=shell.run_cell(cell.source,store_history=True)
        outputs=[]
        if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for rich in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata or {}))
        cell.outputs=outputs;cell.execution_count=count
        cell.metadata['execution_seconds']=round(time.perf_counter()-before,3)
        nbformat.write(nb,path)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Notebook cell {count} failed: {result.error_before_exec or result.error_in_exec}')
        print(f'Executed cell {count} ({cell.metadata["execution_seconds"]} s)',flush=True)
    nb.metadata['pathsnap_execution']='Every code cell executed in order with embedded IPython; no fabricated outputs.'
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nbformat.write(nb,path)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('notebook');execute(a.parse_args().notebook)
