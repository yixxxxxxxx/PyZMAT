import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from scientific_support import molecule

spec=importlib.util.spec_from_file_location('compare_hessians',Path(__file__).resolve().parents[1]/'scripts/compare_hessians.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


@pytest.mark.parametrize('corrupt',[False,True])
def test_cli_numerical_failure_exit(zmat_data,tmp_path,monkeypatch,corrupt):
    m=molecule(zmat_data)
    source=tmp_path/'input.com';m.save_gaussian_com(source)
    monkeypatch.setattr(cli.ZMatrix,'load_from_gaussian_input',lambda path:m)
    monkeypatch.setattr(m,'attach_calculator',lambda *args,**kwargs:None)
    if corrupt: monkeypatch.setattr(m,'get_full_fd_hessian',lambda *args:np.zeros((9,9)))
    monkeypatch.setattr(sys,'argv',['compare',str(source),'--output-dir',str(tmp_path/'output')])
    assert cli.main()==int(corrupt)
    import json
    report=json.loads((tmp_path/'output/metrics.json').read_text())
    assert report['passed'] is not corrupt
    assert report['input_sha256']
