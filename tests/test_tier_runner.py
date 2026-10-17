"""Contract validation must catch corruption before launching scientific work."""
import importlib.util
import hashlib
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('tier_runner',Path(__file__).resolve().parents[1]/'tools/run_tiers.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_repository_contracts():
    cases=runner.cases()
    assert {c['level'] for c in cases}=={'L0','L1','L2','L3'}


@pytest.mark.parametrize('corruption',['checksum','registry','budget'])
def test_contract_rejects_corruption(tmp_path,monkeypatch,corruption):
    case=json.loads((runner.ROOT/'tests/cases'/runner.cases()[0]['test_id']/'case.json').read_text())
    case['level']='L1';case['limits'].update(wall_s=60,cpu_s=45,threads=1)
    folder=tmp_path/'tests/case';(folder/'result').mkdir(parents=True)
    (folder/'result/reference.json').write_text('{}')
    fixture=tmp_path/'input.txt';fixture.write_text('frozen')
    (folder/'SHA256SUMS').write_text(hashlib.sha256(fixture.read_bytes()).hexdigest()+'  input.txt\n')
    row={k:case[k] for k in ('test_id','level','package','status')};row['path']='tests/case'
    if corruption=='checksum':fixture.write_text('changed')
    if corruption=='registry':row['level']='L0'
    if corruption=='budget':case['limits']['wall_s']=61
    (folder/'case.json').write_text(json.dumps(case))
    (tmp_path/'tests/meta.json').write_text(json.dumps({'cases':[row]}))
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    with pytest.raises(ValueError):runner.cases()
