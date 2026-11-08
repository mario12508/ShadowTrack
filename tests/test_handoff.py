"""A data handoff never overwrites existing inputs or trusts ZIP paths."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec=importlib.util.spec_from_file_location('handoff',Path(__file__).parents[1]/'scripts/handoff.py')
handoff=importlib.util.module_from_spec(spec);spec.loader.exec_module(handoff)


def archive_at(path,name='data/raw/camera_new_1.mkv',content=b'private test bytes',expected=None):
    expected=content if expected is None else expected
    manifest={'schema_version':1,'package_kind':'shadowtrack_private_demo_data',
              'files':{name:{'bytes':len(content),'sha256':hashlib.sha256(expected).hexdigest()}}}
    with zipfile.ZipFile(path,'w') as z:
        z.writestr(name,content);z.writestr('shadowtrack_data_manifest.json',json.dumps(manifest))


def test_restore_reuses_identical_input_without_overwriting(tmp_path):
    package=tmp_path/'data.zip';archive_at(package);project=tmp_path/'project';project.mkdir()
    handoff.restore(project,package);target=project/'data/raw/camera_new_1.mkv';before=target.stat().st_mtime_ns
    handoff.restore(project,package)
    assert target.read_bytes()==b'private test bytes' and target.stat().st_mtime_ns==before


def test_conflicting_source_is_never_overwritten(tmp_path):
    package=tmp_path/'data.zip';archive_at(package);target=tmp_path/'project/data/raw/camera_new_1.mkv'
    target.parent.mkdir(parents=True);target.write_bytes(b'existing original')
    with pytest.raises(ValueError,match='Existing file differs'):handoff.restore(tmp_path/'project',package)
    assert target.read_bytes()==b'existing original'


@pytest.mark.parametrize('name',['../outside.mkv','data/../../outside.mkv','data/C:/outside.mkv'])
def test_archive_paths_cannot_escape_project(tmp_path,name):
    package=tmp_path/'data.zip';archive_at(package,name=name)
    with pytest.raises(ValueError):handoff.restore(tmp_path/'project',package)
    assert not (tmp_path/'outside.mkv').exists()


def test_corrupt_archive_member_is_not_published(tmp_path):
    package=tmp_path/'data.zip';archive_at(package,expected=b'other bytes')
    with pytest.raises(ValueError,match='Corrupt archive member'):handoff.restore(tmp_path/'project',package)
    assert not (tmp_path/'project/data/raw/camera_new_1.mkv').exists()


def test_missing_data_is_reported_before_launch(tmp_path,capsys):
    assert handoff.main(['--root',str(tmp_path),'check'])==1
    assert 'snapshot.json' in capsys.readouterr().err
