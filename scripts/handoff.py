"""Offline developer handoff. Standard library only; no inference or uploads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import zipfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):h.update(block)
    return h.hexdigest()


def safe(root,name):
    if not isinstance(name,str) or '\\' in name or ':' in name or name.startswith('/') or '..' in name.split('/'):
        raise ValueError(f'Unsafe relative path: {name}')
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError(f'Path leaves project: {name}')
    return path


def read(path):return json.loads(path.read_text(encoding='utf8'))


def inventory(root):
    files={};errors=[]
    def add(name,expected=None):
        path=safe(root,name)
        if not path.is_file():errors.append(f'Missing: {name}');return
        sha=digest(path)
        if expected and sha!=expected:errors.append(f'SHA256 mismatch: {name}')
        files[name]={'bytes':path.stat().st_size,'sha256':sha}
    stamp=root/'artifacts/mvp/server/snapshot.json'
    if not stamp.is_file():raise ValueError('Missing artifacts/mvp/server/snapshot.json: restore the private demo data pack first')
    spec=read(stamp)
    if spec.get('closed_snapshot') is not True:raise ValueError('Snapshot is not a published closed database')
    add('artifacts/mvp/server/snapshot.json');add('artifacts/mvp/server/snapshot.sqlite3',spec['sha256'])
    snapshot=root/'artifacts/mvp/server/snapshot.sqlite3'
    if errors:raise ValueError('\n'.join(errors))
    if snapshot.stat().st_size!=spec['bytes']:raise ValueError('Snapshot size mismatch')
    with sqlite3.connect(snapshot.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Snapshot SQLite integrity check failed')
        cameras=[dict(r) for r in db.execute('SELECT id,source_name,source_sha256,frame_count,fps FROM camera ORDER BY id')]
        if len(cameras)!=5:raise ValueError('Expected five cameras in the snapshot')
        source=json.loads(db.execute("SELECT value FROM meta WHERE key='source_run_id'").fetchone()[0])
        summary=json.loads(db.execute("SELECT value FROM meta WHERE key='mtmc_summary'").fetchone()[0])
        for c in cameras:add('data/raw/'+c['source_name'],c['source_sha256'])
        for row in db.execute('SELECT path,sha256 FROM media_segment ORDER BY path'):
            name='artifacts/mvp/media/'+row['path'];add(name,row['sha256'])
            side=safe(root,name).with_suffix('.frames.jsonl')
            if side.is_file():add(side.relative_to(root).as_posix())
    add('data/metadata/office_camera.png')
    tracks=root/'artifacts/mvp/local_tracks/manifest.json';add('artifacts/mvp/local_tracks/manifest.json')
    if tracks.is_file():
        tm=read(tracks)
        if tm['source_run_id']!=source:errors.append('Local tracks refer to a different source run')
        for item in tm['files']:add('artifacts/mvp/local_tracks/'+item['path'],item['sha256'])
    alignment=root/'artifacts/sync/time_alignment.json';add('artifacts/sync/time_alignment.json')
    if alignment.is_file():
        for item in read(alignment)['cameras']:
            add('artifacts/sync/'+item['local_frame_times_file'],item['local_frame_times_sha256'])
    if errors:raise ValueError('\n'.join(errors))
    return {'schema_version':1,'package_kind':'shadowtrack_private_demo_data','private':True,
            'source_run_id':source,'mtmc_revision':summary['revision'],'files':dict(sorted(files.items())),
            'total_bytes':sum(v['bytes'] for v in files.values()),'frames':sum(c['frame_count'] for c in cameras),
            'cameras':[{k:c[k] for k in ('id','source_name','frame_count','fps')} for c in cameras],
            'weights_required_for_demo':False,'contains_source_recordings':True,
            'scope':'Ready demo and cached MTMC recompute; original YOLO chunks and feature-regeneration caches are separate optional development inputs'}


def validate_manifest(manifest):
    if manifest.get('schema_version')!=1 or manifest.get('package_kind')!='shadowtrack_private_demo_data':
        raise ValueError('Unsupported demo data manifest')
    for name,entry in manifest['files'].items():
        if not name.startswith(('data/','artifacts/')):raise ValueError('Demo pack can only contain data/ and artifacts/')
        if entry['bytes']<0 or len(entry['sha256'])!=64:raise ValueError('Invalid file metadata')


def pack(root,target):
    manifest=inventory(root)
    if target.exists():raise ValueError('Output already exists; choose a new filename')
    target.parent.mkdir(parents=True,exist_ok=True);pending=target.with_suffix(target.suffix+'.pending')
    if pending.exists():raise ValueError('Pending archive exists; choose a new filename')
    try:
        with zipfile.ZipFile(pending,'w',zipfile.ZIP_STORED,allowZip64=True) as archive:
            for index,(name,entry) in enumerate(manifest['files'].items(),1):
                path=safe(root,name)
                if digest(path)!=entry['sha256']:raise ValueError(f'Input changed during packaging: {name}')
                archive.write(path,name)
                print(f'[{index}/{len(manifest["files"])}] {name}',flush=True)
            archive.writestr('shadowtrack_data_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        os.replace(pending,target)
    finally:pending.unlink(missing_ok=True)
    target.with_suffix(target.suffix+'.sha256').write_text(digest(target)+'  '+target.name+'\n',encoding='ascii')
    return manifest


def restore(root,source):
    with zipfile.ZipFile(source) as archive:
        manifest=json.loads(archive.read('shadowtrack_data_manifest.json'));validate_manifest(manifest)
        # Reject all unsafe paths and conflicting existing files before writing.
        for name,entry in manifest['files'].items():
            target=safe(root,name);info=archive.getinfo(name)
            if info.file_size!=entry['bytes'] or (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Invalid archive member: '+name)
            if target.exists() and (not target.is_file() or digest(target)!=entry['sha256']):raise ValueError('Existing file differs; use a fresh project directory: '+name)
        for index,(name,entry) in enumerate(manifest['files'].items(),1):
            target=safe(root,name)
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True);pending=target.with_name(target.name+'.handoff-pending')
                try:
                    with archive.open(name) as src,pending.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
                    if digest(pending)!=entry['sha256']:raise ValueError('Corrupt archive member: '+name)
                    os.replace(pending,target)
                finally:pending.unlink(missing_ok=True)
            print(f'[{index}/{len(manifest["files"])}] restored/verified {name}',flush=True)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path.cwd())
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('check');m=commands.add_parser('manifest');m.add_argument('--output',type=Path,default=Path('artifacts/handoff/demo_manifest.json'))
    p=commands.add_parser('pack');p.add_argument('--output',type=Path,required=True)
    r=commands.add_parser('restore');r.add_argument('--source',type=Path,required=True)
    commands.add_parser('init-local')
    args=parser.parse_args(argv);root=args.root.resolve()
    try:
        if args.command=='restore':result=restore(root,args.source.resolve())
        elif args.command=='pack':result=pack(root,args.output.resolve())
        elif args.command=='init-local':
            inventory(root);target=root/'artifacts/mvp/shadowtrack.sqlite3'
            if target.exists():raise ValueError('Host database already exists; it was left unchanged')
            with sqlite3.connect((root/'artifacts/mvp/server/snapshot.sqlite3').as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(target) as dst:src.backup(dst)
            print('Created host database from the closed snapshot');return 0
        else:
            result=inventory(root)
            for name in ('src/shadowtrack/mvp/mtmc.py','src/shadowtrack/mvp/accounting.py','src/shadowtrack/mvp/snapshot.py','configs/mtmc.json','configs/scene_layout.json','frontend/src/tracking.tsx','docker-compose.yml'):
                if not (root/name).is_file():raise ValueError('Missing source file (check uncommitted Git files): '+name)
            if args.command=='manifest':
                target=root/args.output;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps({'status':'READY','files':len(result['files']),'bytes':result['total_bytes'],'frames':result['frames'],'source_run_id':result['source_run_id']},ensure_ascii=True));return 0
    except (ValueError,OSError,KeyError,sqlite3.DatabaseError,zipfile.BadZipFile) as exc:
        print('HANDOFF ERROR: '+str(exc),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
