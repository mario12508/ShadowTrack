"""A closed SQLite snapshot crosses the host/container boundary, never a WAL.

Docker Desktop bind mounts are not safe for concurrent Windows/Linux SQLite
locking. The serving database lives on a Linux named volume. Publication is an
explicit, hash-bound file handoff; native API recomputations stay on that volume.
"""
import os
import sqlite3
from contextlib import closing

from ..detection.common import atomic_json,read_json
from ..probe import sha256


def publish_snapshot(settings):
    folder=settings.artifacts/'server';folder.mkdir(exist_ok=True,parents=True)
    target=folder/'snapshot.sqlite3';pending=folder/'snapshot.pending.sqlite3'
    # Explicit path is owned by this publisher, never a computed source recording.
    if pending.exists():pending.unlink()
    with closing(sqlite3.connect(settings.database)) as source,closing(sqlite3.connect(pending)) as dest:
        source.backup(dest);dest.execute('PRAGMA journal_mode=DELETE')
        assert dest.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    os.replace(pending,target)
    result={'schema_version':1,'file':target.name,'sha256':sha256(target),'bytes':target.stat().st_size,'closed_snapshot':True}
    atomic_json(folder/'snapshot.json',result);return result


def bootstrap_native_database(settings):
    if settings.database_override is None:return
    folder=settings.artifacts/'server';manifest=read_json(folder/'snapshot.json');source=folder/'snapshot.sqlite3'
    if manifest.get('closed_snapshot') is not True or source.stat().st_size!=manifest['bytes'] or sha256(source)!=manifest['sha256']:
        raise ValueError('Serving snapshot hash mismatch; publish a complete snapshot first')
    target=settings.database;stamp=target.with_suffix('.snapshot.json')
    if target.exists() and stamp.exists() and read_json(stamp)['sha256']==manifest['sha256']:return
    target.parent.mkdir(exist_ok=True,parents=True)
    # Called before FastAPI accepts requests, with no other process opening this
    # native database. Checkpoint its old WAL before replacing the complete file.
    if target.exists():
        with closing(sqlite3.connect(target)) as db:db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    pending=target.with_suffix('.pending.sqlite3')
    with source.open('rb') as src,pending.open('wb') as dst:
        import shutil
        shutil.copyfileobj(src,dst,1024*1024)
    if sha256(pending)!=manifest['sha256']:raise ValueError('Incomplete native snapshot copy')
    os.replace(pending,target);atomic_json(stamp,manifest)
