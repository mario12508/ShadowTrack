"""Indexed SQLite for this single-host MVP; machine artifacts remain independent."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA='''
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY);
INSERT OR IGNORE INTO schema_version VALUES(1);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS camera(id TEXT PRIMARY KEY,source_name TEXT,source_sha256 TEXT,width INTEGER,height INTEGER,fps REAL,frame_count INTEGER,first_pts REAL,last_pts REAL,duration REAL,sync_status TEXT,sync_json TEXT);
CREATE TABLE IF NOT EXISTS frame(camera_id TEXT,frame_index INTEGER,raw_pts REAL,media_time REAL,segment_id TEXT,global_time REAL,global_uncertainty TEXT,input_empty INTEGER,PRIMARY KEY(camera_id,frame_index));
CREATE INDEX IF NOT EXISTS frame_time ON frame(camera_id,media_time);
CREATE INDEX IF NOT EXISTS frame_global_time ON frame(camera_id,global_time);
CREATE TABLE IF NOT EXISTS detection(camera_id TEXT,frame_index INTEGER,detection_index INTEGER,bbox TEXT,confidence REAL,decision TEXT,reasons TEXT,PRIMARY KEY(camera_id,frame_index,detection_index));
CREATE TABLE IF NOT EXISTS local_track(id TEXT PRIMARY KEY,source_id TEXT UNIQUE,camera_id TEXT,segment_id TEXT,display_id TEXT,start_frame INTEGER,end_frame INTEGER,start_time REAL,end_time REAL,observations INTEGER,mean_confidence REAL,global_episode_id TEXT);
CREATE INDEX IF NOT EXISTS track_camera_time ON local_track(camera_id,start_time,end_time);
CREATE TABLE IF NOT EXISTS observation(track_id TEXT,camera_id TEXT,frame_index INTEGER,media_time REAL,raw_pts REAL,bbox TEXT,confidence REAL,detection_index INTEGER,PRIMARY KEY(track_id,frame_index));
CREATE INDEX IF NOT EXISTS observation_frame ON observation(camera_id,frame_index);
CREATE INDEX IF NOT EXISTS observation_time ON observation(track_id,media_time);
CREATE TABLE IF NOT EXISTS episode(id TEXT PRIMARY KEY,scope TEXT,global_confirmed INTEGER,status TEXT,start_time REAL,last_seen REAL,terminated_at REAL,termination_reason TEXT,confidence REAL,cameras TEXT);
CREATE TABLE IF NOT EXISTS episode_member(episode_id TEXT,track_id TEXT,PRIMARY KEY(episode_id,track_id));
CREATE TABLE IF NOT EXISTS event(id INTEGER PRIMARY KEY,camera_id TEXT,track_id TEXT,kind TEXT,media_time REAL,detail TEXT);
CREATE INDEX IF NOT EXISTS event_time ON event(camera_id,media_time);
CREATE TABLE IF NOT EXISTS calibration(id TEXT PRIMARY KEY,camera_id TEXT,segment_id TEXT,raw_start REAL,raw_end REAL,offset_seconds REAL,drift_ppm REAL,uncertainty_seconds REAL,checked INTEGER,evidence TEXT,updated_at TEXT);
CREATE TABLE IF NOT EXISTS association(id TEXT PRIMARY KEY,track_a TEXT,track_b TEXT,similarity REAL,confidence REAL,status TEXT,reasons TEXT,evidence TEXT);
CREATE TABLE IF NOT EXISTS embedding(track_id TEXT PRIMARY KEY,vector BLOB,dimensions INTEGER,samples INTEGER,model_hash TEXT,evidence TEXT);
CREATE TABLE IF NOT EXISTS job(id TEXT PRIMARY KEY,kind TEXT,status TEXT,progress REAL,detail TEXT,updated_at TEXT);
CREATE TABLE IF NOT EXISTS media_segment(id TEXT PRIMARY KEY,camera_id TEXT,segment_id TEXT,kind TEXT,path TEXT,sha256 TEXT,start_time REAL,end_time REAL,frame_start INTEGER,frame_stop INTEGER,frames INTEGER,width INTEGER,height INTEGER,metadata TEXT);
'''


@contextmanager
def connect(path,readonly=False):
    path=Path(path)
    if readonly:
        connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=30)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        connection=sqlite3.connect(path,timeout=30)
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA busy_timeout=30000')
        connection.executescript(SCHEMA)
    connection.row_factory=sqlite3.Row
    try:
        yield connection
        if not readonly:connection.commit()
    except BaseException:
        if not readonly:connection.rollback()
        raise
    finally:connection.close()


def set_meta(db,key,value):db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(key,json.dumps(value,ensure_ascii=False)))


def meta(db,key,default=None):
    row=db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
    return json.loads(row[0]) if row else default


def rows(db,query,params=()):return [dict(r) for r in db.execute(query,params)]

