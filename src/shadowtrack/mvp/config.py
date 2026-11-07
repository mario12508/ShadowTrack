"""Portable paths: the application never depends on Windows or Kaggle filenames."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path
    artifacts: Path
    raw: Path
    evidence: Path
    alignment: Path
    episode_timeout: float = 120.

    @classmethod
    def environment(cls):
        root=Path(os.environ.get('SHADOWTRACK_ROOT',Path.cwd())).resolve()
        def path(name,default):return (root/os.environ.get(name,default)).resolve()
        timeout=float(os.environ.get('SHADOWTRACK_EPISODE_TIMEOUT','120'))
        if not 60 <= timeout <= 120:raise ValueError('Episode timeout must be 60–120 seconds')
        return cls(root,path('SHADOWTRACK_ARTIFACTS','artifacts/mvp'),path('SHADOWTRACK_RAW','data/raw'),
                   path('SHADOWTRACK_EVIDENCE','artifacts/detection/input_evidence'),path('SHADOWTRACK_ALIGNMENT','artifacts/sync/time_alignment.json'),timeout)

    @property
    def database(self):return self.artifacts/'shadowtrack.sqlite3'

    @property
    def media(self):return self.artifacts/'media'

