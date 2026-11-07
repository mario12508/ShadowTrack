"""Episode timeouts describe absence in an observable clock domain, never identity."""
from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeState:
    status: str
    terminated_at: float | None
    reason: str


def lifecycle(last_seen,now,timeout=120.,coverage_complete=True):
    if now < last_seen:raise ValueError('Observation clock cannot move backwards')
    if not 60 <= timeout <= 120:raise ValueError('Timeout must be 60–120 seconds')
    absent=now-last_seen
    if absent <= .2:return EpisodeState('active',None,'observed')
    if absent >= timeout and coverage_complete:return EpisodeState('terminated',last_seen+timeout,'inactivity_timeout')
    return EpisodeState('lost',None,'coverage_unknown' if not coverage_complete else 'within_inactivity_timeout')


def may_continue(last_seen,next_seen,timeout=120.):
    return 0 <= next_seen-last_seen < timeout

