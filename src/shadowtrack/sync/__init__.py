"""Evidence-constrained local and inter-camera time alignment."""

__all__ = ["TimeAlignment", "map_time"]


def __getattr__(name):
    if name in __all__:
        from . import api
        return getattr(api, name)
    raise AttributeError(name)
