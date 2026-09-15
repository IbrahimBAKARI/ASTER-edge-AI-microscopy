"""Reproducible WBC localisation and patient/session-level AML MIL inference."""

__all__ = ["LeukemiaPipeline"]


def __getattr__(name):
    if name == "LeukemiaPipeline":
        from .pipeline import LeukemiaPipeline
        return LeukemiaPipeline
    raise AttributeError(name)
