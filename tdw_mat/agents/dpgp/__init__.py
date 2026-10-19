"""TDW DPGP method; configuration imports do not load the agent runtime."""

__all__ = ["DPGP"]


def __getattr__(name):
    if name == "DPGP":
        from .agent import DPGP

        globals()[name] = DPGP
        return DPGP
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
