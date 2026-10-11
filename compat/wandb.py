"""No-op W&B shim used only when configs set wandb.enabled to false."""

run = None


class config:
    @staticmethod
    def update(*_args, **_kwargs):
        return None


def init(*_args, **_kwargs):
    return None


def log(*_args, **_kwargs):
    return None


def finish(*_args, **_kwargs):
    return None
