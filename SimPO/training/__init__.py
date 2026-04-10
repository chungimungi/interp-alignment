from .app import app
from . import train_simpo  # noqa: F401 — registers @app.function and @app.local_entrypoint
from . import push          # noqa: F401 — registers push_from_volume

__all__ = ["app"]
