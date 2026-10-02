"""Settings read from environment variables. The defaults suit local development."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT / "artifacts" / "dart_v1.0"


DTYPES = ("float32", "bfloat16", "float16")


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    device: str | None
    dtype: str | None  # None: float32 on CPU, the exported dtype (float16) on a GPU
    mongodb_uri: str
    mongodb_db: str
    cors_origins: list
    rate_limit_per_minute: int


def load_settings(env=None):
    env = os.environ if env is None else env
    origins = env.get("CORS_ORIGINS", "http://localhost:5173")
    dtype = env.get("DART_DTYPE") or None
    if dtype is not None and dtype not in DTYPES:
        raise ValueError(f"DART_DTYPE must be one of {', '.join(DTYPES)}, got {dtype!r}")
    return Settings(
        model_dir=Path(env.get("DART_MODEL_DIR", str(DEFAULT_MODEL_DIR))),
        device=env.get("DART_DEVICE") or None,
        dtype=dtype,
        mongodb_uri=env.get("MONGODB_URI", "mongodb://localhost:27017"),
        mongodb_db=env.get("MONGODB_DB", "dart"),
        cors_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
        rate_limit_per_minute=int(env.get("RATE_LIMIT_PER_MINUTE", "60")),
    )
