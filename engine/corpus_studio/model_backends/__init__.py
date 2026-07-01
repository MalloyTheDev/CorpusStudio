"""Model backend skeletons for future local-first model execution."""

from corpus_studio.model_backends.base import BackendHealthReport
from corpus_studio.model_backends.base import BackendGenerateRequest, BackendGenerateResponse
from corpus_studio.model_backends.base import BackendModelListReport
from corpus_studio.model_backends.base import ModelBackendConfig

__all__ = [
    "BackendGenerateRequest",
    "BackendGenerateResponse",
    "BackendHealthReport",
    "BackendModelListReport",
    "ModelBackendConfig",
]
