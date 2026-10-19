"""Local, read-only comparison of LoRA/DoRA adapters and saved merged weights."""

from .audit import audit
from .storage import AuditError

__all__ = ["AuditError", "audit"]
__version__ = "0.1.0"
