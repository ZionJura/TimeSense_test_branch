"""TimeSense model extensions for Qwen2.5 and ChatTS."""

__all__ = ["Qwen2TSConfig", "Qwen2TSProcessor", "sp_encoding"]


def __getattr__(name: str):
    """Avoid importing the heavyweight model stack for CLI help and data tools."""
    if name == "Qwen2TSConfig":
        from .configuration_qwen2 import Qwen2TSConfig

        return Qwen2TSConfig
    if name in {"Qwen2TSProcessor", "sp_encoding"}:
        from .processing_qwen2_ts import Qwen2TSProcessor, sp_encoding

        return {"Qwen2TSProcessor": Qwen2TSProcessor, "sp_encoding": sp_encoding}[name]
    raise AttributeError(name)
