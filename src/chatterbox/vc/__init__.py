from .backends import OnnxVCBackend, TensorRTVCBackend, TorchVCBackend
from .conditioning import VoiceConditionTensors
from .postprocess import apply_initial_trim_fade, trim_fade
from .service import ChatterboxVC
from .types import VCBackend, VCResult

__all__ = [
    "ChatterboxVC",
    "OnnxVCBackend",
    "TensorRTVCBackend",
    "TorchVCBackend",
    "VCBackend",
    "VCResult",
    "VoiceConditionTensors",
    "apply_initial_trim_fade",
    "trim_fade",
]
