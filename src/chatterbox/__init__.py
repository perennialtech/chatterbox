try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

from .vc import VoiceConverter

__version__ = version("chatterbox-tts")
__all__ = ["VoiceConverter"]
