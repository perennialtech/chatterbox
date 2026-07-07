from ...audio import S3GEN_SR
from .pipeline import S3Token2Mel, S3Token2Wav

S3Gen = S3Token2Wav

__all__ = ["S3GEN_SR", "S3Token2Mel", "S3Token2Wav", "S3Gen"]
