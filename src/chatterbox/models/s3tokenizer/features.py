import librosa
import torch

from ...audio.constants import S3_HOP, S3_SR


class S3TokenizerLogMel(torch.nn.Module):
    def __init__(self, n_fft: int = 400, n_mels: int = 128):
        super().__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        mel_filters = librosa.filters.mel(sr=S3_SR, n_fft=n_fft, n_mels=n_mels)
        self.register_buffer("_mel_filters", torch.FloatTensor(mel_filters))
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(dtype=self._mel_filters.dtype, device=self._mel_filters.device)
        stft = torch.stft(
            audio,
            self.n_fft,
            S3_HOP,
            window=self.window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self._mel_filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0
