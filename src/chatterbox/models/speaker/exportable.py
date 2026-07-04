import torch


class SpeakerEncoderExport(torch.nn.Module):
    def __init__(self, speaker_encoder: torch.nn.Module):
        super().__init__()
        self.speaker_encoder = speaker_encoder

    def forward(self, fbank: torch.Tensor):
        return self.speaker_encoder(fbank)
