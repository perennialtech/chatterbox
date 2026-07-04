import torch

from ...audio import S3GEN_SR, MelSpectrogram
from ..dynamic_axes import REFERENCE_MEL_DYNAMIC_AXES

input_names = ["wav_24k"]
output_names = ["prompt_feat", "prompt_feat_len"]
dynamic_axes = REFERENCE_MEL_DYNAMIC_AXES


class ReferenceMel24kExport(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = MelSpectrogram(sampling_rate=S3GEN_SR)

    def forward(self, wav_24k):
        prompt_feat = self.mel(wav_24k).transpose(1, 2).contiguous()
        prompt_feat_len = torch.full(
            (prompt_feat.size(0),),
            prompt_feat.size(1),
            dtype=torch.long,
            device=prompt_feat.device,
        )
        return prompt_feat, prompt_feat_len


def make_dummy_inputs(batch: int = 1, samples: int = S3GEN_SR):
    return (torch.randn(batch, samples, dtype=torch.float32),)
