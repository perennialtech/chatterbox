import torch

from ..dynamic_axes import S3_TOKENIZER_DYNAMIC_AXES

input_names = ["log_mel", "mel_lengths"]
output_names = ["speech_tokens", "speech_token_lengths"]
dynamic_axes = S3_TOKENIZER_DYNAMIC_AXES


class S3TokenizerQuantizerExport(torch.nn.Module):
    def __init__(self, tokenizer: torch.nn.Module):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, log_mel, mel_lengths):
        speech_tokens, speech_token_lengths = self.tokenizer.quantize(
            log_mel, mel_lengths
        )
        return speech_tokens.long(), speech_token_lengths.long()


def make_dummy_inputs(batch: int = 1, mel_frames: int = 128, n_mels: int = 128):
    return (
        torch.randn(batch, n_mels, mel_frames, dtype=torch.float32),
        torch.full((batch,), mel_frames, dtype=torch.long),
    )
