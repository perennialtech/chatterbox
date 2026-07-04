import torch


class S3TokenizerQuantizerExport(torch.nn.Module):
    def __init__(self, tokenizer: torch.nn.Module):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, log_mel: torch.Tensor, mel_lengths: torch.Tensor):
        speech_tokens, speech_token_lengths = self.tokenizer.quantize(
            log_mel, mel_lengths
        )
        return speech_tokens.long(), speech_token_lengths.long()
