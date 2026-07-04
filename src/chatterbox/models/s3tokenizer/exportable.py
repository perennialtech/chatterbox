import torch


class S3TokenizerQuantizerExport(torch.nn.Module):
    def __init__(self, tokenizer: torch.nn.Module):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, log_mel: torch.Tensor, mel_lengths: torch.Tensor):
        hidden, speech_token_lengths = self.tokenizer.encoder(log_mel, mel_lengths)
        speech_tokens = self.tokenizer.quantizer.encode(hidden)
        return speech_tokens.long(), speech_token_lengths.long()
