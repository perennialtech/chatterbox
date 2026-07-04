import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ...audio import S3_SR, S3GEN_SR, mel_spectrogram, resample_audio
from ..s3tokenizer import SPEECH_VOCAB_SIZE, S3Tokenizer
from .conditioning import S3ReferenceCondition
from .configs import CFM_PARAMS
from .decoder import ConditionalDecoder
from .f0_predictor import ConvRNNF0Predictor
from .flow import CausalMaskedDiffWithXvec
from .flow_matching import CausalConditionalCFM
from .hifigan import HiFTGenerator
from .token_encoder import S3TokenEncoder
from .transformer.upsample_encoder import UpsampleConformerEncoder
from .xvector import CAMPPlus

logger = logging.getLogger(__name__)

_TOKEN_LENGTH_BUCKETS = (32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072)


def drop_invalid_tokens(x):
    if not (len(x.shape) <= 2 and x.shape[0] == 1):
        raise ValueError("only batch size of one is supported")
    return x[x < SPEECH_VOCAB_SIZE]


def _bucket_token_length(length: int) -> int:
    for bucket in _TOKEN_LENGTH_BUCKETS:
        if length <= bucket:
            return bucket
    return ((length + 511) // 512) * 512


class S3Token2Mel(torch.nn.Module):
    def __init__(self, meanflow: bool = False):
        super().__init__()
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        self.mel_extractor = mel_spectrogram
        self.speaker_encoder = CAMPPlus(memory_efficient=False)
        self.meanflow = meanflow

        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer="linear",
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )

        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn="gelu",
            meanflow=self.meanflow,
        )
        decoder = CausalConditionalCFM(
            spk_emb_dim=80,
            cfm_params=CFM_PARAMS,
            estimator=estimator,
        )

        self.flow = CausalMaskedDiffWithXvec(encoder=encoder, decoder=decoder)
        self.token_encoder = S3TokenEncoder(
            input_embedding=self.flow.input_embedding,
            encoder=self.flow.encoder,
            encoder_proj=self.flow.encoder_proj,
            vocab_size=self.flow.vocab_size,
        )

    @property
    def device(self):
        return next(self.tokenizer.parameters()).device

    @property
    def dtype(self):
        return next(self.flow.parameters()).dtype

    def prepare_ref_condition(
        self, ref_dict: dict | S3ReferenceCondition
    ) -> S3ReferenceCondition:
        if isinstance(ref_dict, S3ReferenceCondition):
            ref_dict.validate()
            return ref_dict
        return S3ReferenceCondition.from_mapping(
            ref_dict, device=self.device, dtype=self.dtype
        )

    def prepare_ref_dict(self, ref_dict: dict) -> dict:
        return self.prepare_ref_condition(ref_dict).as_dict()

    @torch.inference_mode()
    def embed_ref(
        self,
        ref_wav: torch.Tensor,
        ref_sr: int,
        device="auto",
        ref_fade_out: bool = True,
    ) -> dict:
        device = self.device if device == "auto" else device
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()

        ref_wav = ref_wav.to(device)
        if ref_wav.ndim == 1:
            ref_wav = ref_wav.unsqueeze(0)

        if ref_wav.size(1) > 10 * ref_sr:
            logger.warning("s3gen received reference longer than 10 seconds")

        ref_wav_24 = resample_audio(ref_wav, ref_sr, S3GEN_SR, device)
        ref_wav_24 = ref_wav_24.to(device=device, dtype=self.dtype)

        ref_mels_24 = (
            self.mel_extractor(ref_wav_24).transpose(1, 2).to(dtype=self.dtype)
        )
        ref_mels_24_len = torch.full(
            (ref_mels_24.size(0),),
            ref_mels_24.size(1),
            dtype=torch.long,
            device=device,
        )

        ref_wav_16 = resample_audio(ref_wav, ref_sr, S3_SR, device)
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16.to(dtype=self.dtype))
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16.float())

        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            logger.warning(
                "Reference mel length differs from 2x token length; trimming."
            )
            ref_speech_tokens = ref_speech_tokens[:, : ref_mels_24.shape[1] // 2]
            ref_speech_token_lens = torch.full(
                (ref_speech_tokens.size(0),),
                ref_speech_tokens.size(1),
                dtype=torch.long,
                device=device,
            )

        return self.prepare_ref_dict(
            {
                "prompt_token": ref_speech_tokens.long().to(device),
                "prompt_token_len": ref_speech_token_lens.long().to(device),
                "prompt_feat": ref_mels_24,
                "prompt_feat_len": ref_mels_24_len,
                "embedding": ref_x_vector,
            }
        )

    def forward(
        self,
        speech_tokens: torch.LongTensor,
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        ref_dict: Optional[dict] = None,
        n_cfm_timesteps: Optional[int] = None,
        finalize: bool = False,
        speech_token_lens: Optional[torch.Tensor] = None,
    ):
        if speech_tokens.size(0) != 1:
            raise ValueError("Only batch size 1 is supported")
        if (ref_wav is None) == (ref_dict is None):
            raise ValueError("Provide exactly one of ref_wav or ref_dict")

        if ref_dict is None:
            ref_dict = self.embed_ref(ref_wav, ref_sr)
        else:
            ref_dict = self.prepare_ref_dict(ref_dict)

        speech_tokens = torch.atleast_2d(speech_tokens).to(
            device=self.device, dtype=torch.long
        )
        if speech_token_lens is None:
            speech_token_lens = torch.full(
                (speech_tokens.size(0),),
                speech_tokens.size(-1),
                dtype=torch.long,
                device=self.device,
            )
        else:
            speech_token_lens = speech_token_lens.to(
                device=self.device, dtype=torch.long
            )

        n_cfm_timesteps = n_cfm_timesteps or (2 if self.meanflow else 10)

        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=finalize,
            n_timesteps=n_cfm_timesteps,
            meanflow=self.meanflow,
            **ref_dict,
        )
        return output_mels


class S3Token2Wav(S3Token2Mel):
    ignore_state_dict_missing = ("tokenizer._mel_filters", "tokenizer.window")

    def __init__(self, meanflow: bool = False):
        super().__init__(meanflow)

        f0_predictor = ConvRNNF0Predictor()
        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )

        n_trim = S3GEN_SR // 50
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False)
        self.estimator_dtype = "fp32"

    def compile_for_inference(self) -> "S3Token2Wav":
        import torch._dynamo

        backend = "inductor"
        self.flow.encoder = torch.compile(
            self.flow.encoder,
            mode="default",
            backend=backend,
            dynamic=True,
        )
        self.flow.decoder.estimator = torch.compile(
            self.flow.decoder.estimator,
            mode="default",
            backend=backend,
            dynamic=True,
        )
        self.mel2wav.compile_for_inference()
        return self

    @torch.inference_mode()
    def warmup(
        self,
        ref_dict: Optional[dict] = None,
        total_token_buckets=(384, 512, 768, 1024),
    ) -> None:
        if ref_dict is None:
            return

        ref_dict = self.prepare_ref_dict(ref_dict)
        prompt_len = (
            ref_dict["prompt_token"].size(1) if "prompt_token" in ref_dict else 0
        )

        for total_bucket in total_token_buckets:
            speech_len = max(1, total_bucket - prompt_len)
            speech_tokens = torch.zeros(
                1, speech_len, dtype=torch.long, device=self.device
            )
            output_mels = self.flow_inference(
                speech_tokens=speech_tokens,
                ref_dict=ref_dict,
                n_cfm_timesteps=2 if self.meanflow else 10,
                finalize=True,
            )
            self.hift_inference(output_mels.to(dtype=self.dtype), None)

        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(torch.device(self.device))

    def forward(
        self,
        speech_tokens,
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
        speech_token_lens: Optional[torch.Tensor] = None,
        skip_vocoder: bool = False,
        n_cfm_timesteps: Optional[int] = None,
    ):
        output_mels = super().forward(
            speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            finalize=finalize,
            n_cfm_timesteps=n_cfm_timesteps,
        )

        if skip_vocoder:
            return output_mels

        hift_cache_source = torch.zeros(1, 1, 0, device=self.device, dtype=self.dtype)
        output_wavs, output_sources = self.mel2wav.inference(
            speech_feat=output_mels,
            cache_source=hift_cache_source,
        )

        if not self.training:
            output_wavs[:, : len(self.trim_fade)] *= self.trim_fade

        return output_wavs, output_sources

    def _pad_tokens_to_bucket(
        self, speech_tokens: torch.Tensor, prompt_len: int = 0
    ) -> torch.Tensor:
        bucket_len = _bucket_token_length(speech_tokens.size(-1) + prompt_len)
        pad_len = bucket_len - (speech_tokens.size(-1) + prompt_len)
        if pad_len > 0:
            return F.pad(speech_tokens, (0, pad_len), value=0)
        return speech_tokens

    @torch.inference_mode()
    def flow_inference(
        self,
        speech_tokens,
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        ref_dict: Optional[dict] = None,
        n_cfm_timesteps: Optional[int] = None,
        finalize: bool = False,
        speech_token_lens: Optional[torch.Tensor] = None,
    ):
        if speech_tokens.size(0) != 1:
            raise ValueError("Only batch size 1 is supported")
        speech_tokens = torch.atleast_2d(speech_tokens).to(
            device=self.device, dtype=torch.long
        )

        if speech_token_lens is None:
            speech_token_lens = torch.full(
                (speech_tokens.size(0),),
                speech_tokens.size(-1),
                dtype=torch.long,
                device=self.device,
            )
        else:
            speech_token_lens = speech_token_lens.to(
                device=self.device, dtype=torch.long
            )

        original_mel_len = int(speech_token_lens.max().detach().cpu()) * 2

        prompt_len = 0
        if ref_dict is not None and "prompt_token" in ref_dict:
            prompt_len = ref_dict["prompt_token"].size(1)

        speech_tokens = self._pad_tokens_to_bucket(speech_tokens, prompt_len)
        n_cfm_timesteps = n_cfm_timesteps or (2 if self.meanflow else 10)

        output_mels = super().forward(
            speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=finalize,
        )
        return output_mels[:, :, :original_mel_len].contiguous()

    @torch.inference_mode()
    def hift_inference(self, speech_feat, cache_source: torch.Tensor | None = None):
        if cache_source is None:
            cache_source = torch.zeros(
                speech_feat.size(0),
                1,
                0,
                device=self.device,
                dtype=self.dtype,
            )
        return self.mel2wav.inference(
            speech_feat=speech_feat, cache_source=cache_source
        )

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens,
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        ref_dict: Optional[dict] = None,
        drop_invalid_tokens: bool = True,
        n_cfm_timesteps: Optional[int] = None,
        speech_token_lens: Optional[torch.Tensor] = None,
    ):
        output_mels = self.flow_inference(
            speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=True,
        )
        output_mels = output_mels.to(dtype=self.dtype)
        output_wavs, output_sources = self.hift_inference(output_mels, None)
        output_wavs[:, : len(self.trim_fade)] *= self.trim_fade
        return output_wavs, output_sources
