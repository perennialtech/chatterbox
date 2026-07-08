import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ...audio import S3_SR, S3GEN_SR, mel_spectrogram, resample_audio
from ..s3tokenizer import S3Tokenizer
from ..speaker.campplus import CAMPPlus
from ..speaker.features import extract_fbank_features
from ..token_utils import drop_invalid_tokens
from .conditioning import S3ReferenceCondition
from .configs import CFM_PARAMS
from .const import S3GEN_SIL
from .decoder import ConditionalDecoder
from .f0_predictor import ConvRNNF0Predictor
from .flow_matching import ConditionalCFM
from .transformer.upsample_encoder import UpsampleConformerEncoder
from .utils.mask import make_pad_mask
from .vocoder import HiFTGenerator


def _repeat_batch_dim(tnsr, B, ndim):
    if tnsr is not None:
        while tnsr.ndim < ndim:
            tnsr = tnsr[None]
        if B > 1 and tnsr.size(0) == 1:
            tnsr = tnsr.repeat(B, *([1] * (ndim - 1)))
        assert tnsr.ndim == ndim, f"Expected {ndim=}, got {tnsr.ndim=}"
    return tnsr


def _prepare_speech_tokens(
    speech_tokens,
    speech_token_lens: Optional[torch.Tensor],
    device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not torch.is_tensor(speech_tokens):
        speech_tokens = torch.as_tensor(speech_tokens)

    if speech_tokens.ndim == 0:
        raise ValueError("speech_tokens must have shape [T] or [1, T]")
    if speech_tokens.ndim > 2:
        raise ValueError(
            "speech_tokens must have shape [T] or [1, T], "
            f"got {tuple(speech_tokens.shape)}"
        )

    speech_tokens = torch.atleast_2d(speech_tokens).to(
        device=device,
        dtype=torch.long,
    )
    if speech_tokens.size(0) != 1:
        raise ValueError("Only batch size 1 is supported")

    if speech_token_lens is None:
        speech_token_lens = torch.full(
            (speech_tokens.size(0),),
            speech_tokens.size(-1),
            dtype=torch.long,
            device=device,
        )
    else:
        speech_token_lens = torch.as_tensor(speech_token_lens, device=device)
        if speech_token_lens.ndim == 0:
            speech_token_lens = speech_token_lens.reshape(1)
        if speech_token_lens.ndim != 1:
            raise ValueError("speech_token_lens must have shape [B]")
        speech_token_lens = speech_token_lens.to(dtype=torch.long)

    if speech_token_lens.numel() != speech_tokens.size(0):
        raise ValueError("speech_token_lens must have one entry per batch item")

    original_token_len = int(speech_token_lens.max().detach().cpu())
    if original_token_len <= 0:
        raise ValueError("At least one speech token is required")
    if original_token_len > speech_tokens.size(1):
        raise ValueError("speech_token_lens exceeds speech_tokens length")

    return (
        speech_tokens[:, :original_token_len].contiguous(),
        speech_token_lens,
        original_token_len,
    )


class OfflineTokenToMelFlow(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        encoder: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        self.input_embedding = torch.nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        embedding,
        finalize,
        n_timesteps=10,
        noised_mels=None,
        meanflow=False,
    ):
        B = token.size(0)
        embedding = torch.atleast_2d(embedding)
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        prompt_token = _repeat_batch_dim(prompt_token, B, ndim=2)
        prompt_token_len = _repeat_batch_dim(prompt_token_len, B, ndim=1)
        prompt_feat = _repeat_batch_dim(prompt_feat, B, ndim=3)
        embedding = _repeat_batch_dim(embedding, B, ndim=2)

        token, token_len = (
            torch.cat([prompt_token, token], dim=1),
            prompt_token_len + token_len,
        )
        mask = (
            (~make_pad_mask(token_len, max_len=token.size(1)))
            .unsqueeze(-1)
            .to(embedding)
        )

        token = self.input_embedding(token.long()) * mask

        h, h_masks = self.encoder(token, token_len)
        if finalize is False:
            h = h[:, : -self.pre_lookahead_len * self.token_mel_ratio]

        h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        conds = torch.zeros(
            [B, mel_len1 + mel_len2, self.output_size], device=token.device
        ).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(h_lengths, max_len=h.shape[1])).unsqueeze(1).to(h)
        if mask.shape[0] != B:
            mask = mask.repeat(B, 1, 1)

        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            meanflow=meanflow,
        )
        feat = feat[:, :, mel_len1:]
        return feat, None


logger = logging.getLogger(__name__)

_TOKEN_LENGTH_BUCKETS = (32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072)


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
        self.speaker_encoder = CAMPPlus()
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
            input_size=512,
        )

        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn="gelu",
            meanflow=self.meanflow,
        )
        decoder = ConditionalCFM(
            in_channels=80,
            cfm_params=CFM_PARAMS,
            estimator=estimator,
        )

        self.flow = OfflineTokenToMelFlow(encoder=encoder, decoder=decoder)

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
            return ref_dict.trim_to_lengths()
        return S3ReferenceCondition.from_mapping(
            ref_dict, device=self.device, dtype=self.dtype
        ).trim_to_lengths()

    def prepare_ref_dict(self, ref_dict: dict) -> dict:
        return self.prepare_ref_condition(ref_dict).as_dict()

    @torch.inference_mode()
    def embed_ref(
        self,
        ref_wav: torch.Tensor,
        ref_sr: int,
        device="auto",
    ) -> dict:
        self.eval()
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

        ref_wav_16 = resample_audio(ref_wav, ref_sr, S3_SR, device)
        ref_fbank = extract_fbank_features(ref_wav_16)
        ref_x_vector = self.speaker_encoder(
            ref_fbank.to(dtype=self.dtype, device=device)
        )
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16.float())

        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            logger.warning(
                "Reference mel length differs from 2x token length; trimming."
            )
            aligned_token_len = min(ref_speech_tokens.size(1), ref_mels_24.size(1) // 2)
            if aligned_token_len <= 0:
                raise ValueError(
                    "reference audio is too short for aligned conditioning"
                )

            aligned_mel_len = aligned_token_len * 2
            ref_speech_tokens = ref_speech_tokens[:, :aligned_token_len].contiguous()
            ref_mels_24 = ref_mels_24[:, :aligned_mel_len].contiguous()
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
                "embedding": ref_x_vector,
            }
        )

    def _generate_mel(
        self,
        speech_tokens: torch.Tensor,
        speech_token_lens: torch.Tensor,
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        ref_dict: Optional[dict],
        n_cfm_timesteps: Optional[int],
        finalize: bool,
    ) -> torch.Tensor:
        if (ref_wav is None) == (ref_dict is None):
            raise ValueError("Provide exactly one of ref_wav or ref_dict")

        if ref_dict is None:
            ref_dict = self.embed_ref(ref_wav, ref_sr)
        else:
            ref_dict = self.prepare_ref_dict(ref_dict)

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
        speech_tokens, speech_token_lens, _ = _prepare_speech_tokens(
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            device=self.device,
        )
        return self._generate_mel(
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=finalize,
        )


class S3Token2Wav(S3Token2Mel):
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
        end_fade = (torch.cos(torch.linspace(0, torch.pi, 2 * n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False)
        self.register_buffer("end_fade", end_fade, persistent=False)

    def compile_for_inference(self) -> "S3Token2Wav":
        self.eval()
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

    @property
    def _token_mel_ratio(self) -> int:
        return int(getattr(self.flow, "token_mel_ratio", 2))

    @property
    def _final_context_token_count(self) -> int:
        return max(0, int(getattr(self.flow, "pre_lookahead_len", 0)))

    def _prepare_flow_inputs(
        self,
        speech_tokens: torch.Tensor,
        speech_token_lens: Optional[torch.Tensor],
        finalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        speech_tokens, speech_token_lens, original_token_len = _prepare_speech_tokens(
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            device=self.device,
        )
        original_mel_len = original_token_len * self._token_mel_ratio

        context_tokens = self._final_context_token_count if finalize else 0
        if context_tokens > 0:
            tail = torch.full(
                (speech_tokens.size(0), context_tokens),
                S3GEN_SIL,
                dtype=torch.long,
                device=self.device,
            )
            speech_tokens = torch.cat([speech_tokens, tail], dim=1)
            speech_token_lens = speech_token_lens + context_tokens

        return speech_tokens, speech_token_lens, original_mel_len

    @torch.inference_mode()
    def _flow_inference_impl(
        self,
        speech_tokens,
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        ref_dict: Optional[dict] = None,
        n_cfm_timesteps: Optional[int] = None,
        finalize: bool = False,
        speech_token_lens: Optional[torch.Tensor] = None,
        keep_final_context: bool = False,
    ) -> tuple[torch.Tensor, int]:
        speech_tokens, speech_token_lens, original_mel_len = self._prepare_flow_inputs(
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            finalize=finalize,
        )
        model_mel_len = (
            int(speech_token_lens.max().detach().cpu()) * self._token_mel_ratio
        )

        if ref_dict is not None:
            ref_dict = self.prepare_ref_dict(ref_dict)

        prompt_len = 0
        if ref_dict is not None and "prompt_token" in ref_dict:
            prompt_len = ref_dict["prompt_token"].size(1)

        speech_tokens = self._pad_tokens_to_bucket(speech_tokens, prompt_len)

        output_mels = self._generate_mel(
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=finalize,
        )
        trim_len = model_mel_len if keep_final_context else original_mel_len
        return output_mels[:, :, :trim_len].contiguous(), original_mel_len

    def _apply_output_fades(self, output_wavs: torch.Tensor) -> torch.Tensor:
        if output_wavs.size(1) == 0:
            return output_wavs

        start_len = min(output_wavs.size(1), self.trim_fade.numel())
        if start_len > 0:
            output_wavs[:, :start_len] *= self.trim_fade[:start_len].to(
                device=output_wavs.device,
                dtype=output_wavs.dtype,
            )

        end_len = min(output_wavs.size(1), self.end_fade.numel())
        if end_len > 0:
            output_wavs[:, -end_len:] *= self.end_fade[-end_len:].to(
                device=output_wavs.device,
                dtype=output_wavs.dtype,
            )

        return output_wavs

    @torch.inference_mode()
    def warmup(
        self,
        ref_dict: Optional[dict] = None,
        total_token_buckets=(384, 512, 768, 1024),
    ) -> None:
        self.eval()
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
            output_mels, _ = self._flow_inference_impl(
                speech_tokens=speech_tokens,
                ref_dict=ref_dict,
                n_cfm_timesteps=2 if self.meanflow else 10,
                finalize=True,
                keep_final_context=True,
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
        output_mels, original_mel_len = self._flow_inference_impl(
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=finalize,
            keep_final_context=finalize,
        )

        if skip_vocoder:
            return output_mels[:, :, :original_mel_len].contiguous()

        hift_cache_source = torch.zeros(
            output_mels.size(0),
            1,
            0,
            device=output_mels.device,
            dtype=output_mels.dtype,
        )
        output_wavs, output_sources = self.mel2wav.inference(
            speech_feat=output_mels,
            cache_source=hift_cache_source,
        )

        if finalize:
            original_samples = min(
                output_wavs.size(1),
                original_mel_len * self.mel2wav.source_hop,
            )
            output_wavs = output_wavs[:, :original_samples].contiguous()
            output_sources = output_sources[:, :, :original_samples].contiguous()

        if not self.training:
            self._apply_output_fades(output_wavs)

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
        self.eval()
        output_mels, _ = self._flow_inference_impl(
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=finalize,
            keep_final_context=False,
        )
        return output_mels

    @torch.inference_mode()
    def hift_inference(self, speech_feat, cache_source: torch.Tensor | None = None):
        self.eval()
        if cache_source is None:
            cache_source = torch.zeros(
                speech_feat.size(0),
                1,
                0,
                device=speech_feat.device,
                dtype=speech_feat.dtype,
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
        self.eval()
        if drop_invalid_tokens:
            speech_tokens = drop_invalid_tokens_fn(speech_tokens)
            speech_token_lens = None

        output_mels, original_mel_len = self._flow_inference_impl(
            speech_tokens,
            speech_token_lens=speech_token_lens,
            ref_wav=ref_wav,
            ref_sr=ref_sr,
            ref_dict=ref_dict,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=True,
            keep_final_context=True,
        )
        output_mels = output_mels.to(dtype=self.dtype)
        output_wavs, output_sources = self.hift_inference(output_mels, None)

        original_samples = min(
            output_wavs.size(1),
            original_mel_len * self.mel2wav.source_hop,
        )
        output_wavs = output_wavs[:, :original_samples].contiguous()
        output_sources = output_sources[:, :, :original_samples].contiguous()

        self._apply_output_fades(output_wavs)
        return output_wavs, output_sources


drop_invalid_tokens_fn = drop_invalid_tokens
