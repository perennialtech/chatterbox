import os

import torch
import gradio as gr
from chatterbox.vc import ChatterboxVC

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRECISION = os.getenv("CHATTERBOX_PRECISION")
if PRECISION is None:
    PRECISION = (
        "bf16"
        if DEVICE == "cuda" and torch.cuda.is_bf16_supported()
        else "fp16" if DEVICE == "cuda" else "fp32"
    )

# NOTE: We explicitly set `torch.backends.cudnn.benchmark = False`
# and do not set `mode="reduce-overhead"` for correct behavior.
COMPILE_MODEL = DEVICE == "cuda" and os.getenv("CHATTERBOX_COMPILE", "1") != "0"
WARMUP_MODEL = os.getenv("CHATTERBOX_WARMUP", "1") != "0"


model = ChatterboxVC.from_pretrained(
    DEVICE,
    precision=PRECISION,
    compile_model=COMPILE_MODEL,
    warmup=WARMUP_MODEL,
)


def generate(audio, target_voice_path, n_cfm_timesteps, cfg_rate):
    wav, timings = model.generate(
        audio,
        target_voice_path=target_voice_path,
        n_cfm_timesteps=int(n_cfm_timesteps),
        cfg_rate=float(cfg_rate),
    )
    timing_str = "\n".join([f"{k}: {v:.4f}" for k, v in timings.items()])
    return (model.sr, wav.squeeze(0).numpy()), timing_str


demo = gr.Interface(
    generate,
    [
        gr.Audio(
            sources=["upload", "microphone"], type="filepath", label="Input audio file"
        ),
        gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Target voice audio file (if none, the default voice is used)",
            value=None,
        ),
        gr.Slider(
            minimum=2,
            maximum=10,
            value=10,
            step=1,
            label="CFM timesteps",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=0.7,
            value=0.7,
            step=0.1,
            label="CFG rate",
        ),
    ],
    [
        gr.Audio(label="Output audio"),
        gr.Textbox(label="Execution Timings"),
    ],
)

if __name__ == "__main__":
    demo.launch()
