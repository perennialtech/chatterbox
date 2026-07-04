import gradio as gr
import torch

from chatterbox.vc import ChatterboxVC

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = ChatterboxVC.from_pretrained(DEVICE)


def generate(audio, target_voice_path, upscale):
    wav, sr, timings = model.generate(
        audio,
        target_voice_path=target_voice_path,
        upscale=upscale,
    )

    timing_str = "\n".join([f"{k}: {v:.4f}" for k, v in timings.items()])
    return (sr, wav.squeeze(0).numpy()), timing_str


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
        gr.Checkbox(label="Upscale final audio via FlowHigh", value=True),
    ],
    [
        gr.Audio(label="Output audio"),
        gr.Textbox(label="Execution Timings"),
    ],
)

if __name__ == "__main__":
    demo.launch()
