import gradio as gr
from pipeline.acgf_pipeline import acgf_pipeline

def generate_music(prompt, strength, reference_audio):
    if not prompt:
        return None

    print(f"\n🎵 Generating: {prompt} | Strength: {strength}")
    result = acgf_pipeline(
        prompt,
        constraint_strength=strength,
        reference_audio=reference_audio.name if reference_audio else None,
    )
    output_path = "output.wav"
    result.save(output_path)
    return output_path

demo = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(label="Folk Music Prompt", lines=3,
                   placeholder="Yoruba folk song with talking drum and call-and-response vocals"),
        gr.Slider(0.0, 1.0, value=0.8, label="Authenticity Constraint Strength"),
        gr.Audio(label="Reference Audio (Optional)", type="filepath"),
    ],
    outputs=gr.Audio(label="Generated Output", type="filepath"),
    title="ACGF - Cultural Heritage Music Generator",
    description="Prototype based on the research paper",
    examples=[
        ["Yoruba praise chant with smooth flute", 0.9, None],
        ["Nigerian Igbo folk song with call-and-response vocals", 0.85, None],
        ["Yoruba folk song with talking drum and emotional storytelling", 0.7, None],
    ]
)

if __name__ == "__main__":
    demo.launch()
