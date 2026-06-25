import gradio as gr
from pipeline.acgf_pipeline import acgf_pipeline

def generate_music(prompt, lyrics, strength, reference_audio):
    if not prompt:
        return None

    print(f"\n🎵 Generating: {prompt} | Strength: {strength}")
    result = acgf_pipeline(
        prompt,
        lyrics_text=lyrics or prompt,  # falls back to prompt if lyrics left blank
        constraint_strength=strength,
        reference_audio=reference_audio.name if reference_audio else None,
    )
    output_path = "output.wav"
    result.save(output_path)
    return output_path

demo = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(label="Folk Music Style Prompt", lines=2,
                   placeholder="Yoruba folk song with talking drum, instrumental style description"),
        gr.Textbox(label="Lyrics (Optional)", lines=3,
                   placeholder="The actual words to be sung - leave blank to reuse the style prompt above"),
        gr.Slider(0.0, 1.0, value=0.8, label="Authenticity Constraint Strength"),
        gr.Audio(label="Reference Audio (Optional)", type="filepath"),
    ],
    outputs=gr.Audio(label="Generated Output", type="filepath"),
    title="ACGF - Cultural Heritage Music Generator",
    description="Prototype based on the research paper",
    examples=[
        ["Yoruba praise chant with smooth flute", "Ę ku ọjọ́ àyẹyẹ, ę ku àánú Olúwa", 0.9, None],
        ["Nigerian Igbo folk song with call-and-response vocals", "", 0.85, None],
        ["Yoruba folk song with talking drum and emotional storytelling", "", 0.7, None],
    ]
)

if __name__ == "__main__":
    demo.launch()
