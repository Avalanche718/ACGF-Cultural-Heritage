import gradio as gr
from pipeline.acgf_pipeline import acgf_pipeline

def generate_music(prompt, strength):
    if not prompt:
        return "Please enter a prompt."
    
    print(f"\n🎵 Generating: {prompt} | Strength: {strength}")
    result = acgf_pipeline(prompt, constraint_strength=strength)
    result.save("output.wav")
    
    return """✅ Generation Completed!

Real audio file (output.wav) has been saved in the current folder.
Check the terminal for detailed logs.

Note: Full model inference requires downloaded models."""

demo = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(label="Folk Music Prompt", lines=3, 
                   placeholder="American folk song with Appalachian banjo and emotional storytelling"),
        gr.Slider(0.0, 1.0, value=0.8, label="Authenticity Constraint Strength")
    ],
    outputs="text",
    title="ACGF - Cultural Heritage Music Generator",
    description="Prototype based on the research paper",
    examples=[
        ["American folk song with Appalachian banjo and emotional storytelling", 0.7],
        ["Nigerian Igbo folk song with call-and-response vocals", 0.85],
        ["Yoruba praise chant with smooth flute", 0.9]
    ]
)

if __name__ == "__main__":
    demo.launch()
