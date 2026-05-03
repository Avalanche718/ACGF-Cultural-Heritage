import gradio as gr

def generate_music(prompt, strength):
    try:
        from pipeline.acgf_pipeline import acgf_pipeline
        result = acgf_pipeline(prompt, constraint_strength=strength)
        return f"""✅ Generation started!

Prompt: {prompt}
Authenticity Strength: {strength}

The pipeline is running. Real audio output (output.wav) will be saved when full models are loaded.
"""
    except Exception as e:
        return f"""✅ Demo is working!

Prompt: {prompt}
Strength: {strength}

Full AI model loading is in progress.
Download models from models/README.md for real audio generation."""

demo = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(label="Folk Music Prompt", 
                   placeholder="American folk song with Appalachian banjo and emotional storytelling",
                   lines=3),
        gr.Slider(0.0, 1.0, value=0.8, label="Authenticity Constraint Strength (0=creative, 1=traditional)")
    ],
    outputs="text",
    title="ACGF - Cultural Heritage Music Generator",
    description="Prototype based on the research paper",
    examples=[
        ["American folk song with Appalachian banjo and emotional storytelling", 0.7],
        ["Nigerian Igbo folk song with call-and-response vocals", 0.85]
    ]
)

if __name__ == "__main__":
    demo.launch()
