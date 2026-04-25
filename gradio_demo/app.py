# Gradio Demo for ACGF Prototype
import gradio as gr

def generate_music(prompt, strength):
    return f"""✅ ACGF Output Ready!

Prompt: {prompt}
Authenticity Strength: {strength}

Note: Full AudioLDM 2 + Voice AI models need to be downloaded first.
"""

demo = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(label="Folk Music Prompt", 
                   placeholder="Nigerian Igbo folk song with call-and-response vocals..."),
        gr.Slider(0.0, 1.0, value=0.8, label="Authenticity Constraint Strength")
    ],
    outputs="text",
    title="ACGF - Cultural Heritage Music Generator",
    description="Prototype based on the research paper"
)

if __name__ == "__main__":
    demo.launch()
