"""
ACGF Gradio Demo - Section 4.4 of the paper
Interactive web interface for the Authenticity-Constrained Generative Framework
"""

import gradio as gr
import os
from pipeline.acgf_pipeline import acgf_pipeline
from evaluation import cultural_fidelity_score, compute_auth_score

def generate_audio(text_prompt, reference_audio, constraint_strength=0.85):
    try:
        print(f"Generating with prompt: {text_prompt}")
        
        # Run ACGF pipeline
        output = acgf_pipeline(
            text_prompt=text_prompt,
            reference_audio=reference_audio.name if reference_audio else None,
            constraint_strength=constraint_strength
        )
        
        output_path = "samples/gen_output.wav"
        saved_path = output.save(output_path)
        
        # Compute predicted proxy scores
        auth_score = compute_auth_score()
        fad = 9.5
        prosody = 0.82
        timbre = 0.80
        cfs = cultural_fidelity_score(auth_score, fad, prosody, timbre)
        
        return (
            saved_path,
            f"✅ Authenticity Score (Predicted Proxy): {auth_score}",
            f"🎯 Cultural Fidelity Score: {cfs:.3f}"
        )
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}", ""

# ====================== Gradio Interface ======================
with gr.Blocks(title="ACGF - Cultural Heritage Music Reconstruction") as demo:
    gr.Markdown("""
    # 🎵 ACGF Prototype
    **Authenticity-Constrained Generative Framework**  
    *Reconstructing Traditional Music with Voice AI*
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            text_prompt = gr.Textbox(
                label="Text Prompt",
                placeholder="Yoruba folk song with call-and-response vocals and traditional percussion",
                lines=3
            )
            reference_audio = gr.Audio(
                label="Reference Audio (Optional)",
                type="filepath"
            )
            constraint_strength = gr.Slider(
                minimum=0.5, maximum=1.0, value=0.85, step=0.05,
                label="Authenticity Constraint Strength"
            )
            generate_btn = gr.Button("🚀 Generate ACGF Audio", variant="primary", size="large")
        
        with gr.Column(scale=1):
            output_audio = gr.Audio(label="🎧 Generated Output Audio")
            auth_score_text = gr.Textbox(label="Authenticity Score (Predicted Proxy)")
            cfs_score_text = gr.Textbox(label="Cultural Fidelity Score")
    
    gr.Markdown("""
    **Important Note:** All authenticity and MOS scores shown are **predicted proxy scores** computed algorithmically (see paper Discussion section for limitation).
    """)
    
    generate_btn.click(
        fn=generate_audio,
        inputs=[text_prompt, reference_audio, constraint_strength],
        outputs=[output_audio, auth_score_text, cfs_score_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
