# ACGF: Authenticity-Constrained Generative Framework

**Authenticity-Creativity Trade-offs in Generative AI Representations of Cultural Heritage: Voice AI Integration in Music Reconstruction**

This repository contains the prototype for the research paper.

### Key Components
- Generative Backbone: AudioLDM 2 (latent diffusion)
- Authenticity Discriminator: Trained on expert-annotated folk music (Yoruba, Igbo, etc.)
- Voice AI: DiffSinger + Fish Speech (S2-Pro) + CosyVoice 2
- Iterative refinement loop

### How Client Can Run It Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/Avalanche718/ACGF-Cultural-Heritage.git
   cd ACGF-Cultural-Heritage
