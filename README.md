# ACGF: Authenticity-Constrained Generative Framework

**Authenticity-Creativity Trade-offs in Generative AI Representations of Cultural Heritage: Voice AI Integration in Music Reconstruction**

Official prototype repository for the research paper.

### Quick Start

```bash
git clone https://github.com/Avalanche718/ACGF-Cultural-Heritage.git
cd ACGF-Cultural-Heritage
pip install -r requirements.txt
cd gradio_demo
python app.py
```

Without any extra setup, the pipeline will run Step 1 (AudioLDM 2) for real and fail gracefully on Step 2 with a clear error telling you what's missing.

### Model Setup

**Step 1 — AudioLDM 2** (instrumental generation): installed automatically via `requirements.txt`. First run will download `cvssp/audioldm2-music` from Hugging Face (~few GB).

**Step 2 — Voice synthesis** (`pipeline/acgf_pipeline.py: voice_synthesis`): these are not on PyPI. Pick one backbone and follow its own setup:

| Backbone | Repo | Notes |
|---|---|---|
| DiffSinger (default) | https://github.com/MoonInTheRiver/DiffSinger | Needs a symbolic score/phoneme alignment as input, not raw text |
| Fish Speech | https://github.com/fishaudio/fish-speech | Zero-shot cloning from a short reference clip |
| CosyVoice | https://github.com/FunAudioLLM/CosyVoice | Good for cross-lingual reference-conditioned synthesis |

Clone your chosen backbone into a `third_party/` folder at the repo root and follow its own checkpoint download instructions, then switch `voice_backbone` in `acgf_pipeline()` accordingly.

**Reference audio**: `reference_audio` should be a short (3-10s), clean, mono clip you have the rights/consent to use. This repo does not ship with any pre-loaded reference voice — supply your own per the corpus described in the paper (§3.3), with documented provenance.

**Step 3 — Authenticity discriminator**: not yet trained in this repo (`discriminator/` is currently empty — see paper §3.2 for the intended architecture and training data). Until a checkpoint is trained and loaded, `acgf_pipeline()` will skip the refinement loop and return the raw Step 1+2 mixdown.

