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

The demo has two text inputs: a **style prompt** (describes the instrumental sound, e.g. "Yoruba praise chant with smooth flute") and **lyrics** (the actual words to be sung — optional, falls back to reusing the style prompt as sung text if left blank, which won't sound right but won't crash either).

### Hardware & Environment Requirements

| Stage | Disk space | GPU (recommended) | CPU-only feasible? |
|---|---|---|---|
| Step 1 — AudioLDM 2 (`audioldm2-music`) | ~2.2 GB (fp16) / ~4.4 GB (fp32) — 350M UNet, 1.1B total params | 8 GB+ VRAM for reasonable speed | Yes, but slow (minutes per 10s clip) |
| Step 2 — Fish Audio S2 Pro | ~10.3 GB (bf16) / ~5.1 GB (int8 quantized) — 4B param Dual-AR transformer | 16 GB+ VRAM (int8) / 24 GB+ (bf16) | Not practical |
| Step 2 — CosyVoice2-0.5B (not yet wired, see below) | ~4.9 GB full checkpoint folder | 8 GB+ VRAM | Possible but slow |
| Step 2 — DiffSinger (not yet wired, see below) | Varies by checkpoint; no official size published | 8 GB+ VRAM | Possible but slow |
| Step 3 — discriminator (not yet trained) | TBD — depends on architecture chosen | TBD | TBD |

**Practical minimum to run Step 1 + Fish Speech S2 Pro (int8) end-to-end:** ~16 GB VRAM GPU, ~20 GB free disk space for model weights alone (more for cache/temp files during download).

**Important environment conflict:** Fish Speech (current `main`) pins `torch==2.8.0` / `torchaudio==2.8.0`. CosyVoice pins `torch==2.3.1` / `torchaudio==2.3.1`. **These cannot coexist in one Python environment.** If you need both backbones, use separate virtual environments (e.g. `conda create -n acgf-fish` / `conda create -n acgf-cosy`) and switch between them, rather than installing both into the same env from `requirements.txt`.

### Third-Party Model Verification

The three voice backbones referenced in `pipeline/acgf_pipeline.py` are GitHub repos, not PyPI packages. Each was independently cloned and its real import paths/APIs checked against what the pipeline code calls — here's the actual state of each, pinned to a specific commit/version so these notes don't go stale silently:

| Backbone | Verified against | Status in `acgf_pipeline.py` | What's real vs. what's still a placeholder |
|---|---|---|---|
| **Fish Speech (S2 Pro)** | commit `e5e2926` (2026-06-09), `pyproject.toml` version `2.0.0` | **Wired up** — `voice_synthesis(backbone="fish_speech")` calls the real `TTSInferenceEngine`, `launch_thread_safe_queue`, `load_decoder_model`, `ServeTTSRequest`/`ServeReferenceAudio` classes, confirmed to exist at those exact import paths. Requires a `lyrics_text` argument (the words to be sung) — `acgf_pipeline()` takes this as a parameter, separate from the style-description `text_prompt` used for Step 1; the Gradio demo exposes both as separate text boxes. | Checkpoint paths (`checkpoints/fish-speech-s2-pro`) are placeholders you fill in after downloading `fishaudio/s2-pro` from Hugging Face — the function calls themselves are real, not fabricated. |
| **CosyVoice2** | commit `074ca6d` (2026-05-26) | **Not wired** — raises `NotImplementedError` with setup instructions. | Real `CosyVoice2` class is at `cosyvoice.cli.cosyvoice.CosyVoice2`, constructed with a local `model_dir` (containing `cosyvoice2.yaml`, `campplus.onnx`, `llm.pt`, `flow.pt`, `hift.pt` — pulled via `snapshot_download`), not a Hugging Face repo-id string. Zero-shot synthesis is `inference_zero_shot()` on the base class. |
| **DiffSinger** | commit `ce7789f` (2025-03-19) | **Not wired** — raises `NotImplementedError` with setup instructions. | No PyPI package; no `from_pretrained()`. Requires a YAML config via `set_hparams()`, a hard-coded Mandarin phoneme table (`cpop_pinyin2ph_func`, built for the Opencpop dataset), and checkpoint files placed by hand. It does **not** accept a `reference_audio` waveform the way the other two backbones do — integrating it for Yoruba/Igbo lyrics needs a different design (custom phoneme mapping), not just glue code. Flagging this as a real open design question rather than pretending it's a drop-in swap. |

An earlier version of this file had plausible-looking but incorrect calls for all three backbones (e.g. `DiffSingerSVS.from_pretrained()`, `FishSpeechS2Pro.clone_and_synthesize()`, `CosyVoice2.from_pretrained()` from a `.inference` submodule) — none of those classes/methods exist in the real repos. That's been corrected above; only Fish Speech is currently a real, callable integration.

### Step 3 — Authenticity Discriminator (`discriminator/.gitkeep`)

`discriminator/` is currently an empty placeholder — there is no trained model and no training script in this repo yet. What it's intended to hold, per paper §3.2:

- A CNN-based binary/scalar classifier trained to distinguish "culturally authentic" Yoruba/Igbo vocal style from generated/out-of-distribution style, trained on the annotated CBAAC/archival corpus described in the paper.
- A `train_discriminator.py` script (not yet written) that would consume labeled (authentic, synthetic) pairs and produce a checkpoint.
- The checkpoint, once trained, gets passed into `iterative_refinement(discriminator=...)` in `acgf_pipeline.py`, which already has the calling convention and CFS formula stubbed in with a `NotImplementedError` and a comment pointing back here.

Until that training work happens, `acgf_pipeline()` runs Steps 1–2 and skips Step 3's gating, returning the raw mixdown. This is a real, separate piece of work (data labeling + model training), not something fillable from the pipeline file alone.

### Evaluation Methodology Note (Tables 1 & 2)

`evaluation.py` was previously deleted from the repo after its last version was found to return FAD, prosody similarity, timbre similarity, and the MOS proxy via `np.random.uniform(...)` calls — none of those numbers were computed from actual audio. It's been rewritten from scratch with real computation:

- **Prosody similarity**: real pitch-contour comparison via `librosa.yin` + DTW alignment (not random).
- **Timbre similarity**: real MFCC cosine similarity (not random).
- **FAD**: real computation via [`fadtk`](https://github.com/microsoft/fadtk) (`pip install fadtk` separately - not bundled in `requirements.txt`, see the script's docstring for why).
- **Cultural Fidelity Score**: same weights as paper §5.2, but `D_conf` (discriminator confidence) is currently a placeholder (0.5) since `discriminator/` has no trained model yet - the script prints this caveat explicitly every time it runs, rather than letting a CFS number look more authoritative than it is.

What's still missing before these numbers can support published Table 1/2 claims:
1. **The actual Yoruba/Igbo reference + generated audio pairs.** `evaluation.py` expects `samples/ref_yoruba.wav`, `samples/gen_yoruba_acgf.wav`, `samples/ref_igbo.wav`, `samples/gen_igbo_acgf.wav` — none of these exist yet (only American folk/dulcimer samples are currently in `samples/`). The script will raise a clear `FileNotFoundError` pointing at this gap rather than silently fabricating a result. **Note:** `.gitignore` currently excludes `*.wav` and `*.mp3` — once you have real audio to add, a plain `git add` will silently skip them. Either add the files with `git add -f path/to/file.wav`, or adjust `.gitignore` to allow `samples/` specifically (e.g. add `!samples/**/*.wav` and `!samples/**/*.mp3` below the existing exclusion rules).
2. **A trained discriminator**, so `D_conf` becomes real instead of a fixed placeholder.
3. **A human listening study** for the MOS claim specifically — a formula-based "predicted MOS" isn't a substitute for actual listener ratings, since proxy-MOS prediction is itself an active research area, not a settled methodology. Even a small study (10-20 raters, 5-point scale, per pair) would give reviewers something real to evaluate.
4. **FAD's known limitations**, worth pre-empting in the paper: it correlates with human perceptual ratings at ~0.52 (Kilgour et al. 2018) — solid but imperfect — and its Gaussian-distribution assumption is debated for real audio. The newer Kernel Audio Distance (KAD) metric is a citable complement/alternative if reviewers push on FAD's validity specifically.
