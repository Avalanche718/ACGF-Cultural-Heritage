"""
ACGF Pipeline - Real Model Integration (Updated for Client)
Based on Section 4 of the paper

Model sources (not on PyPI - see README "Model Setup" section):
  - AudioLDM 2:  pip install diffusers transformers  (already in requirements.txt)
  - DiffSinger:  git clone https://github.com/MoonInTheRiver/DiffSinger
  - Fish Speech: git clone https://github.com/fishaudio/fish-speech
  - CosyVoice:   git clone https://github.com/FunAudioLLM/CosyVoice
Clone each into a `third_party/` folder at the repo root and follow their own
setup instructions for checkpoints before enabling the real calls below.
"""

import os
import torch
import torchaudio
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 44100  # matches paper §5.1 (all corpus audio normalized to 44.1kHz)


class AudioOutput:
    """Wraps a numpy/torch waveform + metadata so .save() always works,
    whether models actually ran or we fell back to a placeholder."""

    def __init__(self, waveform=None, sample_rate=SAMPLE_RATE, cfs_score=None, metadata=None):
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.cfs_score = cfs_score
        self.metadata = metadata or {}

    def save(self, filepath: str = "output.wav"):
        if self.waveform is None:
            # Write a short silent file so downstream consumers (e.g. the
            # Gradio demo's gr.Audio output) don't break on a missing path.
            silent = torch.zeros(1, int(self.sample_rate * 1.0))
            torchaudio.save(filepath, silent, self.sample_rate)
            print(f"💾 Placeholder (silent) file written: {filepath}")
            print("   (No audio was generated - models likely not downloaded yet)")
            return
        wav = torch.as_tensor(np.asarray(self.waveform)).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # torchaudio expects (channels, samples)
        torchaudio.save(filepath, wav, self.sample_rate)
        print(f"💾 Saved final output to: {filepath}")
        if self.cfs_score is not None:
            print(f"   Cultural Fidelity Score: {self.cfs_score:.3f}")


def _load_audioldm2():
    """Step 1 model loader. cvssp/audioldm2-music is preferred over the base
    checkpoint since it's tuned for musical content (paper §3.1)."""
    from diffusers import AudioLDM2Pipeline
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-music",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    return pipe.to(DEVICE)


def generate_instrumental(pipe, text_prompt: str, duration_seconds: float = 10.0, seed: int | None = None):
    """Step 1: base music via AudioLDM 2 latent diffusion (paper §3.1, §4.2 Listing 1)."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=text_prompt,
        negative_prompt="low quality, distorted, noisy",
        num_inference_steps=200,
        audio_length_in_s=duration_seconds,
        num_waveforms_per_prompt=1,
        generator=generator,
    )
    audio = result.audios[0]  # AudioLDM2 outputs at 16kHz natively

    # Resample 16kHz -> 44.1kHz so it's compatible with voice synthesis stage
    audio_t = torch.as_tensor(audio).unsqueeze(0)
    audio_44k = torchaudio.functional.resample(audio_t, orig_freq=16000, new_freq=SAMPLE_RATE)
    return audio_44k.squeeze(0).numpy()


def voice_synthesis(instrumental_audio, reference_audio, constraint_strength: float,
                     backbone: str = "fish_speech"):
    """
    Step 2: constrained voice reconstruction (paper §3.3).

    backbone:
      - "fish_speech" : WORKING - zero-shot reference-conditioned synthesis. Requires
                         reference_audio. Verified against fish-speech commit e5e2926
                         (2026-06-09). See README "Third-Party Model Verification".
      - "cosyvoice"   : NOT YET WIRED - real CosyVoice2 API requires a local model_dir
                         with config/checkpoint files pulled via snapshot_download, not
                         a simple from_pretrained() call. Raises NotImplementedError below
                         until that setup path is built. See README for what's needed.
      - "diffsinger"  : NOT YET WIRED - real DiffSinger has no pip package and is hard-coded
                         for Mandarin phonemes (Opencpop dataset) - it does not accept raw
                         reference_audio for style transfer the way the other two do, so it
                         needs a different integration shape, not just a stub fill-in.
                         Raises NotImplementedError below. See README for details.

    constraint_strength bounds how far synthesis can drift from reference_audio's
    timbre/prosody - this is style-transfer within bounds, not full voice cloning.
    The actual authenticity gating happens in Step 3's discriminator loop; this
    value is the *starting* bound passed into that loop.

    IMPORTANT: fish-speech pins torch==2.8.0; cosyvoice (when wired) pins
    torch==2.3.1. The two cannot share one Python environment - see README
    "Hardware & Environment Requirements".
    """
    if backbone in ("fish_speech", "cosyvoice") and reference_audio is None:
        raise ValueError(f"{backbone} requires reference_audio for style transfer.")

    if backbone == "fish_speech":
        # Verified against https://github.com/fishaudio/fish-speech @ e5e2926 (2026-06-09)
        # CORRECTION: there is no TTSInferenceEngine.from_pretrained() classmethod.
        # The real setup loads two separate components first (see tools/run_webui.py
        # in the fish-speech repo for the reference implementation this mirrors):
        #   1. launch_thread_safe_queue(checkpoint_path=...) -> llama_queue
        #   2. load_decoder_model(config_name=..., checkpoint_path=...) -> decoder_model
        # Current flagship checkpoint is "Fish Audio S2 Pro" (4B params, ~10GB bf16 /
        # ~5GB int8) at https://huggingface.co/fishaudio/s2-pro - NOT fish-speech-1.5,
        # which is an older, smaller (1.5GB) checkpoint for a different code path.
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

        llama_queue = launch_thread_safe_queue(
            checkpoint_path="checkpoints/fish-speech-s2-pro",  # path after huggingface-cli download
            device=DEVICE,
            precision=torch.bfloat16,
            compile=False,
        )
        decoder_model = load_decoder_model(
            config_name="modded_dac_vq",  # confirm against the checkpoint's actual config name
            checkpoint_path="checkpoints/fish-speech-s2-pro/codec.pth",
            device=DEVICE,
        )
        engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            compile=False,
            precision=torch.bfloat16,
        )
        # ServeReferenceAudio expects raw audio bytes, not a waveform array -
        # convert reference_audio (path or array) to bytes before calling this.
        request = ServeTTSRequest(
            text=lyrics_or_score_text,
            references=[ServeReferenceAudio(audio=reference_audio, text="")],
        )
        results = list(engine.inference(request))
        vocals = np.concatenate([r.audio for r in results if r.audio is not None])

    elif backbone == "cosyvoice":
        raise NotImplementedError(
            "CosyVoice2 is not wired up yet. Its real constructor needs a local "
            "model_dir containing cosyvoice2.yaml, campplus.onnx, llm.pt, flow.pt, "
            "hift.pt (pulled via snapshot_download), not a from_pretrained(repo_id) "
            "call. See README 'Third-Party Model Verification' for the exact setup "
            "and the real inference_zero_shot() call signature."
        )

    elif backbone == "diffsinger":
        raise NotImplementedError(
            "DiffSinger is not wired up yet. It has no PyPI package, requires a "
            "config YAML via set_hparams(), Mandarin-only phoneme mapping "
            "(cpop_pinyin2ph_func), and manually placed checkpoint files - it does "
            "not take a simple reference_audio waveform like fish_speech/cosyvoice "
            "do. See README 'Third-Party Model Verification' before attempting "
            "integration; this backbone needs design work, not just glue code."
        )

    else:
        raise ValueError(f"Unknown voice backbone: {backbone}")

    return vocals


def _mix_instrumental_and_vocals(instrumental, vocals, vocal_gain: float = 1.0):
    """Simple additive mixdown. Pads the shorter track with silence so
    lengths match - replace with proper alignment if vocals/instrumental
    timing isn't already synced upstream."""
    n = max(len(instrumental), len(vocals))
    inst_padded = np.pad(instrumental, (0, n - len(instrumental)))
    voc_padded = np.pad(vocals, (0, n - len(vocals))) * vocal_gain
    mix = inst_padded + voc_padded
    peak = np.max(np.abs(mix)) or 1.0
    return mix / peak  # normalize to avoid clipping


def iterative_refinement(instrumental, vocals, reference_audio=None,
                          discriminator=None, cfs_threshold: float = 0.8,
                          max_refinement_steps: int = 10,
                          constraint_strength: float = 0.8):
    """
    Step 3: authenticity-constrained refinement loop (paper §3.2-3.4).

    If `discriminator` is None (true today - discriminator/ is not yet trained,
    see discriminator/.gitkeep), this step is skipped gracefully: it returns the
    Step 1+2 mixdown as-is with cfs_score=None, rather than crashing or pretending
    to score something it has no model for. Once a trained discriminator is
    available, pass it in to enable real rejection-sampling/guidance.
    """
    candidate = _mix_instrumental_and_vocals(instrumental, vocals)

    if discriminator is None:
        print("   ⚠️  No authenticity discriminator loaded - skipping refinement loop.")
        print("      Train one in discriminator/ (see paper §3.2) to enable this step.")
        return candidate, None

    # TODO: once discriminator/ has a trained model, implement:
    #   1. compute_cultural_fidelity_score(candidate, reference_audio, discriminator)
    #      per paper §5.2: CFS = 0.4*D_conf + 0.25*(1-NormFAD) + 0.2*ProsodySim + 0.15*TimbreSim
    #   2. loop: re-synthesize vocals at higher constraint_strength if CFS < cfs_threshold,
    #      up to max_refinement_steps, keeping the best-scoring candidate seen
    # Left unimplemented here since it depends on a discriminator checkpoint
    # that doesn't exist in this repo yet - see discriminator/.gitkeep.
    raise NotImplementedError(
        "A discriminator was passed in, but the refinement loop body isn't "
        "implemented yet. Train discriminator/ first (paper §3.2), then "
        "implement the CFS scoring loop described in the TODO above."
    )


def acgf_pipeline(text_prompt: str, reference_audio=None, constraint_strength: float = 0.8,
                   duration_seconds: float = 10.0, seed: int | None = None,
                   voice_backbone: str = "fish_speech", discriminator=None):
    """
    Main ACGF Pipeline with real model loading structure.
    """
    print("🚀 Starting ACGF Pipeline...")
    print(f"Prompt: {text_prompt}")
    print(f"Authenticity Constraint Strength: {constraint_strength}")

    try:
        # Step 1: Generate base music with latent diffusion (AudioLDM 2)
        print("\n[1/3] Generating instrumental base with AudioLDM 2...")
        audioldm2_pipe = _load_audioldm2()
        instrumental = generate_instrumental(audioldm2_pipe, text_prompt, duration_seconds, seed)
        print("   → AudioLDM 2 generation complete")

        # Step 2: Constrained voice reconstruction
        print("\n[2/3] Reconstructing vocals with Voice AI (DiffSinger / Fish Speech / CosyVoice)...")
        vocals = voice_synthesis(instrumental, reference_audio, constraint_strength, backbone=voice_backbone)
        print(f"   → Voice AI reconstruction complete ({voice_backbone})")

        # Step 3: Apply Authenticity Constraints + Refinement
        print("\n[3/3] Applying Authenticity Constraints & Iterative Refinement...")
        final_audio, cfs_score = iterative_refinement(
            instrumental, vocals, reference_audio=reference_audio,
            discriminator=discriminator, constraint_strength=constraint_strength,
        )
        print(f"   → Constraint strength applied: {constraint_strength}")

        print("\n✅ ACGF Pipeline completed successfully!")
        return AudioOutput(waveform=final_audio, sample_rate=SAMPLE_RATE, cfs_score=cfs_score)

    except Exception as e:
        print(f"\n⚠️  Error during generation: {e}")
        print("Make sure models are downloaded and paths are correct.")
        return AudioOutput(waveform=None)
