"""
ACGF Evaluation - Real metric computation (replaces deleted placeholder version)

This computes FAD, pitch-contour (prosody) similarity, and MFCC (timbre)
similarity from actual audio files, rather than np.random.uniform() stand-ins.

REQUIRES real audio pairs in samples/ - see "Evaluation Methodology Note" in
README.md for what's still missing (the Yoruba/Igbo reference + generated
pairs referenced below don't exist in this repo yet; this script will raise
a clear FileNotFoundError pointing at exactly that gap rather than silently
returning fake numbers).

FAD here uses fadtk (https://github.com/microsoft/fadtk) - install separately:
    pip install fadtk
fadtk is not bundled in requirements.txt because it pulls its own embedding
model dependencies (VGGish/CLAP/etc.) that are large and only needed for
evaluation, not for the generation pipeline itself.
"""

import os
import numpy as np
import librosa


def load_audio_pair(ref_path: str, gen_path: str, sample_rate: int = 44100):
    """Loads and validates a reference/generated audio pair. Raises a clear
    error rather than letting downstream metrics silently operate on missing
    or mismatched data."""
    for path in (ref_path, gen_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected audio file not found: {path}\n"
                "This evaluation script needs real reference + generated audio "
                "pairs. See README.md 'Evaluation Methodology Note' - these "
                "Yoruba/Igbo sample pairs are not yet present in samples/."
            )
    ref, _ = librosa.load(ref_path, sr=sample_rate, mono=True)
    gen, _ = librosa.load(gen_path, sr=sample_rate, mono=True)
    return ref, gen


def pitch_contour_similarity(ref_audio: np.ndarray, gen_audio: np.ndarray,
                              sample_rate: int = 44100) -> float:
    """
    Prosody similarity via pitch contour comparison (librosa.yin), DTW-aligned
    so the two contours don't need to be the same length.

    Returns a value in [0, 1]; higher = closer pitch contour match.
    """
    from librosa.sequence import dtw

    f0_ref = librosa.yin(ref_audio, fmin=65, fmax=1000, sr=sample_rate)
    f0_gen = librosa.yin(gen_audio, fmin=65, fmax=1000, sr=sample_rate)

    # Replace unvoiced NaNs with 0 so DTW has a well-defined distance to work with
    f0_ref = np.nan_to_num(f0_ref)
    f0_gen = np.nan_to_num(f0_gen)

    cost_matrix, _ = dtw(f0_ref.reshape(1, -1), f0_gen.reshape(1, -1))
    raw_distance = cost_matrix[-1, -1] / max(len(f0_ref), len(f0_gen))

    # Normalize by a reasonable max pitch range (Hz) so the result lands in [0, 1].
    # This normalization constant is a starting point, not validated against the
    # paper's actual eval set - calibrate it once you have your real corpus.
    max_expected_distance = 200.0
    similarity = 1.0 - min(raw_distance / max_expected_distance, 1.0)
    return round(float(similarity), 3)


def timbre_similarity(ref_audio: np.ndarray, gen_audio: np.ndarray,
                       sample_rate: int = 44100, n_mfcc: int = 20) -> float:
    """
    Timbre similarity via MFCC cosine similarity.
    Returns a value in [-1, 1]; higher = closer timbral match.
    """
    mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=sample_rate, n_mfcc=n_mfcc).mean(axis=1)
    mfcc_gen = librosa.feature.mfcc(y=gen_audio, sr=sample_rate, n_mfcc=n_mfcc).mean(axis=1)

    dot = np.dot(mfcc_ref, mfcc_gen)
    norm = np.linalg.norm(mfcc_ref) * np.linalg.norm(mfcc_gen)
    similarity = dot / norm if norm > 0 else 0.0
    return round(float(similarity), 3)


def compute_fad(ref_dir: str, gen_dir: str) -> float:
    """
    Frechet Audio Distance between a directory of reference clips and a
    directory of generated clips, via fadtk.

    NOTE: requires `pip install fadtk` separately (see module docstring).
    Lower FAD = more similar distributions (better).
    """
    try:
        from fadtk.fad import FrechetAudioDistance
        from fadtk.model_loader import VGGishModel
    except ImportError as e:
        raise ImportError(
            "fadtk is required for real FAD computation but isn't installed. "
            "Run: pip install fadtk\n"
            "(Not in requirements.txt - see module docstring for why.)"
        ) from e

    model = VGGishModel()
    fad = FrechetAudioDistance(model)
    score = fad.score(ref_dir, gen_dir)
    return round(float(score), 3)


def cultural_fidelity_score(d_conf: float, norm_fad: float, prosody_sim: float,
                             timbre_sim: float) -> float:
    """
    Cultural Fidelity Score, exact weights from paper §5.2:
        CFS = 0.4*D_conf + 0.25*(1-NormFAD) + 0.2*ProsodySim + 0.15*TimbreSim

    d_conf: discriminator confidence in [0,1] - NOT YET AVAILABLE, since
    discriminator/ has no trained model (see README). Pass a placeholder only
    if you understand the resulting CFS is not yet a real authenticity score.
    """
    w1, w2, w3, w4 = 0.4, 0.25, 0.2, 0.15
    return round(w1 * d_conf + w2 * (1 - norm_fad) + w3 * prosody_sim + w4 * timbre_sim, 3)


# Sample pairs this script expects to find under samples/ once the real
# Yoruba/Igbo corpus is added (see README "Evaluation Methodology Note").
PAIRS = [
    ("samples/ref_yoruba.wav", "samples/gen_yoruba_acgf.wav"),
    ("samples/ref_igbo.wav", "samples/gen_igbo_acgf.wav"),
]


if __name__ == "__main__":
    print("=== ACGF Evaluation - Real Metric Computation ===\n")
    print("NOTE: discriminator/ has no trained model yet, so D_conf below is a")
    print("      placeholder (0.5, i.e. maximally uncertain) - CFS values are")
    print("      NOT a real authenticity score until that's trained.\n")

    for ref_path, gen_path in PAIRS:
        print(f"Pair: {gen_path}")
        try:
            ref_audio, gen_audio = load_audio_pair(ref_path, gen_path)
        except FileNotFoundError as e:
            print(f"  ⚠️  Skipped: {e}\n")
            continue

        prosody = pitch_contour_similarity(ref_audio, gen_audio)
        timbre = timbre_similarity(ref_audio, gen_audio)

        try:
            fad = compute_fad(os.path.dirname(ref_path), os.path.dirname(gen_path))
        except ImportError as e:
            print(f"  ⚠️  FAD skipped: {e}")
            fad = None

        d_conf_placeholder = 0.5
        norm_fad = min(fad / 20.0, 1.0) if fad is not None else 0.5
        cfs = cultural_fidelity_score(d_conf_placeholder, norm_fad, prosody, timbre)

        print(f"  Prosody Similarity: {prosody}")
        print(f"  Timbre Similarity: {timbre}")
        print(f"  FAD: {fad if fad is not None else 'N/A (fadtk not installed)'}")
        print(f"  Cultural Fidelity Score (D_conf placeholder): {cfs}\n")

    print("Done. See README.md 'Evaluation Methodology Note' for what's still")
    print("needed before these numbers can support published Table 1/2 claims.")
