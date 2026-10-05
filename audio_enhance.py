"""
audio_enhance.py
-----------------
Post-extraction "studio" audio cleanup for the Luxury Studio Engine pipeline.

SCOPE: this module ONLY touches audio. It reads a raw extracted .wav (or an
externally pre-enhanced .wav) and writes a NEW, separate .wav file — it never
opens, decodes, re-encodes, or otherwise touches the video track. The caller
(face_track_crop.py) muxes the resulting file against the already-processed
video separately; nothing here re-encodes video.

Runs after audio has been extracted/passed through from the source clip and
before final mux, per the pipeline's existing flow.

CHAIN (raw path — no pre-enhanced input supplied), in order:
  1. Declip          - repair hard-clipped/distorted PA recordings (adeclip)
  2. Declick          - remove pops/impulsive clicks (adeclick)
  3. Denoise + enhance  - TIERED, best available wins (see DENOISE TIERS
                          note below):
                            Tier 1: DeepFilterNet3 (real neural speech
                                    enhancement model, local inference)
                            Tier 2: arnndn (RNNoise — older but still a
                                    genuine trained model)
                            Tier 3: afftdn + anlmdn (spectral fallback,
                                    always available, no extra deps)
  4. De-reverb          - room echo / tail reduction via WPE (Weighted
                          Prediction Error). Only runs on the Tier 2/3
                          path — see DE-REVERB note below for why it's
                          skipped after DeepFilterNet.
  5. High-pass            - sub-bass rumble & handling noise (~80Hz)
  6. Corrective EQ          - cut boxy/muddy 250-500Hz buildup, boost
                              2.5-4kHz presence
  7. Air / exciter            - harmonic exciter to restore perceived
                                high-end clarity
  8. De-esser                   - gentle sibilance control (after the
                                  presence boost, before compression — the
                                  boost can add sibilance, and de-essing
                                  before the compressor keeps sibilants from
                                  triggering extra gain-reduction pumping)
  9. Compression                  - even out quiet/loud phrases
                                    (acompressor)
  10. Loudness normalization        - -14 LUFS integrated / -1.5 dBTP,
                                      two-pass loudnorm

MANUAL PRE-PASS PATH (a `preenhanced_audio_path` is supplied — e.g. output
from Adobe Podcast's Enhance Speech run manually outside this pipeline,
because Demucs-style vocal-vs-instrumental separation can't split a speaker
from a background that is ITSELF vocals, like a nasheed): skips straight to
the final compression + loudness-normalization stage on that file, since an
external tool has already handled denoise/de-reverb. This remains the single
most reliable way to get literally-Adobe-Podcast-quality output — it runs
Adobe's own model instead of approximating it locally.

DENOISE TIERS NOTE: afftdn is spectral subtraction — it can't tell voice
from noise, only "steady" from "not steady", and it smears transients.
arnndn (RNNoise) is a real trained model and a solid step up, but it's a
small 2018-era network. DeepFilterNet3 (2022) is a materially stronger,
actively-benchmarked speech-enhancement model — closer in spirit to what a
tool like Adobe Podcast's Enhance Speech does, though still a different,
smaller model than Adobe's proprietary one. This module tries DeepFilterNet3
first and falls back down the tiers automatically if it's not usable in the
current environment, logging which tier actually ran.

DeepFilterNet3 needs PyTorch and is NOT installed by default — it's the one
part of this chain that requires an explicit one-time setup step:

    pip install "torch==2.2.2" "torchaudio==2.2.2" deepfilternet

Those exact versions matter: DeepFilterNet's current release (0.5.6) imports
`torchaudio.backend.common.AudioMetaData`, which newer torchaudio (2.3+)
removed — installing plain `pip install torch torchaudio deepfilternet`
today pulls the newest torchaudio and breaks at import time. This 2.2.2 pin
was verified end-to-end (denoise + resample round-trip back to the source
sample rate) at the time this was written. If a newer DeepFilterNet release
fixes the import, later versions may work too — this module doesn't pin
anything itself, it just tries the import and falls back cleanly if it
fails for any reason. Model weights ship inside the pip package itself, so
there's no separate network download/cache step like the arnndn tier below.
Expect a few GB of disk for torch + CUDA libraries even for CPU-only use
(PyPI's default index bundles them; there's no slim CPU wheel on the indexes
this pipeline can reach). Set USE_DEEPFILTERNET = False below to skip trying
it entirely (e.g. to save the dependency weight/load time and just use the
arnndn tier).

DE-REVERB NOTE: ffmpeg itself has no dereverberation filter. On the Tier
2/3 (non-DeepFilterNet) path, this module does real dereverberation via WPE
(nara_wpe — pip install nara_wpe), a well-established statistical algorithm
from speech-enhancement research that estimates and subtracts the late-
reverberant tail of a signal. It's skipped when DeepFilterNet3 ran: DFN3's
training data includes reverberant augmentation, so it already suppresses
a fair amount of room tail, and running a second, unrelated statistical
dereverb pass on its output tends to fight the first pass's assumptions and
add artifacts rather than help. Set honest expectations either way: reduces
room "tail"/boominess, doesn't work miracles on a genuinely bad room.

DEPENDENCIES: an ffmpeg build with adeclip / adeclick / afftdn / anlmdn /
arnndn / deesser / aexciter / astats / loudnorm support (present in
mainstream ffmpeg builds >= 4.3, confirmed available in this environment).
For DeepFilterNet3 (Tier 1), see the pip command above. For real de-reverb
on the fallback path, `pip install nara_wpe` (numpy/soundfile, no GPU).
Everything degrades gracefully without any of it — no torch installed ->
tries arnndn -> no network on first run -> afftdn+anlmdn; no nara_wpe ->
de-reverb is skipped. The rest of the chain always still runs.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# TUNABLE CONSTANTS — tune per-clip here rather than hardcoding inline below.
# ---------------------------------------------------------------------------

# Step 1: declip
DECLIP_ENABLED = True

# Step 2: declick
DECLICK_ENABLED = True

# Step 3: denoise/enhance tiers.
USE_DEEPFILTERNET = True   # Tier 1 — set False to skip straight to arnndn.

# afftdn/anlmdn + arnndn params are only used when DeepFilterNet is off,
# missing, or fails on a given clip.
DENOISE_NOISE_FLOOR_DB = -25   # more negative = more conservative afftdn
RNNOISE_MODEL_URL = "https://raw.githubusercontent.com/GregorR/rnnoise-models/master/beguiling-drafter-2018-08-30/bd.rnnn"
RNNOISE_MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "luxury_studio_engine", "models")
RNNOISE_MODEL_PATH = os.path.join(RNNOISE_MODEL_CACHE_DIR, "rnnoise_bd.rnnn")
RNNOISE_DOWNLOAD_TIMEOUT_SEC = 20

# Step 4: de-reverb (WPE). Only used on the Tier 2/3 (non-DeepFilterNet)
# path — see DE-REVERB NOTE above. Set DEREVERB_ENABLED = False to skip
# entirely even on that path.
DEREVERB_ENABLED = True
DEREVERB_TAPS = 10
DEREVERB_DELAY = 3
DEREVERB_ITERATIONS = 3
DEREVERB_STFT_SIZE = 512
DEREVERB_STFT_SHIFT = 128

# Step 5: high-pass
HIGHPASS_FREQ_HZ = 80

# Step 6: corrective EQ
EQ_CUT_FREQ_HZ = 350       # boxy/muddy cut, within the 250-500Hz band
EQ_CUT_WIDTH_Q = 1.5
EQ_CUT_GAIN_DB = -4
EQ_BOOST_FREQ_HZ = 3000    # presence boost, within the 2.5-4kHz band
EQ_BOOST_WIDTH_Q = 1.0
EQ_BOOST_GAIN_DB = 3

# Step 7: harmonic exciter ("air"). Synthesizes harmonics — see
# _try_ai_bandwidth_extension() below for the preferred-if-available path.
EXCITER_AMOUNT = 1.0
EXCITER_BLEND = 0.4
EXCITER_FREQ_HZ = 7500
# ffmpeg's aexciter enforces ceil >= 9999Hz — 9000 (the spec's nominal
# upper bound) is below its valid range and errors out, so this is pinned
# to the lowest value the filter actually accepts.
EXCITER_CEIL_HZ = 9999

# Step 8: de-esser. Kept gentle by default to avoid a lisped/dulled "s" —
# ffmpeg's deesser intensity runs 0-1.
DEESSER_ENABLED = True
DEESSER_INTENSITY = 0.3

# Step 9: compression
COMPRESSOR_THRESHOLD_DB = -18
COMPRESSOR_RATIO = 3
COMPRESSOR_ATTACK_MS = 5
COMPRESSOR_RELEASE_MS = 80
COMPRESSOR_MAKEUP_DB = 4

# Step 10: loudness normalization (Reels target)
LOUDNORM_TARGET_I = -14.0    # LUFS integrated
LOUDNORM_TARGET_TP = -1.5    # dBTP true-peak ceiling
LOUDNORM_TARGET_LRA = 11.0
LOUDNORM_TWO_PASS = True     # measure then apply, for accuracy

# Sync/verification
DURATION_TOLERANCE_SEC = 0.05   # drift beyond this gets corrected (pad/trim)

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _run(cmd, description):
    """Runs a subprocess command, returns (ok, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except FileNotFoundError:
        print(f"❌ [audio_enhance] ffmpeg/ffprobe not found on PATH — cannot run: {description}")
        return False, "", "binary not found"
    except Exception as e:
        print(f"❌ [audio_enhance] Error while {description}: {e}")
        return False, "", str(e)


def get_duration(path):
    """Returns duration in seconds via ffprobe, or None if it can't be read."""
    cmd = [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "json", path]
    ok, stdout, _ = _run(cmd, f"reading duration of {path}")
    if not ok:
        return None
    try:
        return float(json.loads(stdout)["format"]["duration"])
    except Exception:
        return None


def analyze_audio(path):
    """
    Quick before/after sanity-check stats: peak level (dB) and integrated
    loudness (LUFS). Best-effort — missing values are left as NaN rather
    than raising, since this is diagnostic output, not a pipeline gate.
    """
    stats = {"peak_db": float("nan"), "integrated_lufs": float("nan")}

    ok, _, stderr = _run(
        [FFMPEG_BIN, "-i", path, "-af", "astats=metadata=0:reset=1", "-f", "null", "-"],
        f"measuring peak level of {path}",
    )
    if ok:
        for line in stderr.splitlines():
            if "Peak level dB" in line:
                try:
                    stats["peak_db"] = float(line.strip().split(":")[-1])
                except ValueError:
                    pass

    ok, _, stderr = _run(
        [FFMPEG_BIN, "-i", path, "-af",
         f"loudnorm=I={LOUDNORM_TARGET_I}:TP={LOUDNORM_TARGET_TP}:LRA={LOUDNORM_TARGET_LRA}:print_format=json",
         "-f", "null", "-"],
        f"measuring integrated loudness of {path}",
    )
    if ok:
        try:
            json_text = stderr[stderr.rindex("{"): stderr.rindex("}") + 1]
            stats["integrated_lufs"] = float(json.loads(json_text)["input_i"])
        except Exception:
            pass

    return stats


# ---------------------------------------------------------------------------
# Tier 1 denoise/enhance: DeepFilterNet3 (neural, local inference)
# ---------------------------------------------------------------------------

_DFN_MODEL = None
_DFN_STATE = None


def _get_deepfilternet():
    """Lazily loads and caches the DeepFilterNet3 model for reuse across calls."""
    global _DFN_MODEL, _DFN_STATE
    if _DFN_MODEL is None:
        from df.enhance import init_df
        _DFN_MODEL, _DFN_STATE, _ = init_df()
    return _DFN_MODEL, _DFN_STATE


def _try_deepfilternet(input_path, output_path):
    """
    Runs DeepFilterNet3 on input_path and writes the enhanced result to
    output_path, resampled back to the INPUT's original sample rate (DFN
    operates internally at 48kHz) so it drops in transparently ahead of the
    rest of the ffmpeg chain. Returns True on success, False if the
    dependency is missing/incompatible or the pass fails for any reason —
    the caller falls back to arnndn in that case. See DENOISE TIERS NOTE
    in the module docstring for the required pinned install.
    """
    if not USE_DEEPFILTERNET:
        return False

    try:
        import soundfile as sf
        import torchaudio
        from df.enhance import enhance, load_audio, save_audio
    except ImportError as e:
        print(f"   [audio_enhance] DeepFilterNet not available ({e}) — using arnndn/afftdn tier instead.")
        return False

    try:
        start = time.time()
        model, df_state = _get_deepfilternet()
        orig_sr = sf.info(input_path).samplerate

        audio, _ = load_audio(input_path, sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)

        if df_state.sr() != orig_sr:
            enhanced = torchaudio.functional.resample(enhanced, df_state.sr(), orig_sr)

        save_audio(output_path, enhanced, orig_sr)
        print(f"   [audio_enhance] DeepFilterNet3 denoise done in {time.time() - start:.1f}s.")
        return True
    except Exception as e:
        print(f"   ⚠️ [audio_enhance] DeepFilterNet pass failed ({e}) — falling back to arnndn/afftdn.")
        return False


# ---------------------------------------------------------------------------
# Tier 2/3 denoise (ffmpeg): arnndn model download/cache, afftdn+anlmdn fallback
# ---------------------------------------------------------------------------

def _ensure_rnnoise_model():
    """
    Makes sure a trained RNNoise model (.rnnn) is available locally for the
    arnndn filter, downloading + caching it on first use if needed. Returns
    the local model path, or None if no model is available (disabled, or a
    download failure/no network on first run) so the caller can fall back
    to the afftdn+anlmdn combo instead. Never raises — a model problem
    should degrade quality, not crash the pipeline.
    """
    if RNNOISE_MODEL_PATH and os.path.exists(RNNOISE_MODEL_PATH) and os.path.getsize(RNNOISE_MODEL_PATH) > 0:
        return RNNOISE_MODEL_PATH

    if not RNNOISE_MODEL_URL:
        return None

    try:
        os.makedirs(RNNOISE_MODEL_CACHE_DIR, exist_ok=True)
        tmp_path = RNNOISE_MODEL_PATH + ".part"
        print(f"🎚️ [audio_enhance] Fetching RNNoise model for arnndn denoise "
              f"(one-time; cached at {RNNOISE_MODEL_PATH})...")
        req = urllib.request.Request(RNNOISE_MODEL_URL, headers={"User-Agent": "luxury-studio-engine/audio_enhance"})
        with urllib.request.urlopen(req, timeout=RNNOISE_DOWNLOAD_TIMEOUT_SEC) as resp, open(tmp_path, "wb") as f:
            shutil.copyfileobj(resp, f)
        shutil.move(tmp_path, RNNOISE_MODEL_PATH)
        return RNNOISE_MODEL_PATH
    except Exception as e:
        print(f"   ⚠️ [audio_enhance] Could not fetch RNNoise model ({e}) — "
              "falling back to afftdn+anlmdn denoise.")
        return None


def _denoise_filter():
    """
    Returns (filter_string, description) for the Tier 2/3 ffmpeg denoise
    step. Prefers arnndn with a real trained model; falls back to
    afftdn+anlmdn if no model is available.
    """
    model_path = _ensure_rnnoise_model()
    if model_path:
        return (f"arnndn=m={model_path}",
                f"denoise (arnndn neural model — {os.path.basename(model_path)})")
    return (f"afftdn=nf={DENOISE_NOISE_FLOOR_DB}:tn=1,anlmdn",
            f"denoise (afftdn nf={DENOISE_NOISE_FLOOR_DB}dB + anlmdn — "
            "neural model unavailable, spectral fallback)")


# ---------------------------------------------------------------------------
# De-reverb (WPE) — Tier 2/3 path only, operates on wav files
# ---------------------------------------------------------------------------

def apply_dereverb(input_path, output_path):
    """
    Runs WPE dereverberation on input_path and writes the result to
    output_path. Returns True on success, False if the step was skipped
    (disabled, dependency missing, or a runtime failure) — in which case
    the caller should keep using input_path unchanged for the next stage.
    """
    if not DEREVERB_ENABLED:
        print("   [audio_enhance] De-reverb disabled — skipping.")
        return False

    try:
        import numpy as np
        import soundfile as sf
        from nara_wpe.utils import istft, stft
        from nara_wpe.wpe import wpe
    except ImportError as e:
        print(f"   ⚠️ [audio_enhance] De-reverb needs the 'nara_wpe' package "
              f"(pip install nara_wpe) — not installed ({e}); skipping de-reverb.")
        return False

    try:
        start = time.time()
        y, sr = sf.read(input_path)
        mono = y.ndim == 1
        y = y[None, :] if mono else y.T  # -> (channels, T)

        Y = stft(y, size=DEREVERB_STFT_SIZE, shift=DEREVERB_STFT_SHIFT)   # (D, T, F)
        Y = Y.transpose(2, 0, 1)                                          # (F, D, T)
        Z = wpe(Y, taps=DEREVERB_TAPS, delay=DEREVERB_DELAY, iterations=DEREVERB_ITERATIONS)
        Z = Z.transpose(1, 2, 0)                                          # (D, T, F)
        z = istft(Z, size=DEREVERB_STFT_SIZE, shift=DEREVERB_STFT_SHIFT)

        z = z[0] if mono else z.T
        if np.isnan(z).any() or np.isinf(z).any():
            raise ValueError("WPE output contained NaN/Inf samples")

        sf.write(output_path, z, sr)
        print(f"   [audio_enhance] De-reverb (WPE) done in {time.time() - start:.1f}s.")
        return True
    except Exception as e:
        print(f"   ⚠️ [audio_enhance] De-reverb pass failed ({e}) — "
              "continuing with the pre-dereverb audio.")
        return False


# ---------------------------------------------------------------------------
# ffmpeg filter-chain construction
# ---------------------------------------------------------------------------

def _try_ai_bandwidth_extension():
    """
    Hook for a true AI bandwidth-extension / speech-enhancement model or API
    for the exciter step. aexciter (used by default) SYNTHESIZES harmonics —
    it does not recover content that genuinely isn't in the recording. If a
    real enhance-speech model/API is wired up in this environment later,
    call it here and return the resulting audio path or an equivalent
    filter; until then this always returns None and the caller falls back
    to aexciter. (Note: DeepFilterNet3, when active, already runs much
    earlier as the Tier 1 denoiser — this hook is for a *further* dedicated
    bandwidth-extension model, should one become available.)
    """
    return None


def _compressor_filter():
    return (f"acompressor=threshold={COMPRESSOR_THRESHOLD_DB}dB:ratio={COMPRESSOR_RATIO}:"
            f"attack={COMPRESSOR_ATTACK_MS}:release={COMPRESSOR_RELEASE_MS}:"
            f"makeup={COMPRESSOR_MAKEUP_DB}dB")


def build_stage_a_filter():
    """Pre-denoise ffmpeg stage: declip -> declick."""
    parts, applied = [], []

    if DECLIP_ENABLED:
        parts.append("adeclip")
        applied.append("declip (adeclip)")

    if DECLICK_ENABLED:
        parts.append("adeclick")
        applied.append("declick (adeclick — pops/impulsive noise)")

    return ",".join(parts), applied


def build_stage_c_filter():
    """Post-dereverb ffmpeg stage: high-pass -> EQ -> exciter -> de-esser -> compression."""
    parts, applied = [], []

    parts.append(f"highpass=f={HIGHPASS_FREQ_HZ}")
    applied.append(f"high-pass ({HIGHPASS_FREQ_HZ}Hz — rumble/handling noise)")

    parts.append(f"equalizer=f={EQ_CUT_FREQ_HZ}:t=q:w={EQ_CUT_WIDTH_Q}:g={EQ_CUT_GAIN_DB}")
    applied.append(f"EQ cut ({EQ_CUT_FREQ_HZ}Hz, {EQ_CUT_GAIN_DB}dB — de-box/mud)")

    parts.append(f"equalizer=f={EQ_BOOST_FREQ_HZ}:t=q:w={EQ_BOOST_WIDTH_Q}:g={EQ_BOOST_GAIN_DB}")
    applied.append(f"EQ boost ({EQ_BOOST_FREQ_HZ}Hz, +{EQ_BOOST_GAIN_DB}dB — presence/intelligibility)")

    ai_air = _try_ai_bandwidth_extension()
    if ai_air is not None:
        parts.append(ai_air)
        applied.append("air restoration (AI bandwidth-extension model)")
    else:
        parts.append(f"aexciter=amount={EXCITER_AMOUNT}:blend={EXCITER_BLEND}:"
                      f"freq={EXCITER_FREQ_HZ}:ceil={EXCITER_CEIL_HZ}")
        applied.append(f"air restoration (aexciter, {EXCITER_FREQ_HZ}-{EXCITER_CEIL_HZ}Hz — "
                        "synthesized harmonics, not recovered content)")

    if DEESSER_ENABLED:
        parts.append(f"deesser=i={DEESSER_INTENSITY}")
        applied.append(f"de-esser (gentle, intensity {DEESSER_INTENSITY})")

    parts.append(_compressor_filter())
    applied.append(f"compression (acompressor, ratio {COMPRESSOR_RATIO}:1, "
                    f"{COMPRESSOR_ATTACK_MS}ms attack / {COMPRESSOR_RELEASE_MS}ms release)")

    return ",".join(parts), applied


def _run_ffmpeg_filter(input_path, output_path, filter_chain, description):
    """Runs a single, plain ffmpeg -af pass (no loudnorm). Returns True/False."""
    cmd = [FFMPEG_BIN, "-y", "-i", input_path]
    if filter_chain:
        cmd += ["-af", filter_chain]
    cmd += [output_path]
    ok, _, stderr = _run(cmd, description)
    if not ok:
        last_err = stderr.strip().splitlines()[-1] if stderr and stderr.strip() else "unknown error"
        print(f"   ⚠️ [audio_enhance] {description} failed ({last_err}).")
    return ok


# ---------------------------------------------------------------------------
# Final stage: prior filters + loudnorm (two-pass measure/apply), with a
# minimal single-pass fallback if the full pass errors on this ffmpeg build.
# ---------------------------------------------------------------------------

def _measure_loudnorm_pass(input_path, prior_filters):
    measure_filter = (f"loudnorm=I={LOUDNORM_TARGET_I}:TP={LOUDNORM_TARGET_TP}:"
                       f"LRA={LOUDNORM_TARGET_LRA}:print_format=json")
    full_chain = f"{prior_filters},{measure_filter}" if prior_filters else measure_filter
    cmd = [FFMPEG_BIN, "-i", input_path, "-af", full_chain, "-f", "null", "-"]
    ok, _, stderr = _run(cmd, "measuring loudness (loudnorm pass 1)")
    if not ok:
        return None
    try:
        json_text = stderr[stderr.rindex("{"): stderr.rindex("}") + 1]
        return json.loads(json_text)
    except Exception:
        return None


def _apply_final_stage_with_loudnorm(input_path, output_path, filter_chain_sans_loudnorm):
    """Runs prior_filters + loudnorm (two-pass if configured) and writes output_path."""
    loudnorm_filter = f"loudnorm=I={LOUDNORM_TARGET_I}:TP={LOUDNORM_TARGET_TP}:LRA={LOUDNORM_TARGET_LRA}"

    if LOUDNORM_TWO_PASS:
        measured = _measure_loudnorm_pass(input_path, filter_chain_sans_loudnorm)
        if measured:
            loudnorm_filter = (
                f"loudnorm=I={LOUDNORM_TARGET_I}:TP={LOUDNORM_TARGET_TP}:LRA={LOUDNORM_TARGET_LRA}:"
                f"measured_I={measured.get('input_i')}:measured_TP={measured.get('input_tp')}:"
                f"measured_LRA={measured.get('input_lra')}:measured_thresh={measured.get('input_thresh')}:"
                f"offset={measured.get('target_offset')}:linear=true:print_format=summary"
            )
        else:
            print("   ⚠️ [audio_enhance] loudnorm measurement pass failed — using single-pass loudnorm instead.")

    full_chain = f"{filter_chain_sans_loudnorm},{loudnorm_filter}" if filter_chain_sans_loudnorm else loudnorm_filter

    cmd = [FFMPEG_BIN, "-y", "-i", input_path, "-af", full_chain, output_path]
    ok, _, stderr = _run(cmd, "applying final EQ/dynamics + loudness stage")
    if not ok:
        # A filter unsupported by this ffmpeg build shouldn't take down the
        # whole render — retry with just loudnorm before giving up. (Denoise
        # and de-reverb already happened upstream; this stage is
        # tone-shaping + loudness, so a minimal fallback here is a safe
        # last resort.)
        last_err = stderr.strip().splitlines()[-1] if stderr and stderr.strip() else "unknown error"
        print(f"   ⚠️ [audio_enhance] Full final-stage chain failed ({last_err}) — "
              "retrying with loudnorm only.")
        cmd = [FFMPEG_BIN, "-y", "-i", input_path, "-af", loudnorm_filter, output_path]
        ok, _, stderr = _run(cmd, "applying loudnorm-only fallback")

    return ok


def _verify_and_fix_duration(output_path, source_duration):
    """Ensures output_path's duration matches source_duration (sync-critical)."""
    if source_duration is None:
        return
    out_duration = get_duration(output_path)
    if out_duration is None:
        print("   ⚠️ [audio_enhance] Could not verify enhanced-audio duration.")
        return

    drift = out_duration - source_duration
    if abs(drift) <= DURATION_TOLERANCE_SEC:
        print(f"   ✅ [audio_enhance] Duration check OK ({out_duration:.3f}s vs source {source_duration:.3f}s).")
        return

    print(f"   ⚠️ [audio_enhance] Duration drift detected ({out_duration:.3f}s vs source "
          f"{source_duration:.3f}s) — correcting to match source so video sync isn't affected.")
    fixed_path = output_path + ".fixed.wav"
    if drift > 0:
        cmd = [FFMPEG_BIN, "-y", "-i", output_path, "-t", f"{source_duration:.6f}", fixed_path]
        ok, _, _ = _run(cmd, "trimming enhanced audio to match source duration")
    else:
        cmd = [FFMPEG_BIN, "-y", "-i", output_path, "-af",
               f"apad=whole_dur={source_duration:.6f}", fixed_path]
        ok, _, _ = _run(cmd, "padding enhanced audio to match source duration")

    if ok and os.path.exists(fixed_path):
        shutil.move(fixed_path, output_path)
    else:
        print("   ⚠️ [audio_enhance] Duration correction failed — using enhanced audio as-is (check sync manually).")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def enhance_audio(source_audio_path, output_audio_path, preenhanced_audio_path=None, enable=True):
    """
    Runs the studio audio-enhancement chain and returns the path to the
    result. Never overwrites source_audio_path or preenhanced_audio_path —
    always writes to output_audio_path (a new file), so the raw/pre-enhanced
    audio stays on disk for before/after comparison.

    source_audio_path        - raw (or vocal-isolated) extracted audio.
    output_audio_path         - where the enhanced result is written.
    preenhanced_audio_path     - optional path to audio already enhanced by
                                  an external tool (e.g. Adobe Podcast's
                                  Enhance Speech). If given, declip/declick/
                                  denoise/de-reverb/EQ/exciter/de-esser are
                                  skipped and only compression + loudness
                                  normalization run on it.
    enable                      - master on/off switch for the whole chain.
    """
    if not enable:
        print("🎚️ [audio_enhance] Enhancement chain disabled — copying source audio through unprocessed.")
        shutil.copy(source_audio_path, output_audio_path)
        return output_audio_path

    source_duration = get_duration(source_audio_path)
    tmp_dir = tempfile.mkdtemp(prefix="audio_enhance_")
    applied = []

    try:
        if preenhanced_audio_path:
            print(f"🎚️ [audio_enhance] Using externally pre-enhanced audio: {preenhanced_audio_path}")
            print("   Skipping declip/declick/denoise/de-reverb/EQ/exciter/de-esser — "
                  "running compression + loudness normalization only.")
            before_stats = analyze_audio(preenhanced_audio_path)
            working_source = preenhanced_audio_path
            final_filter_chain = _compressor_filter()
            applied.append(f"compression (acompressor, ratio {COMPRESSOR_RATIO}:1)")

        else:
            before_stats = analyze_audio(source_audio_path)

            # Stage A: declip -> declick (ffmpeg)
            stage_a_filter, stage_a_applied = build_stage_a_filter()
            if stage_a_filter:
                stage_a_out = os.path.join(tmp_dir, "stage_a.wav")
                if _run_ffmpeg_filter(source_audio_path, stage_a_out,
                                       stage_a_filter, "applying declip/declick (stage A)"):
                    applied += stage_a_applied
                    stage_a_result = stage_a_out
                else:
                    print("   ⚠️ [audio_enhance] Stage A failed — continuing with unprocessed source for later stages.")
                    stage_a_result = source_audio_path
            else:
                stage_a_result = source_audio_path

            # Denoise/enhance: Tier 1 DeepFilterNet3 -> Tier 2/3 arnndn/afftdn (ffmpeg)
            denoise_out = os.path.join(tmp_dir, "denoised.wav")
            used_deepfilternet = _try_deepfilternet(stage_a_result, denoise_out)
            if used_deepfilternet:
                applied.append("denoise + enhance (DeepFilterNet3 — neural speech enhancement, Tier 1)")
                denoise_result = denoise_out
            else:
                denoise_filter_str, denoise_desc = _denoise_filter()
                if _run_ffmpeg_filter(stage_a_result, denoise_out, denoise_filter_str,
                                       "applying denoise (ffmpeg tier)"):
                    applied.append(denoise_desc)
                    denoise_result = denoise_out
                else:
                    print("   ⚠️ [audio_enhance] Denoise failed entirely — continuing unprocessed.")
                    denoise_result = stage_a_result

            # De-reverb (WPE) — only on the Tier 2/3 path; see DE-REVERB NOTE.
            if used_deepfilternet:
                applied.append("de-reverb: SKIPPED (DeepFilterNet3 already ran — see DE-REVERB NOTE "
                                "in module docstring for why stacking WPE on top isn't done)")
                stage_b_result = denoise_result
            else:
                stage_b_out = os.path.join(tmp_dir, "stage_b_dereverbed.wav")
                if apply_dereverb(denoise_result, stage_b_out):
                    applied.append(f"de-reverb (WPE, {DEREVERB_TAPS} taps / {DEREVERB_ITERATIONS} iterations)")
                    stage_b_result = stage_b_out
                else:
                    applied.append("de-reverb: SKIPPED (see warning above — disabled, dependency missing, or pass failed)")
                    stage_b_result = denoise_result

            # Stage C filter chain (applied together with loudnorm below)
            stage_c_filter, stage_c_applied = build_stage_c_filter()
            applied += stage_c_applied
            working_source = stage_b_result
            final_filter_chain = stage_c_filter

        applied.append(f"loudness normalization (loudnorm, two-pass, target {LOUDNORM_TARGET_I} LUFS / "
                        f"{LOUDNORM_TARGET_TP} dBTP)")

        ok = _apply_final_stage_with_loudnorm(working_source, output_audio_path, final_filter_chain)
        if not ok:
            print("❌ [audio_enhance] Enhancement chain failed entirely — falling back to unprocessed source audio.")
            shutil.copy(source_audio_path, output_audio_path)
            return output_audio_path

        _verify_and_fix_duration(output_audio_path, source_duration)
        after_stats = analyze_audio(output_audio_path)

        print(f"🎚️ [audio_enhance] Filters applied: {', '.join(applied)}")
        print(f"   Before: peak {before_stats['peak_db']:.1f} dB, integrated loudness {before_stats['integrated_lufs']:.1f} LUFS")
        print(f"   After:  peak {after_stats['peak_db']:.1f} dB, integrated loudness {after_stats['integrated_lufs']:.1f} LUFS")
        print(f"   Raw/pre-enhancement audio preserved at: {source_audio_path}")

        return output_audio_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Standalone test/debug entry point — not used by the main pipeline.
    # Usage: python audio_enhance.py input.wav output.wav [preenhanced.wav]
    import sys
    if len(sys.argv) < 3:
        print("Usage: python audio_enhance.py <input.wav> <output.wav> [preenhanced.wav]")
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    pre_path = sys.argv[3] if len(sys.argv) > 3 else None
    enhance_audio(in_path, out_path, preenhanced_audio_path=pre_path)