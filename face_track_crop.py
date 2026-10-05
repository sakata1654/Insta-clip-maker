import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import sys
import os
import math
import random
import urllib.request
import shutil
import re
import platform
import subprocess
from datetime import datetime
from moviepy.editor import VideoFileClip, AudioFileClip
import audio_enhance  # local module: post-extraction "studio" audio cleanup chain

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("⚠️ WARNING: faster_whisper not installed. Subtitles will fail. Run: pip install faster-whisper")

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    HAS_ARABIC_SUPPORT = True
except ImportError:
    print("⚠️ WARNING: arabic-reshaper or python-bidi not installed. Arabic text may look broken.")
    HAS_ARABIC_SUPPORT = False

# ---------------------------------------------------------------------------
# TUNABLE CONSTANTS (new in this revision)
# ---------------------------------------------------------------------------
WHISPER_MODEL_CANDIDATES = [
    # (model_size, device, compute_type) — tried in order, first success wins.
    ("large-v3", "cuda", "int8_float16"),
    ("distil-large-v3", "cuda", "int8_float16"),
    ("medium", "cuda", "int8_float16"),
    ("medium", "cpu", "int8"),
]

MAX_SUBTITLE_LINES = 4
BASE_SUBTITLE_FONT_SIZE = 76
MIN_SUBTITLE_FONT_SIZE = 44
SUBTITLE_FONT_STEP = 4

SUBTITLE_FADE_DURATION = 0.35   # seconds
SUBTITLE_RISE_PX = 18           # how far subtitles rise into place while fading in

# --- Dual Subtitles: small original-language (e.g. Arabic) reference line ---
ORIGINAL_SUB_FONT_SIZE = 34          # small, fixed size — this line is a reference, not the main read (was 28)
ORIGINAL_SUB_MIN_FONT_SIZE = 22      # was 18
ORIGINAL_SUB_MAX_LINES = 2
ORIGINAL_SUB_MAX_WIDTH_FRACTION = 0.82   # narrower than the main subtitle wrap width
ORIGINAL_SUB_SHADOW_PAD = 14              # padding around the text before the shadow blur is applied
ORIGINAL_SUB_SHADOW_BLUR = 8              # gaussian blur radius on the shadow patch — soft "blur gradient" edge
ORIGINAL_SUB_SHADOW_ALPHA = 190           # 0-255 strength of the black shadow behind the text
ORIGINAL_SUB_GAP_ABOVE_MAIN = 10          # vertical gap between the original line and the main subtitle, when stacked

# --- Studio Card mode: vertical spacing between the card bottom and the
# speaker banner top. Centralized here so the card position, the banner
# position, and the original-language subtitle band all stay in sync
# instead of drifting out of alignment (previously this was hardcoded as a
# bare "55" in three separate places).
STUDIO_BANNER_GAP = 95   # was hardcoded 55 in 3 places

KEN_BURNS_MARGIN = 1.18         # extra buffer around the framed static image for pan/zoom room
KEN_BURNS_ZOOM_RANGE = 0.10     # ends ~10% more zoomed-in than the start framing
KEN_BURNS_PAN_FRACTION = 0.5    # how much of the available margin room is used for drift

# --- Simple Full-Bleed Mode: persistent heading text ---
HEADING_MAIN_FONT_SIZE = 74
HEADING_SUB_FONT_SIZE = 50
HEADING_MIN_FONT_SIZE = 34
HEADING_FONT_STEP = 4
HEADING_MAX_LINES_PER_SECTION = 3
HEADING_Y_FRACTION = 0.60        # where the heading block is vertically centered (0=top,1=bottom)
HEADING_SCRIM_PAD = 60           # extra vertical padding around the text for the dark scrim

# --- Simple Full-Bleed Mode: overall light black vertical gradient ---
# A subtle, persistent darkening toward the lower portion of the frame
# (matches the soft gradient seen in the reference screenshots) — sits
# behind the heading/subtitles for readability, independent of exactly
# where the heading text lands.
SIMPLE_GRADIENT_START_FRACTION = 0.58   # where the darkening starts to appear (moved higher, closer under the heading)
SIMPLE_GRADIENT_END_FRACTION = 0.74     # where it reaches its max strength (tighter span = more concentrated falloff)
SIMPLE_GRADIENT_MAX_ALPHA = 140         # 0-255, "mid" strength — noticeably darkens/blurs the feel without fully hiding anything underneath

# --- Manual framing window: max on-screen preview size --------------------
# Used by pick_manual_crop_position() and pick_gradient_settings() so the
# preview window is always fit-to-screen regardless of the source video's
# resolution or aspect ratio (portrait sources previously produced windows
# taller than the screen, with no way to resize them).
MAX_PREVIEW_DISPLAY_W = 1000
MAX_PREVIEW_DISPLAY_H = 800


def pick_image_file_dialog():
    """
    Opens a native file-browser window so the user can pick an image file
    from disk (used for the Static Image foreground mode). Falls back to a
    typed path if a GUI dialog can't be shown for any reason.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select image for the foreground card",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path.strip()
    except Exception as e:
        print(f"⚠️ Could not open the file browser ({e}).")
        return input("🖼️ Enter the full path to the image file instead: ").strip()

print("\n" + "="*60)
print("👑 LUXURY STUDIO ENGINE: STATIC CUT-SNAPPING (v26.22)!")
print("="*60)

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"

# Optional metadata forwarded by clip_downloader.bat: link, start, end, title
YT_LINK_ARG = sys.argv[3] if len(sys.argv) > 3 else ""
CLIP_START_ARG = sys.argv[4] if len(sys.argv) > 4 else ""
CLIP_END_ARG = sys.argv[5] if len(sys.argv) > 5 else ""
VIDEO_TITLE_ARG = sys.argv[6] if len(sys.argv) > 6 else ""

# --- ENTERPRISE AUTO-ARCHIVING SYSTEM ---
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
PROJECT_DIR = os.path.join(os.getcwd(), "Studio_Projects", f"Project_{timestamp}")
os.makedirs(PROJECT_DIR, exist_ok=True)
print(f"📁 Auto-Archive Directory: {PROJECT_DIR}")

# --- OUTPUT MODE SELECTION ---
print("\n🎛️ OUTPUT MODE:")
print("  [1] Studio Card   (rounded card, blurred background, color themes, speaker banner — default)")
print("  [2] Simple Crop   (manual camera positioning per cut, full-bleed video, optional persistent heading)")
output_mode_choice = input("👉 Enter choice (1/2) [Default is 1]: ").strip()
OUTPUT_MODE = "simple" if output_mode_choice == '2' else "studio"

# --- CARD STYLE SELECTION (Studio Card mode only) ---
CARD_W, CARD_H, CORNER_RADIUS, CARD_Y_OFFSET = 940, 1020, 40, 100
if OUTPUT_MODE == "studio":
    print("\n🔲 CARD STYLE:")
    print("  [1] Circle Frame    (compact perfect-circle card)")
    print("  [2] Classic Square   (original larger card, subtle corner rounding — default)")
    card_style_choice = input("👉 Enter choice (1/2) [Default is 2]: ").strip()
    if card_style_choice == '1':
        CARD_W = CARD_H = 760
        CORNER_RADIUS = CARD_W // 2
        CARD_Y_OFFSET = 200
    else:
        CARD_W, CARD_H, CORNER_RADIUS, CARD_Y_OFFSET = 940, 1020, 40, 100

# --- THE USER PROMPTS ---
FOREGROUND_MODE = "video"
STATIC_IMAGE_PATH = ""

if OUTPUT_MODE == "studio":
    print("\n🖼️ FOREGROUND CONTENT:")
    print("  [1] Video Frame   (auto face-tracked/cropped from the input video — default)")
    print("  [2] Static Image  (pick a separate photo to display in the card instead of the video)")
    fg_choice = input("👉 Enter choice (1/2) [Default is 1]: ").strip()
    FOREGROUND_MODE = "image" if fg_choice == '2' else "video"

    if FOREGROUND_MODE == "image":
        print("📂 Opening the file browser — choose the image to display in the card...")
        STATIC_IMAGE_PATH = pick_image_file_dialog()
        if not STATIC_IMAGE_PATH or not os.path.exists(STATIC_IMAGE_PATH):
            print("⚠️ No valid image selected — falling back to Video Frame mode.")
            FOREGROUND_MODE = "video"
        else:
            print(f"✅ Image selected: {STATIC_IMAGE_PATH}")
            print("   (You'll be asked to frame/zoom it manually in a moment, once the video loads.)")

user_input_subs = input("\n💬 Generate and Intercept AI Subtitles for manual translation? (y/n): ").strip().lower()
ENABLE_SUBTITLES = user_input_subs in ['y', 'yes']

WHISPER_LANG = None
ENABLE_DUAL_SUBTITLES = False
if ENABLE_SUBTITLES:
    print("\n🗣️ NATIVE AUDIO LANGUAGE (To generate accurate reference text):")
    print("  [1] Urdu")
    print("  [2] Hindi")
    print("  [3] Arabic")
    print("  [4] Auto-Detect")
    lang_choice = input("👉 Enter choice (1/2/3/4) [Default is 1]: ").strip()
    
    if lang_choice == '2': WHISPER_LANG = "hi"
    elif lang_choice == '3': WHISPER_LANG = "ar"
    elif lang_choice == '4': WHISPER_LANG = None
    else: WHISPER_LANG = "ur"

    user_input_dual = input("\n🈴 Also show a small original-language subtitle line (e.g. Arabic) alongside the translation? (y/n): ").strip().lower()
    ENABLE_DUAL_SUBTITLES = user_input_dual in ['y', 'yes']

ENABLE_VOCAL_ISOLATION = False
DEMUCS_MODEL = "htdemucs"
user_input_vocals = input("\n🎚️ Strip background music/nasheed so mainly the speaker's voice is heard? (y/n): ").strip().lower()
ENABLE_VOCAL_ISOLATION = user_input_vocals in ['y', 'yes']
if ENABLE_VOCAL_ISOLATION:
    print("   Runs an AI source-separation pass (Demucs) on the audio before subtitles/final mix.")
    print("   Works best on instrumental background music. A background nasheed that's itself")
    print("   pure vocals is a much harder voice-vs-voice split — expect it reduced, not perfectly gone.")
    print("   Model: [1] htdemucs     (fast, balanced — default)")
    print("          [2] htdemucs_ft  (slower, 4 passes — usually cleaner separation)")
    demucs_choice = input("   👉 Enter choice (1/2) [Default is 1]: ").strip()
    DEMUCS_MODEL = "htdemucs_ft" if demucs_choice == '2' else "htdemucs"

# --- STUDIO AUDIO ENHANCEMENT ---
# Declip/denoise/EQ/air/de-reverb/compression/loudnorm chain, run on the
# extracted audio before final mux. See audio_enhance.py. Defaults ON.
user_input_audio_enhance = input(
    "\n🎚️ Apply studio audio enhancement (declip, denoise, EQ, air, compression, "
    "loudness normalize)? (Y/n): "
).strip().lower()
ENABLE_AUDIO_ENHANCE = user_input_audio_enhance not in ['n', 'no']

PREENHANCED_AUDIO_PATH = ""
if ENABLE_AUDIO_ENHANCE:
    user_input_preenhanced = input(
        "   Already have a speaker-isolated/enhanced audio file for this clip (e.g. from an\n"
        "   external tool like Adobe Podcast's Enhance Speech), to use instead of running the\n"
        "   full chain on the raw audio? Enter its path, or press Enter to skip: "
    ).strip()
    if user_input_preenhanced and os.path.exists(user_input_preenhanced):
        PREENHANCED_AUDIO_PATH = user_input_preenhanced
        print("   ✅ Will skip straight to compression + loudness normalization on the provided file.")
    elif user_input_preenhanced:
        print(f"   ⚠️ Path not found ({user_input_preenhanced}) — running the full chain on the raw audio instead.")

ENABLE_COLOR_SHIFT = False
if OUTPUT_MODE == "studio":
    user_input_shift = input("\n🎨 Enable Dynamic Multi-Act Color Shifting across the video timeline? (y/n): ").strip().lower()
    ENABLE_COLOR_SHIFT = user_input_shift in ['y', 'yes']

CAMERA_MODE = "static"
if OUTPUT_MODE == "simple":
    # Simple Crop mode is manual-positioning by definition — you frame the
    # shot yourself and get re-prompted on every detected cut.
    CAMERA_MODE = "manual"
    print("\n🎥 CAMERA MODE: Manual Positioning (you'll frame the shot now, and again on every cut).")
elif FOREGROUND_MODE == "image":
    # The card shows a fixed static image now, so live face-tracking camera
    # modes don't apply — quietly default without asking.
    CAMERA_MODE = "static"
else:
    print("\n🎥 CAMERA MODE:")
    print("  [1] Active Face Tracking   (smooth follow + auto re-center on cuts)")
    print("  [2] Static, Auto Re-center (locks between cuts, snaps to face on each cut)")
    print("  [3] Manual Positioning     (you place the camera, and choose again on every cut)")
    track_choice = input("👉 Enter choice (1/2/3) [Default is 2]: ").strip()

    if track_choice == '1':
        CAMERA_MODE = "active"
    elif track_choice == '3':
        CAMERA_MODE = "manual"
    else:
        CAMERA_MODE = "static"

ACTIVE_TRACKING = (CAMERA_MODE == "active")
MANUAL_TRACKING = (CAMERA_MODE == "manual")

# --- PERSISTENT HEADING (Simple Crop mode only) ---
ENABLE_HEADING = False
HEADING_TEXT = ""
if OUTPUT_MODE == "simple":
    user_input_heading = input("\n🏷️ Add a persistent heading text shown throughout the video? (y/n): ").strip().lower()
    ENABLE_HEADING = user_input_heading in ['y', 'yes']
    if ENABLE_HEADING:
        HEADING_TEXT = input("✏️ Enter Heading Text (use '|' for a second, smaller line): ").strip()
        if not HEADING_TEXT:
            ENABLE_HEADING = False

USER_WATERMARK = input("©️ Enter Permanent Bottom Watermark Stamp (e.g. markazussunnah) (Press Enter to skip): ").strip()

SPEAKER_NAME = ""
if OUTPUT_MODE == "studio":
    SPEAKER_NAME = input("🎙️ Enter Speaker Name (Mixed English/Urdu supported. Use '|' for sub-lines): ").strip()

user_input_caption = input("\n📝 Generate an Instagram Reel caption file? (y/n): ").strip().lower()
ENABLE_CAPTION = user_input_caption in ['y', 'yes']

VIDEO_TITLE, YT_LINK, CLIP_START, CLIP_END, CAPTION_EXPLANATION = "", "", "", "", ""
if ENABLE_CAPTION:
    VIDEO_TITLE = VIDEO_TITLE_ARG
    YT_LINK = YT_LINK_ARG
    CLIP_START = CLIP_START_ARG
    CLIP_END = CLIP_END_ARG

    # Fallback prompts only fire if this was run standalone (not via clip_downloader.bat)
    if not YT_LINK:
        YT_LINK = input("  🔗 Original YouTube Link (not received from downloader): ").strip()
    if not CLIP_START:
        CLIP_START = input("  ⏱️  Clip Start Timestamp (not received from downloader): ").strip()
    if not CLIP_END:
        CLIP_END = input("  ⏱️  Clip End Timestamp (not received from downloader): ").strip()
    if not VIDEO_TITLE:
        VIDEO_TITLE = input("  🎬 Video Title (not received from downloader): ").strip()

    print("\n📝 CAPTION DETAILS (auto-filled from downloader):")
    print(f"  🎬 Title: {VIDEO_TITLE or '(none)'}")
    print(f"  🔗 Link:  {YT_LINK or '(none)'}")
    print(f"  ⏱️  Range: {CLIP_START or '?'} – {CLIP_END or '?'}")
    print("  📝 Explanation will be auto-built from the subtitle transcript once translation is done.")

print("="*60 + "\n")

# --- MODEL & FONT DOWNLOADER ---
MODEL_PATH = "blaze_face_short_range.tflite"
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading Face Detector model...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", MODEL_PATH)

URDU_FONT_PATH = "Amiri-Bold.ttf"
for old_font in [URDU_FONT_PATH, "NotoNaskhArabic-Bold.ttf"]:
    if os.path.exists(old_font) and os.path.getsize(old_font) < 50000:
        os.remove(old_font)

if not os.path.exists(URDU_FONT_PATH):
    print("📥 Downloading Premium Font (Amiri)...")
    font_urls = [
        "https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Bold.ttf",
        "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Bold.ttf"
    ]
    for url in font_urls:
        try:
            urllib.request.urlretrieve(url, URDU_FONT_PATH)
            if os.path.getsize(URDU_FONT_PATH) > 50000: break
        except Exception: pass

def format_srt_time(seconds):
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def parse_srt_time(time_str):
    time_str = time_str.replace('.', ',')
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

def read_intercepted_srt(filename):
    subs = []
    if not os.path.exists(filename): return subs
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\s*\n', content.strip())
    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) >= 3:
            time_line = lines[1]
            if '-->' in time_line:
                try:
                    start_str, end_str = time_line.split('-->')
                    start = parse_srt_time(start_str.strip())
                    end = parse_srt_time(end_str.strip())
                    text = " ".join(lines[2:])
                    subs.append({'start': start, 'end': end, 'text': text})
                except Exception:
                    pass
    return subs

def blend_colors(c1, c2, factor):
    return tuple(int(a + (b - a) * factor) for a, b in zip(c1, c2))

class UltimateDirector:
    def __init__(self):
        self.TARGET_W = 1080
        self.TARGET_H = 1920
        self.CROP_RATIO = 0.92  
        self.CORNER_RADIUS = 40 
        
        self.CARD_W = 940       
        self.CARD_H = 1020 
        self.x_offset = (self.TARGET_W - self.CARD_W) // 2
        self.y_offset = 120  

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.3)
        self.detector = vision.FaceDetector.create_from_options(options)
        
        self.cam_x, self.cam_y = None, None
        self.manual_cam_pos = None
        self.manual_zoom = 1.0
        self.duration = 1.0
        self.enable_color_shift = ENABLE_COLOR_SHIFT
        self.themes = []

        # Output mode: "studio" is the original rounded-card / blurred
        # background / color-theme pipeline. "simple" is a full-bleed crop
        # driven entirely by manual camera positioning, with an optional
        # persistent heading overlay instead of the card/banner.
        self.output_mode = "studio"
        self.enable_heading = False
        self.heading_sections = []   # list of {"text":..., "font_size":...} built at init
        self.heading_cache = None    # cached wrapped/shaped lines+fonts, rebuilt whenever the framing changes
        self.heading_y_fraction = HEADING_Y_FRACTION  # where the heading is centered vertically — set via the manual picker
        self.heading_size_scale = 1.0  # multiplier on HEADING_MAIN/SUB_FONT_SIZE — set via the manual picker
        self.heading_needs_placement = True  # set True on init and after every reframe/cut

        # Dual Subtitles: small original-language (e.g. Arabic) reference
        # line shown alongside the main translated subtitle. Font object is
        # built once in the init block of each render path.
        self.enable_dual_subs = False
        self.original_sub_font = None
        self.active_original_sub_id = None
        self.cached_original_sub_lines = []

        # Static Image foreground mode: "video" crops/tracks the source clip
        # as before; "image" instead animates a pre-framed static photo with
        # a slow Ken Burns pan/zoom, and builds its own blurred backdrop from
        # the photo (rather than the source video frame) so a letterboxed or
        # otherwise black source clip never shows through behind the card.
        # Audio/subtitles still come from the original input video either way.
        self.foreground_mode = "video"
        self.static_fg_image_full = None   # oversized cached crop, gives Ken Burns room to move
        self.static_bg_small = None        # pre-blurred quarter-res backdrop built from the photo
        
        self.prev_frame_gray = None

        # Simple Crop mode: cached full-frame light black vertical gradient
        # overlay (built once in render_simple_frame's init block).
        self.simple_gradient_overlay = None
        # Default gradient settings — overridden by pick_gradient_settings()
        # in __main__ when the user dials them in visually during setup.
        self.simple_gradient_start_frac = SIMPLE_GRADIENT_START_FRACTION
        self.simple_gradient_end_frac = SIMPLE_GRADIENT_END_FRACTION
        self.simple_gradient_max_alpha = SIMPLE_GRADIENT_MAX_ALPHA

        # Simple Crop mode: optional custom overlay PNG (e.g. a pre-made
        # gradient/vignette graphic) composited on top of every frame, on
        # top of the generated gradient (if any) and underneath the
        # heading/watermark/subtitles. Stored as a derived pure-black shadow
        # mask (see build_shadow_mask_from_image) — only the dark part of
        # the source graphic survives; light/white areas are transparent.
        # Pre-scaled to the full TARGET_W x TARGET_H canvas once at setup
        # time in __main__ — set to None to skip it entirely.
        self.overlay_png_img = None
        
        if ENABLE_SUBTITLES:
            print("🧠 Loading AI Transcription Model...")
            self.whisper_model = None
            self.whisper_model_size = None
            for size, device, compute in WHISPER_MODEL_CANDIDATES:
                try:
                    print(f"   → trying '{size}' on {device} ({compute})...")
                    self.whisper_model = WhisperModel(size, device=device, compute_type=compute)
                    self.whisper_model_size = size
                    print(f"✅ Loaded Whisper '{size}' on {device}.")
                    break
                except Exception as e:
                    print(f"   ⚠️ '{size}' on {device} failed: {e}")
            if self.whisper_model is None:
                print("❌ Could not load any Whisper model — subtitles will fail.")
        else:
            self.whisper_model = None
            self.whisper_model_size = None
            
        self.current_subs = []
        self.fps = 30.0
        self.main_frame_count = 0
        self.is_initialized = False
        
        self.active_sub_id = None
        self.cached_sub_lines = []
        self.cached_sub_font = None
        
        self.b_w, self.b_h, self.banner_radius = 0, 0, 0
        self.speaker_banner_mask = None
        self.speaker_lines = []
        self.font_main = None
        self.font_sub = None
        self.branding_img = None
        
        self.generate_grain_pool()

    def clean_and_shape_harakat(self, text):
        if not text: return ""
        sanitized = re.sub(r'\s+([\u064B-\u0652])', r'\1', text)
        sanitized = re.sub(r'([\u064E\u064F\u0610])([\u0651])', r'\2\1', sanitized)
        
        if HAS_ARABIC_SUPPORT and any("\u0600" <= c <= "\u06FF" for c in sanitized):
            reshaped = arabic_reshaper.reshape(sanitized)
            return get_display(reshaped)
        return sanitized

    def generate_grain_pool(self):
        self.grain_pool = []
        for _ in range(8):
            noise = np.random.normal(0, 6, (self.TARGET_H, self.TARGET_W))
            noise_rgb = np.stack([noise]*3, axis=2).astype(np.float32)
            self.grain_pool.append(noise_rgb)

    def generate_vertical_gradient(self):
        mask = np.zeros((self.TARGET_H, self.TARGET_W), dtype=np.float32)
        start_y, end_y = int(self.TARGET_H * 0.55), int(self.TARGET_H * 0.88)
        for y in range(self.TARGET_H):
            if y < start_y: mask[y, :] = 0.0
            elif y > end_y: mask[y, :] = 0.98
            else:
                progress = (y - start_y) / (end_y - start_y)
                mask[y, :] = (progress * progress * (3 - 2 * progress)) * 0.98
        self.vertical_gradient_mask = np.expand_dims(mask, axis=2)

    def build_simple_gradient_overlay(self):
        """
        Simple Crop mode: builds a cached full-frame RGBA overlay with a
        soft black vertical gradient — invisible above the band, then
        ramping up to a "mid" strength darkening (noticeable, slightly
        obscuring, but not opaque/hiding) over a short, compact vertical
        span rather than spreading across most of the lower frame. Built
        once per run; composited onto every frame in render_simple_frame,
        underneath the heading, watermark, and subtitles so those stay
        fully legible on top of it. If simple_gradient_max_alpha is 0
        (density dialed all the way down), this is skipped entirely so no
        gradient layer is composited at all.
        """
        if self.simple_gradient_max_alpha <= 0:
            self.simple_gradient_overlay = None
            return

        mask = np.zeros((self.TARGET_H, self.TARGET_W), dtype=np.float32)
        start_y = int(self.TARGET_H * self.simple_gradient_start_frac)
        end_y = int(self.TARGET_H * self.simple_gradient_end_frac)
        for y in range(self.TARGET_H):
            if y < start_y:
                mask[y, :] = 0.0
            elif y > end_y:
                mask[y, :] = 1.0
            else:
                progress = (y - start_y) / max(1, (end_y - start_y))
                mask[y, :] = progress * progress * (3 - 2 * progress)  # smoothstep

        alpha = (mask * self.simple_gradient_max_alpha).astype(np.uint8)
        overlay_arr = np.zeros((self.TARGET_H, self.TARGET_W, 4), dtype=np.uint8)
        overlay_arr[:, :, 3] = alpha  # black (0,0,0) at every pixel, alpha only varies
        self.simple_gradient_overlay = Image.fromarray(overlay_arr, mode='RGBA')

    def get_font(self, size, bold=True, is_subtitle=False):
        if is_subtitle:
            fallbacks = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/tahomabd.ttf"]
            for path in fallbacks:
                try: return ImageFont.truetype(path, int(size))
                except IOError: continue
        if os.path.exists(URDU_FONT_PATH) and os.path.getsize(URDU_FONT_PATH) > 50000:
            try: return ImageFont.truetype(URDU_FONT_PATH, int(size))
            except Exception: pass
        return ImageFont.load_default()

    def generate_theme_profiles(self, clip):
        print("\n🎨 Analyzing video to generate 5 unique vibrant color profiles...")
        default_color = (20, 15, 135) 
        try:
            sample_times = [clip.duration * 0.15, clip.duration * 0.50, clip.duration * 0.85]
            pixels = []
            for t in sample_times:
                frame = clip.get_frame(min(t, clip.duration - 0.1))
                h, w, _ = frame.shape
                core = frame[h//4:3*h//4, w//4:3*w//4]
                resized = cv2.resize(core, (30, 30), interpolation=cv2.INTER_AREA)
                pixels.append(resized.reshape(-1, 3))
            
            combined_pixels = np.vstack(pixels).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
            _, _, centers = cv2.kmeans(combined_pixels, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            valid_centers = []
            for center in centers:
                b, g, r = center
                brightness = 0.299*r + 0.587*g + 0.114*b
                if 20 < brightness < 240: valid_centers.append(center)
            
            valid_centers = sorted(valid_centers, key=lambda c: cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2HSV)[0][0][1], reverse=True)
            
            while len(valid_centers) < 5:
                base = valid_centers[0] if valid_centers else default_color
                base_hsv = cv2.cvtColor(np.uint8([[base]]), cv2.COLOR_RGB2HSV)[0][0]
                new_hsv = base_hsv.copy()
                new_hsv[0] = (new_hsv[0] + (len(valid_centers) * 36)) % 180 
                valid_centers.append(cv2.cvtColor(np.uint8([[new_hsv]]), cv2.COLOR_HSV2RGB)[0][0])
            
            themes = []
            for rgb in valid_centers[:5]:
                hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
                hsv_bg = hsv.copy()
                hsv_bg[1] = max(160, min(int(hsv_bg[1]) + 60, 255)) 
                hsv_bg[2] = max(70, min(int(hsv_bg[2]), 140))        
                bg_bgr = cv2.cvtColor(np.uint8([[hsv_bg]]), cv2.COLOR_HSV2BGR)[0][0]
                
                hsv_banner = hsv_bg.copy()
                hsv_banner[2] = max(50, int(hsv_banner[2]) - 20)
                banner_bgr = cv2.cvtColor(np.uint8([[hsv_banner]]), cv2.COLOR_HSV2BGR)[0][0]
                
                hsv_accent = hsv.copy()
                hsv_accent[0] = (hsv_accent[0] + 12) % 180  
                hsv_accent[1] = max(35, min(hsv_accent[1], 55))  
                hsv_accent[2] = max(220, min(hsv_accent[2], 250)) 
                accent_rgb = cv2.cvtColor(np.uint8([[hsv_accent]]), cv2.COLOR_HSV2RGB)[0][0]
                
                themes.append({
                    "bg": tuple(int(c) for c in bg_bgr),
                    "accent": (int(accent_rgb[0]), int(accent_rgb[1]), int(accent_rgb[2]), 255),
                    "banner": (int(banner_bgr[0]), int(banner_bgr[1]), int(banner_bgr[2]), 245)
                })
            self.themes = themes
            return themes
        except Exception:
            themes = [{"bg": default_color, "accent": (255,215,0,255), "banner": (15,10,100,245)}] * 5
            self.themes = themes
            return themes

    def semantic_chunker(self, raw_subs):
        chunked = []
        for sub in raw_subs:
            text = sub['text']
            original_text = sub.get('original', '')
            if '|' in text:
                parts = [p.strip() for p in text.split('|') if p.strip()]
            else:
                parts = [text] 
            
            total_words = sum(len(p.split()) for p in parts)
            duration = sub['end'] - sub['start']
            current_time = sub['start']
            
            for part in parts:
                word_c = len(part.split())
                time_share = duration * (word_c / max(1, total_words))
                chunked.append({
                    'start': current_time,
                    'end': current_time + time_share,
                    'text': part,
                    # Every split part of this block shares the same original-
                    # language reference text — we don't have per-word timing
                    # for the original transcript, only per-block.
                    'original': original_text,
                })
                current_time += time_share
                
        return chunked

    def wrap_and_shape_text_cache(self, text, draw, font, max_width=None):
        if max_width is None:
            max_width = self.TARGET_W - 240
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            disp = self.clean_and_shape_harakat(test_line)
            w = draw.textlength(disp, font=font)
            if w <= max_width: 
                current_line.append(word)
            else:
                if current_line: 
                    lines.append(self.clean_and_shape_harakat(" ".join(current_line)))
                    current_line = [word]
                else: 
                    lines.append(self.clean_and_shape_harakat(word))
                    current_line = []
        if current_line: 
            lines.append(self.clean_and_shape_harakat(" ".join(current_line)))
        return lines

    def fit_subtitle_font_and_lines(self, text, draw, max_lines=MAX_SUBTITLE_LINES,
                                     start_size=BASE_SUBTITLE_FONT_SIZE, min_size=MIN_SUBTITLE_FONT_SIZE):
        """
        Shrinks the subtitle font in steps until the wrapped text fits within
        max_lines (default: 3-4 lines). Never goes below min_size; if the
        text still doesn't fit at the smallest size, it's returned as-is
        (extra lines) rather than shrinking further into unreadable territory
        — that's the signal to add a manual '|' split in the SRT.
        """
        size = start_size
        font, lines = None, None
        while size >= min_size:
            font = self.get_font(size, bold=True, is_subtitle=True)
            lines = self.wrap_and_shape_text_cache(text, draw, font)
            if len(lines) <= max_lines:
                return font, lines
            size -= SUBTITLE_FONT_STEP
        print(f"⚠️ Subtitle block still exceeds {max_lines} lines at the smallest font — "
              f"consider adding a '|' split for: \"{text[:50]}...\"")
        return font, lines

    def draw_cached_subtitles(self, draw, shaped_lines, font, accent_color, opacity=1.0, rise=0,
                               top_bound=None, bottom_bound=None):
        line_height = int(font.size * 1.3)
        total_text_height = len(shaped_lines) * line_height

        if top_bound is not None and bottom_bound is not None:
            available_top = top_bound
            available_bottom = bottom_bound
        else:
            # Dynamic bounds: use the ACTUAL banner height (self.b_h) rather than
            # a hardcoded value, and shrink the reserved zones to near-zero when
            # the speaker banner / watermark aren't in use. Then hard-clamp the
            # block so it can never bleed into either neighbor's space, no
            # matter how many lines this particular chunk wraps into.
            banner_bottom_y = self.y_offset + self.CARD_H + STUDIO_BANNER_GAP + (self.b_h if SPEAKER_NAME else 0)
            watermark_top_y = self.TARGET_H - (150 if USER_WATERMARK else 50)
            top_pad, bottom_pad = 24, 24
            available_top = banner_bottom_y + top_pad
            available_bottom = watermark_top_y - bottom_pad

        available_height = max(10, available_bottom - available_top)

        start_y = available_top + (available_height - total_text_height) // 2
        start_y = max(available_top, min(start_y, available_bottom - total_text_height))

        alpha_text = int(220 * opacity)
        alpha_stroke = int(255 * opacity)
        stroke_color = (accent_color[0], accent_color[1], accent_color[2], alpha_stroke)

        for i, disp in enumerate(shaped_lines):
            total_w = draw.textlength(disp, font=font)
            curr_x = (self.TARGET_W - total_w) / 2
            curr_y = start_y + (i * line_height) + rise
            
            draw.text((curr_x + 3, curr_y + 3), disp, font=font, fill=(0, 0, 0, alpha_text))
            draw.text((curr_x, curr_y), disp, font=font, fill=stroke_color, stroke_width=2, stroke_fill=(0, 0, 0, alpha_stroke))
        return start_y, total_text_height

    def fit_original_sub_lines(self, text, draw):
        """
        Dual Subtitles: wraps/fits the small original-language reference
        line (e.g. Arabic) into at most ORIGINAL_SUB_MAX_LINES, shrinking
        the font a little if needed. Kept fixed and small by design — this
        line is a reference, not the primary read.
        """
        size = ORIGINAL_SUB_FONT_SIZE
        font, lines = None, None
        max_width = int(self.TARGET_W * ORIGINAL_SUB_MAX_WIDTH_FRACTION)
        while size >= ORIGINAL_SUB_MIN_FONT_SIZE:
            font = self.get_font(size, bold=True, is_subtitle=False)
            lines = self.wrap_and_shape_text_cache(text, draw, font, max_width=max_width)
            if len(lines) <= ORIGINAL_SUB_MAX_LINES:
                return font, lines
            size -= 2
        return font, lines

    def draw_original_subtitle_block(self, pil_final, shaped_lines, font, center_y):
        """
        Dual Subtitles: draws the small white original-language line(s)
        vertically centered on center_y, with a soft black blurred "shadow
        patch" sized tightly to the text itself (not a full-width band) —
        built by rendering the shadow on its own small transient canvas,
        gaussian-blurring it, then pasting only that patch behind the text.
        """
        if not shaped_lines:
            return pil_final

        temp_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
        line_height = int(font.size * 1.25)
        line_widths = [temp_draw.textlength(line, font=font) for line in shaped_lines]
        block_w = int(max(line_widths)) if line_widths else 0
        block_h = line_height * len(shaped_lines)

        pad = ORIGINAL_SUB_SHADOW_PAD
        patch_w = block_w + pad * 2
        patch_h = block_h + pad * 2

        # Build the shadow patch on its own small canvas so the blur stays
        # tight around the text instead of smearing across the full frame.
        shadow_patch = Image.new('RGBA', (patch_w, patch_h), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_patch)
        shadow_draw.rounded_rectangle(
            (pad * 0.3, pad * 0.3, patch_w - pad * 0.3, patch_h - pad * 0.3),
            radius=int(pad * 0.9), fill=(0, 0, 0, ORIGINAL_SUB_SHADOW_ALPHA)
        )
        shadow_patch = shadow_patch.filter(ImageFilter.GaussianBlur(ORIGINAL_SUB_SHADOW_BLUR))

        patch_x = int((self.TARGET_W - patch_w) / 2)
        patch_y = int(center_y - patch_h / 2)

        pil_final.alpha_composite(shadow_patch, (patch_x, patch_y))

        overlay = Image.new('RGBA', pil_final.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(overlay)
        curr_y = patch_y + pad
        for line, lw in zip(shaped_lines, line_widths):
            x = (self.TARGET_W - lw) / 2
            text_draw.text((x, curr_y), line, font=font, fill=(255, 255, 255, 255))
            curr_y += line_height

        return Image.alpha_composite(pil_final, overlay)

    def apply_theme_colors(self, bg, accent, banner):
        self.bg_color = bg
        self.accent_color = accent
        self.banner_color = banner
        self.generate_vertical_gradient()

    def compute_crop_dims(self, h, w, zoom=1.0):
        zoom = max(0.6, min(zoom, 2.0))
        ch = (min(h, w) * 0.92) / zoom
        cw = ch * self.CROP_RATIO
        if ch > h:  # can't crop taller than the source frame
            ch = float(h)
            cw = ch * self.CROP_RATIO
        if cw > w:  # can't crop wider than the source frame
            cw = float(w)
            ch = cw / self.CROP_RATIO
        return int(cw), int(ch)

    def compute_full_crop_dims(self, h, w, zoom=1.0):
        """
        Simple Crop mode: crops the source frame to the exact TARGET_W:TARGET_H
        (9:16) aspect ratio so it fills the whole vertical canvas edge-to-edge —
        no card, no letterboxing/blurred background needed.
        """
        zoom = max(0.6, min(zoom, 2.0))
        target_aspect = self.TARGET_W / self.TARGET_H
        ch = h / zoom
        cw = ch * target_aspect
        if cw > w:
            cw = float(w)
            ch = cw / target_aspect
        if ch > h:
            ch = float(h)
            cw = ch * target_aspect
        return int(cw), int(ch)

    def get_crop_dims(self, h, w, zoom=1.0):
        if self.output_mode == "simple":
            return self.compute_full_crop_dims(h, w, zoom)
        return self.compute_crop_dims(h, w, zoom)

    def get_ken_burns_frame(self, t):
        """
        Static-image mode only. Slowly zooms/pans across the oversized cached
        buffer (static_fg_image_full), starting at the exact framing the user
        picked and easing into a slightly tighter, slightly drifted crop by
        the end of the clip. Bounded so it never reveals the buffer's edges.
        """
        full = self.static_fg_image_full
        fh, fw = full.shape[:2]
        progress = min(1.0, t / max(1.0, self.duration))
        eased = progress * progress * (3 - 2 * progress)  # smoothstep

        zoom = 1.0 + KEN_BURNS_ZOOM_RANGE * eased
        win_w = min(fw, self.CARD_W / zoom)
        win_h = min(fh, self.CARD_H / zoom)

        max_dx = max(0.0, (fw - win_w) / 2.0)
        max_dy = max(0.0, (fh - win_h) / 2.0)
        dx = max_dx * KEN_BURNS_PAN_FRACTION * eased
        dy = max_dy * KEN_BURNS_PAN_FRACTION * eased * 0.4  # subtler vertical drift

        cx = fw / 2.0 + dx
        cy = fh / 2.0 + dy

        x1 = int(max(0, min(fw - win_w, cx - win_w / 2)))
        y1 = int(max(0, min(fh - win_h, cy - win_h / 2)))
        crop = full[y1:y1 + int(win_h), x1:x1 + int(win_w)]
        return cv2.resize(crop, (self.CARD_W, self.CARD_H), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # Simple Crop mode: heading text setup + rendering
    # ------------------------------------------------------------------
    def build_heading_cache(self, main_size=None, sub_size=None):
        """
        Word-wraps and font-fits each '|'-separated heading section once per
        shot (frame-independent), so every frame just blits the cached
        lines instead of re-measuring text 30x/sec. First section uses the
        larger font, any further sections use the smaller sub-font, each
        shrinking independently if it doesn't fit within
        HEADING_MAX_LINES_PER_SECTION. main_size/sub_size let the
        face-avoidance placement logic request a smaller starting point
        than the defaults when the full-size heading won't fit cleanly.
        """
        main_size = main_size if main_size is not None else HEADING_MAIN_FONT_SIZE
        sub_size = sub_size if sub_size is not None else HEADING_SUB_FONT_SIZE
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        raw_sections = [s.strip() for s in HEADING_TEXT.split('|') if s.strip()]
        cache = []
        for idx, section_text in enumerate(raw_sections):
            start_size = main_size if idx == 0 else sub_size
            size = start_size
            font, lines = None, None
            while size >= HEADING_MIN_FONT_SIZE:
                font = self.get_font(size, bold=True, is_subtitle=True)
                lines = self.wrap_and_shape_text_cache(section_text, draw, font, max_width=self.TARGET_W - 160)
                if len(lines) <= HEADING_MAX_LINES_PER_SECTION:
                    break
                size -= HEADING_FONT_STEP
            cache.append({"font": font, "lines": lines})
        self.heading_cache = cache

    def _heading_line_specs(self):
        """Returns ([(text, font, line_height), ...], total_block_height) for the current heading_cache."""
        if not self.heading_cache:
            return [], 0
        line_specs = []
        for section in self.heading_cache:
            font = section["font"]
            line_height = int(font.size * 1.25)
            for line in section["lines"]:
                line_specs.append((line, font, line_height))
        total_height = sum(lh for _, _, lh in line_specs)
        return line_specs, total_height

    def _heading_size_ladder(self):
        """Decreasing (main_size, sub_size) pairs to try when the heading needs to shrink to clear the face."""
        main, sub = HEADING_MAIN_FONT_SIZE, HEADING_SUB_FONT_SIZE
        step = HEADING_FONT_STEP * 2
        while main >= HEADING_MIN_FONT_SIZE:
            yield main, max(HEADING_MIN_FONT_SIZE, sub)
            main -= step
            sub = max(HEADING_MIN_FONT_SIZE, sub - step)

    def compute_heading_placement(self, frame, x1, y1, cw, ch):
        """
        Runs once per shot (on init and again after every scene cut/reframe,
        not every frame) — detects the face inside the *current* crop and
        chooses a heading zone (below the face by preference, matching the
        reference style, or above it) with enough clear room, shrinking the
        heading font through _heading_size_ladder() if the full size won't
        fit. If nothing fits cleanly even at the smallest size, it pauses
        and asks where to put it instead of guessing wrong over the face.
        """
        scale_x = self.TARGET_W / cw
        scale_y = self.TARGET_H / ch

        face_box = None
        try:
            res = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
            if res.detections:
                largest = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
                bx = largest.bounding_box
                fx1, fy1 = bx.origin_x - x1, bx.origin_y - y1
                fx2, fy2 = fx1 + bx.width, fy1 + bx.height
                face_box = (fx1 * scale_x, fy1 * scale_y, fx2 * scale_x, fy2 * scale_y)
        except Exception:
            face_box = None

        watermark_reserve = 150 if USER_WATERMARK else 60
        top_safe, bottom_safe = 60, self.TARGET_H - watermark_reserve
        margin = 45

        no_face = (face_box is None or face_box[3] < 0 or face_box[1] > self.TARGET_H)
        if no_face:
            self.build_heading_cache()
            self.heading_y_fraction = HEADING_Y_FRACTION
            return

        face_top = max(0.0, face_box[1])
        face_bottom = min(float(self.TARGET_H), face_box[3])

        for main_size, sub_size in self._heading_size_ladder():
            self.build_heading_cache(main_size, sub_size)
            _, req_h = self._heading_line_specs()

            below_top = face_bottom + margin
            below_room = bottom_safe - below_top
            above_bottom = face_top - margin
            above_room = above_bottom - top_safe

            if below_room >= req_h:
                self.heading_y_fraction = (below_top + req_h / 2) / self.TARGET_H
                return
            if above_room >= req_h:
                self.heading_y_fraction = (top_safe + req_h / 2) / self.TARGET_H
                return

        # Nothing fit cleanly even at the smallest size — ask rather than guess.
        print("\n⚠️ The heading text doesn't cleanly fit above or below the detected face at this framing.")
        print(f"   Face occupies roughly y={int(face_top)}-{int(face_bottom)} out of {self.TARGET_H}px tall.")
        choice = input("   Place it [b]elow anyway, [a]bove anyway, or type a vertical position 0.0-1.0 (Enter = below anyway): ").strip().lower()
        _, req_h = self._heading_line_specs()
        if choice == 'a':
            self.heading_y_fraction = (top_safe + req_h / 2) / self.TARGET_H
        elif choice and choice not in ('b',):
            try:
                frac = max(0.05, min(0.95, float(choice)))
                self.heading_y_fraction = frac
            except ValueError:
                self.heading_y_fraction = HEADING_Y_FRACTION
        else:
            below_top = face_bottom + margin
            self.heading_y_fraction = (below_top + req_h / 2) / self.TARGET_H

    def draw_persistent_heading(self, pil_final):
        """
        Draws the heading text (built by build_heading_cache) plus a dark
        gradient scrim behind it, centered around self.heading_y_fraction —
        set per-shot by compute_heading_placement to clear the face — same
        on every frame of that shot, no fade/timing.
        """
        if not self.heading_cache:
            return pil_final

        overlay = Image.new('RGBA', pil_final.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        line_specs, total_height = self._heading_line_specs()
        center_y = int(self.TARGET_H * self.heading_y_fraction)
        start_y = center_y - total_height // 2

        # Dark gradient scrim behind the text block for readability, same
        # look as the reference screenshot (text sits on a soft dark fade).
        scrim_top = max(0, start_y - HEADING_SCRIM_PAD)
        scrim_bottom = min(self.TARGET_H, start_y + total_height + HEADING_SCRIM_PAD)
        scrim_h = scrim_bottom - scrim_top
        if scrim_h > 0:
            for y in range(scrim_h):
                progress = y / max(1, scrim_h)
                fade = int(150 * math.sin(progress * math.pi))  # soft in, soft out
                draw.line([(0, scrim_top + y), (self.TARGET_W, scrim_top + y)], fill=(0, 0, 0, fade))

        curr_y = start_y
        for text, font, line_height in line_specs:
            w = draw.textlength(text, font=font)
            x = (self.TARGET_W - w) / 2
            draw.text((x + 3, curr_y + 3), text, font=font, fill=(0, 0, 0, 200))
            draw.text((x, curr_y), text, font=font, fill=(255, 255, 255, 255),
                       stroke_width=2, stroke_fill=(0, 0, 0, 230))
            curr_y += line_height

        return Image.alpha_composite(pil_final, overlay)

    def render_simple_frame(self, frame, t, is_sample, is_scene_cut, h, w):
        """
        Simple Crop mode: full-bleed manual-camera crop with an optional
        persistent heading and (optionally) subtitles/watermark — no card,
        no blurred background, no color themes.
        """
        if not self.is_initialized:
            if USER_WATERMARK:
                b_text = self.clean_and_shape_harakat(USER_WATERMARK)
                font_brand = self.get_font(42, bold=True)
                temp_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
                w_b = temp_draw.textlength(b_text, font=font_brand)

                stamp_w = int(w_b) + 80
                stamp_h = 75
                self.branding_img = Image.new('RGBA', (self.TARGET_W, stamp_h + 30), (0, 0, 0, 0))
                draw_b = ImageDraw.Draw(self.branding_img)

                stamp_x = (self.TARGET_W - stamp_w) // 2
                stamp_y = 15
                draw_b.rounded_rectangle((stamp_x, stamp_y, stamp_x + stamp_w, stamp_y + stamp_h), radius=16, outline=(255, 255, 255, 160), width=3)
                draw_b.text((stamp_x + 40, stamp_y + 12), b_text, font=font_brand, fill=(255, 255, 255, 240))

            self.build_simple_gradient_overlay()
            self.is_initialized = True

        # --- Manual camera positioning (re-prompts on scene cuts) ---
        if self.manual_cam_pos is None or (is_scene_cut and not is_sample):
            if is_scene_cut and not is_sample:
                print("\n🎥 Scene cut detected — reposition/rezoom the camera (check for a new window).")
            try:
                cam_x, cam_y, zoom = pick_manual_crop_position(
                    self, frame, initial_pos=self.manual_cam_pos, initial_zoom=self.manual_zoom
                )
                self.manual_cam_pos = (cam_x, cam_y)
                self.manual_zoom = zoom
            except Exception as e:
                print(f"⚠️ Manual positioning window failed ({e}) — keeping previous position/zoom.")
                if self.manual_cam_pos is None:
                    self.manual_cam_pos = (w / 2, h / 2)
            self.heading_needs_placement = True  # framing changed — re-check the heading against the new face position
        self.cam_x, self.cam_y = self.manual_cam_pos

        cw, ch = self.get_crop_dims(h, w, self.manual_zoom)
        x1 = max(0, min(w - cw, int(self.cam_x - cw / 2)))
        y1 = max(0, min(h - ch, int(self.cam_y - ch * 0.45)))
        full_frame = cv2.resize(frame[y1:y1 + ch, x1:x1 + cw], (self.TARGET_W, self.TARGET_H), interpolation=cv2.INTER_LINEAR)

        if self.enable_heading and HEADING_TEXT and self.heading_needs_placement and not is_sample:
            self.compute_heading_placement(frame, x1, y1, cw, ch)
            self.heading_needs_placement = False

        # Subtle grain to match the studio pipeline's finish.
        noise = self.grain_pool[self.main_frame_count % len(self.grain_pool)]
        full_frame = np.clip(full_frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        pil_final = Image.fromarray(full_frame).convert("RGBA")

        # Really light black vertical gradient (darkens gently toward the
        # lower part of the frame) — composited before the heading/
        # watermark/subtitles so they render crisp on top of it.
        if self.simple_gradient_overlay is not None:
            pil_final = Image.alpha_composite(pil_final, self.simple_gradient_overlay)

        # Optional custom overlay PNG (e.g. a pre-made gradient/vignette
        # graphic) — composited on top of the generated gradient (if any),
        # still underneath the heading/watermark/subtitles so those stay
        # crisp on top of it.
        if self.overlay_png_img is not None:
            pil_final = Image.alpha_composite(pil_final, self.overlay_png_img)

        if USER_WATERMARK and self.branding_img:
            pil_final.paste(self.branding_img, (0, self.TARGET_H - 120), self.branding_img)

        if self.enable_heading and self.heading_cache:
            pil_final = self.draw_persistent_heading(pil_final)

        if ENABLE_SUBTITLES and not is_sample:
            active_blocks = [c for c in self.current_subs if c['start'] <= t <= c['end']]
            if active_blocks:
                block = active_blocks[0]

                if self.active_sub_id != block['start']:
                    self.active_sub_id = block['start']
                    temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
                    temp_draw = ImageDraw.Draw(temp_img)
                    self.cached_sub_font, self.cached_sub_lines = self.fit_subtitle_font_and_lines(block['text'], temp_draw)

                if self.active_original_sub_id != block['start']:
                    self.active_original_sub_id = block['start']
                    if self.enable_dual_subs and block.get('original'):
                        temp_draw2 = ImageDraw.Draw(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
                        self.original_sub_font, self.cached_original_sub_lines = self.fit_original_sub_lines(block['original'], temp_draw2)
                    else:
                        self.cached_original_sub_lines = []

                time_on_screen = max(0.0, t - block['start'])
                raw_progress = min(1.0, time_on_screen / SUBTITLE_FADE_DURATION)
                eased_progress = 1.0 - (1.0 - raw_progress) ** 3
                fade_opacity = eased_progress
                rise_offset = int(SUBTITLE_RISE_PX * (1.0 - eased_progress))

                overlay = Image.new('RGBA', pil_final.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)

                # Subtitles sit in the bottom band, above the watermark —
                # kept clear of the heading, which lives higher up.
                watermark_top_y = self.TARGET_H - (150 if USER_WATERMARK else 50)
                sub_bottom = watermark_top_y - 24
                sub_top = int(self.TARGET_H * 0.80)

                main_start_y, main_h = self.draw_cached_subtitles(draw, self.cached_sub_lines, self.cached_sub_font,
                                            (255, 255, 255, 255), opacity=fade_opacity, rise=rise_offset,
                                            top_bound=sub_top, bottom_bound=sub_bottom)
                pil_final = Image.alpha_composite(pil_final, overlay)

                # Dual Subtitles: small original-language line stacked just
                # above the main translated subtitle block, same fade timing.
                if self.enable_dual_subs and self.cached_original_sub_lines:
                    orig_line_height = int(self.original_sub_font.size * 1.25)
                    orig_block_h = orig_line_height * len(self.cached_original_sub_lines)
                    orig_center_y = (main_start_y + rise_offset) - ORIGINAL_SUB_GAP_ABOVE_MAIN - orig_block_h / 2
                    orig_center_y = max(orig_block_h / 2 + 10, orig_center_y)
                    pil_final = self.draw_original_subtitle_block(
                        pil_final, self.cached_original_sub_lines, self.original_sub_font, orig_center_y
                    )

        return cv2.cvtColor(np.array(pil_final), cv2.COLOR_RGBA2RGB)

    def render_single_frame(self, frame, t, is_sample=False):
        h, w, _ = frame.shape
        
        # --- SCENE CUT DETECTION ALGORITHM ---
        is_scene_cut = False
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.prev_frame_gray is not None and not is_sample:
            diff = cv2.mean(cv2.absdiff(gray_frame, self.prev_frame_gray))[0]
            if diff > 25.0:  
                is_scene_cut = True
                
        if not is_sample:
            self.prev_frame_gray = gray_frame

        # --- SIMPLE CROP MODE: separate, lighter-weight render path ---
        if self.output_mode == "simple":
            return self.render_simple_frame(frame, t, is_sample, is_scene_cut, h, w)
        
        if self.enable_color_shift and len(self.themes) > 1 and not is_sample:
            progress = min(0.999, t / max(1.0, self.duration))
            val = progress * (len(self.themes) - 1)
            idx = int(math.floor(val))
            factor = val - idx
            
            active_bg = blend_colors(self.themes[idx]["bg"], self.themes[idx+1]["bg"], factor)
            active_accent = blend_colors(self.themes[idx]["accent"], self.themes[idx+1]["accent"], factor)
            active_banner = blend_colors(self.themes[idx]["banner"], self.themes[idx+1]["banner"], factor)
            self.apply_theme_colors(active_bg, active_accent, active_banner)
        
        if not self.is_initialized:
            mask = np.zeros((self.CARD_H, self.CARD_W), dtype=np.uint8)
            cv2.rectangle(mask, (self.CORNER_RADIUS, 0), (self.CARD_W-self.CORNER_RADIUS, self.CARD_H), 255, -1)
            cv2.rectangle(mask, (0, self.CORNER_RADIUS), (self.CARD_W, self.CARD_H-self.CORNER_RADIUS), 255, -1)
            for c in [(self.CORNER_RADIUS, self.CORNER_RADIUS), (self.CARD_W-self.CORNER_RADIUS, self.CORNER_RADIUS), 
                      (self.CORNER_RADIUS, self.CARD_H-self.CORNER_RADIUS), (self.CARD_W-self.CORNER_RADIUS, self.CARD_H-self.CORNER_RADIUS)]: 
                cv2.circle(mask, c, self.CORNER_RADIUS, 255, -1)
                
            self.fg_mask = np.expand_dims(mask.astype(np.float32)/255.0, axis=2)
            self.fg_inv_mask = 1.0 - self.fg_mask
            
            shadow = np.zeros((self.TARGET_H, self.TARGET_W), dtype=np.uint8)
            shadow[self.y_offset+25:self.y_offset+25+self.CARD_H, self.x_offset:self.x_offset+self.CARD_W] = mask
            self.shadow_multiplier = np.expand_dims(1.0 - (cv2.GaussianBlur(shadow, (99,99), 0).astype(np.float32)/255.0*0.75), axis=2)
            
            if SPEAKER_NAME:
                self.speaker_lines = [self.clean_and_shape_harakat(line.strip()) for line in SPEAKER_NAME.split('|')]
                self.font_main = self.get_font(46, bold=True)
                self.font_sub = self.get_font(32, bold=False)
                
                temp_img = Image.new('RGBA', (1,1), (0,0,0,0))
                draw_t = ImageDraw.Draw(temp_img)
                w1 = draw_t.textlength(self.speaker_lines[0], font=self.font_main)
                w2 = draw_t.textlength(self.speaker_lines[1], font=self.font_sub) if len(self.speaker_lines) > 1 else 0
                
                self.b_h = 135 if len(self.speaker_lines) > 1 else 95
                self.b_w = int(max(w1, w2)) + 120 
                self.banner_radius = self.b_h // 2
                
                mask_img = Image.new('L', (self.b_w, self.b_h), 0)
                draw_m = ImageDraw.Draw(mask_img)
                draw_m.rounded_rectangle((0, 0, self.b_w, self.b_h), radius=self.banner_radius, fill=255)
                self.speaker_banner_mask = np.expand_dims(np.array(mask_img, dtype=np.float32) / 255.0, axis=2)
                
            if USER_WATERMARK:
                b_text = self.clean_and_shape_harakat(USER_WATERMARK)
                font_brand = self.get_font(42, bold=True)
                temp_draw = ImageDraw.Draw(Image.new('RGBA', (1,1), (0,0,0,0)))
                w_b = temp_draw.textlength(b_text, font=font_brand)
                
                stamp_w = int(w_b) + 80
                stamp_h = 75
                self.branding_img = Image.new('RGBA', (self.TARGET_W, stamp_h + 30), (0,0,0,0))
                draw_b = ImageDraw.Draw(self.branding_img)
                
                stamp_x = (self.TARGET_W - stamp_w) // 2
                stamp_y = 15
                draw_b.rounded_rectangle((stamp_x, stamp_y, stamp_x + stamp_w, stamp_y + stamp_h), radius=16, outline=(255, 255, 255, 120), width=3)
                draw_b.text((stamp_x + 40, stamp_y + 12), b_text, font=font_brand, fill=(255, 255, 255, 240))

            self.is_initialized = True
            
        # Static Image mode skips face detection and all camera-tracking
        # logic entirely — the card uses the Ken Burns pan/zoom below,
        # while the background/audio/subtitles keep coming from `frame`
        # or the cached static backdrop (see below).
        use_static_image = (self.foreground_mode == "image" and self.static_fg_image_full is not None)

        if not use_static_image:
            res = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
            if res.detections:
                largest_face = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
                fx = largest_face.bounding_box.origin_x + largest_face.bounding_box.width/2
                fy = largest_face.bounding_box.origin_y + largest_face.bounding_box.height * 0.35 
            else:
                fx, fy = w/2, h/2
                
            # --- ADVANCED PODCAST FIXED CROP CORE ---
            if not is_sample:
                if MANUAL_TRACKING:
                    # Prompt when we have no position yet, or a real cut just happened.
                    # (self.manual_cam_pos is already set from the pre-render pick on frame 0,
                    # so this only fires again on genuine mid-render cuts.)
                    if self.manual_cam_pos is None or is_scene_cut:
                        if is_scene_cut:
                            print("\n🎥 Scene cut detected — reposition/rezoom the camera (check for a new window).")
                        try:
                            cam_x, cam_y, zoom = pick_manual_crop_position(
                                self, frame, initial_pos=self.manual_cam_pos, initial_zoom=self.manual_zoom
                            )
                            self.manual_cam_pos = (cam_x, cam_y)
                            self.manual_zoom = zoom
                        except Exception as e:
                            print(f"⚠️ Manual positioning window failed ({e}) — keeping previous position/zoom.")
                            if self.manual_cam_pos is None:
                                self.manual_cam_pos = (w / 2, h / 2)
                    self.cam_x, self.cam_y = self.manual_cam_pos
                # Condition 1: If it's a camera cut OR if the engine hasn't initialized the camera coordinates yet
                elif is_scene_cut or self.cam_x is None:
                    # Force instant coordinate teleportation to perfectly center the new speaker
                    self.cam_x = fx
                    self.cam_y = fy
                # Condition 2: If we are in the middle of a continuous scene and active tracking is on, smoothly follow
                elif ACTIVE_TRACKING:
                    self.cam_x += (fx - self.cam_x) * 0.25
                    self.cam_y += (fy - self.cam_y) * 0.25
                # Condition 3: If ACTIVE_TRACKING is False and there is no scene cut, it skips execution entirely.
                # This locks the camera completely static, keeping the image 100% still until the next cut occurs.
            else:
                self.cam_x, self.cam_y = fx, fy

        # --- BACKDROP: source video frame (video mode) OR the static photo
        # itself, pre-blurred (image mode) — using the photo avoids a black
        # void when the source video is letterboxed or otherwise unsuitable
        # as a backdrop, and gives image mode its own rich, on-theme texture.
        if use_static_image and self.static_bg_small is not None:
            small_blur = self.static_bg_small
        else:
            bg = cv2.resize(frame, (int(w*(self.TARGET_H/h)), self.TARGET_H))
            bg_cropped = bg[:, (bg.shape[1] - self.TARGET_W) // 2 : (bg.shape[1] - self.TARGET_W) // 2 + self.TARGET_W]
            small_blur = cv2.GaussianBlur(cv2.resize(bg_cropped, (self.TARGET_W//4, self.TARGET_H//4)), (45,45), 0)
        
        leak_small = np.zeros((self.TARGET_H//4, self.TARGET_W//4, 3), dtype=np.uint8)
        leak_radius = 165 if use_static_image else 120
        leak_alpha = 0.65 if use_static_image else 0.45
        lx = int((self.TARGET_W//4) * 0.5 + math.sin(t * 0.6) * (self.TARGET_W//4 * 0.4))
        ly = int((self.TARGET_H//4) * 0.5 + math.cos(t * 0.4) * (self.TARGET_H//4 * 0.3))
        cv2.circle(leak_small, (lx, ly), leak_radius, (self.accent_color[2], self.accent_color[1], self.accent_color[0]), -1)
        if use_static_image:
            # Second, slower counter-drifting leak so a still-image backdrop
            # still has layered motion instead of a single moving dot.
            lx2 = int((self.TARGET_W//4) * 0.5 + math.cos(t * 0.35 + 2.0) * (self.TARGET_W//4 * 0.35))
            ly2 = int((self.TARGET_H//4) * 0.5 + math.sin(t * 0.25 + 1.0) * (self.TARGET_H//4 * 0.35))
            cv2.circle(leak_small, (lx2, ly2), int(leak_radius * 0.7), self.bg_color, -1)
        small_bg_with_leak = cv2.addWeighted(small_blur, 1.0, cv2.GaussianBlur(leak_small, (99,99), 0), leak_alpha, 0)
        
        final = cv2.resize(small_bg_with_leak, (self.TARGET_W, self.TARGET_H))
        final = (final.astype(np.float32) * self.shadow_multiplier).astype(np.uint8)
        
        if use_static_image:
            fg_uint = self.get_ken_burns_frame(t)
        else:
            cw, ch = self.compute_crop_dims(h, w, self.manual_zoom)
            x1 = max(0, min(w - cw, int(self.cam_x - cw / 2)))
            y1 = max(int(h*0.05), min(h - int(h*0.05) - ch, int(self.cam_y - ch * 0.45)))
            fg_uint = cv2.resize(frame[y1:y1+ch, x1:x1+cw], (self.CARD_W, self.CARD_H))
        x_start, y_start = self.x_offset, self.y_offset
        final[y_start:y_start+self.CARD_H, x_start:x_start+self.CARD_W] = (
            fg_uint.astype(np.float32) * self.fg_mask + final[y_start:y_start+self.CARD_H, x_start:x_start+self.CARD_W].astype(np.float32) * self.fg_inv_mask
        ).astype(np.uint8)

        luxury_canvas = np.full((self.TARGET_H, self.TARGET_W, 3), self.bg_color, dtype=np.float32)
        final = (final.astype(np.float32) * (1.0 - self.vertical_gradient_mask)) + (luxury_canvas * self.vertical_gradient_mask)
        
        vid_progress = min(1.0, t / max(1.0, self.duration))

        if SPEAKER_NAME:
            banner_y = self.y_offset + self.CARD_H + STUDIO_BANNER_GAP 
            intro_start, intro_dur = 0.5, 1.2
            start_x = -self.b_w - 50
            target_x = 0
            
            if is_sample: current_x = target_x
            else:
                if t < intro_start: current_x = start_x
                elif t < intro_start + intro_dur:
                    current_x = start_x + (target_x - start_x) * (1.0 - (1.0 - ((t - intro_start) / intro_dur))**5)
                else: current_x = target_x
            
            b_x = int(current_x)
            rx1, rx2 = max(0, b_x), min(self.TARGET_W, b_x + self.b_w)
            ry1, ry2 = max(0, int(banner_y)), min(self.TARGET_H, int(banner_y) + self.b_h)
            
            if rx2 > rx1 and ry2 > ry1:
                roi = final[ry1:ry2, rx1:rx2]
                glass_blur = cv2.GaussianBlur(roi, (75, 75), 0)
                tint = np.full(glass_blur.shape, self.banner_color[:3][::-1], dtype=np.float32)
                glass_tinted = cv2.addWeighted(glass_blur, 0.4, tint, 0.6, 0)
                mask_roi = self.speaker_banner_mask[ry1-int(banner_y):ry2-int(banner_y), rx1-b_x:rx2-b_x]
                final[ry1:ry2, rx1:rx2] = (glass_tinted * mask_roi + roi * (1.0 - mask_roi))

        noise = self.grain_pool[self.main_frame_count % len(self.grain_pool)]
        final = np.clip(final + noise, 0, 255).astype(np.uint8)
        pil_final = Image.fromarray(final).convert("RGBA")
        
        if USER_WATERMARK and self.branding_img:
            tinted_brand = Image.new('RGBA', self.branding_img.size, (self.accent_color[0], self.accent_color[1], self.accent_color[2], 255))
            pil_final.paste(tinted_brand, (0, self.TARGET_H - 120), self.branding_img)

        if SPEAKER_NAME and current_x > start_x + 5:
            spk_img = Image.new('RGBA', (self.b_w, self.b_h), (0,0,0,0))
            draw_s = ImageDraw.Draw(spk_img)
            draw_s.rounded_rectangle((0, 0, self.b_w, self.b_h), radius=self.banner_radius, outline=(255, 255, 255, 55), width=2)
            draw_s.text((45, 22), self.speaker_lines[0], font=self.font_main, fill=self.accent_color)
            if len(self.speaker_lines) > 1:
                draw_s.text((45, 80), self.speaker_lines[1], font=self.font_sub, fill=(self.accent_color[0], self.accent_color[1], self.accent_color[2], 195))
            
            pil_final.paste(spk_img, (int(current_x), int(banner_y)), spk_img)
            
            vis_overlay = Image.new('RGBA', pil_final.size, (0,0,0,0))
            draw_vis = ImageDraw.Draw(vis_overlay)

            prog_start_x = b_x + self.banner_radius
            prog_max_w = self.b_w - (self.banner_radius * 2)
            prog_curr_w = int(prog_max_w * vid_progress)
            
            if prog_curr_w > 0:
                draw_vis.line(
                    [(prog_start_x, int(banner_y) + self.b_h - 2), (prog_start_x + prog_curr_w, int(banner_y) + self.b_h - 2)], 
                    fill=self.accent_color, width=1
                )
                head_x = prog_start_x + prog_curr_w
                draw_vis.ellipse(
                    [(head_x - 2, int(banner_y) + self.b_h - 4), (head_x + 2, int(banner_y) + self.b_h)], 
                    fill=(255, 255, 255, 255)
                )

            is_speaking = False
            if is_sample: is_speaking = True
            else:
                if any(c['start'] <= t <= c['end'] for c in self.current_subs): is_speaking = True

            vis_x = int(current_x) + 45
            vis_y = int(banner_y) - 20 
            wave_width = 75
            wave_points = []

            for i in range(wave_width):
                nx = i / (wave_width - 1)
                envelope = math.sin(nx * math.pi)
                if is_speaking:
                    wave_val = math.sin(t * 5.5 + nx * 12.0) * 0.6 + math.cos(t * 7.2 - nx * 18.0) * 0.4
                    amplitude = 22.0
                else:
                    wave_val = math.sin(t * 2.0 - nx * 6.0)
                    amplitude = 4.0
                y_offset = wave_val * amplitude * envelope
                wave_points.append((vis_x + i, vis_y - y_offset))
                
            draw_vis.line(wave_points, fill=self.accent_color, width=3, joint="curve")
            pil_final = Image.alpha_composite(pil_final, vis_overlay)
        
        if ENABLE_SUBTITLES and not is_sample:
            active_blocks = [c for c in self.current_subs if c['start'] <= t <= c['end']]
            if active_blocks:
                block = active_blocks[0]
                
                if self.active_sub_id != block['start']:
                    self.active_sub_id = block['start']
                    temp_img = Image.new('RGBA', (1,1), (0,0,0,0))
                    temp_draw = ImageDraw.Draw(temp_img)
                    # Fits the block into MAX_SUBTITLE_LINES by shrinking the
                    # font first, rather than letting long quotes grow tall
                    # and crowd the banner/watermark.
                    self.cached_sub_font, self.cached_sub_lines = self.fit_subtitle_font_and_lines(block['text'], temp_draw)

                if self.active_original_sub_id != block['start']:
                    self.active_original_sub_id = block['start']
                    if self.enable_dual_subs and block.get('original'):
                        temp_draw2 = ImageDraw.Draw(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
                        self.original_sub_font, self.cached_original_sub_lines = self.fit_original_sub_lines(block['original'], temp_draw2)
                    else:
                        self.cached_original_sub_lines = []

                time_on_screen = max(0.0, t - block['start'])
                raw_progress = min(1.0, time_on_screen / SUBTITLE_FADE_DURATION)
                eased_progress = 1.0 - (1.0 - raw_progress) ** 3  # ease-out cubic
                fade_opacity = eased_progress
                rise_offset = int(SUBTITLE_RISE_PX * (1.0 - eased_progress))

                overlay = Image.new('RGBA', pil_final.size, (0,0,0,0))
                draw = ImageDraw.Draw(overlay)
                
                self.draw_cached_subtitles(draw, self.cached_sub_lines, self.cached_sub_font, self.accent_color,
                                            opacity=fade_opacity, rise=rise_offset)
                pil_final = Image.alpha_composite(pil_final, overlay)

                # Dual Subtitles (Studio mode): the small original-language
                # line lives in the tight gap between the bottom of the card
                # and the top of the speaker banner (or, if there's no
                # banner, a matching small gap below the card). Centered
                # strictly within [card bottom, banner top] so it can never
                # land on top of the banner itself.
                if self.enable_dual_subs and self.cached_original_sub_lines:
                    gap_top = self.y_offset + self.CARD_H
                    gap_bottom = self.y_offset + self.CARD_H + STUDIO_BANNER_GAP
                    orig_center_y = (gap_top + gap_bottom) / 2
                    pil_final = self.draw_original_subtitle_block(
                        pil_final, self.cached_original_sub_lines, self.original_sub_font, orig_center_y
                    )

        return cv2.cvtColor(np.array(pil_final), cv2.COLOR_RGBA2RGB)

    def process_main_frame(self, frame):
        t = self.main_frame_count / self.fps
        self.main_frame_count += 1
        return self.render_single_frame(frame, t, is_sample=False)

    def reload_whisper_model(self, size):
        """
        Swaps the loaded Whisper model for a different size on the fly —
        used by the transcription review/retry loop when the user wants to
        try a bigger (or smaller/faster) model without restarting the whole
        script. Tries CUDA first, then falls back to CPU, same ladder logic
        as the startup load. Keeps the previous model loaded if every
        attempt fails, so a bad retry choice can't leave whisper_model unset.
        """
        print(f"🔄 Reloading Whisper as '{size}'...")
        for sz, device, compute in [(size, "cuda", "int8_float16"), (size, "cpu", "int8")]:
            try:
                self.whisper_model = WhisperModel(sz, device=device, compute_type=compute)
                self.whisper_model_size = sz
                print(f"✅ Loaded Whisper '{sz}' on {device}.")
                return True
            except Exception as e:
                print(f"   ⚠️ '{sz}' on {device} failed: {e}")
        print("❌ Could not reload with that model size — keeping the previously loaded model.")
        return False

    def extract_srt_only(self, audio_path, srt_path, language=None, initial_prompt=None,
                          condition_on_previous_text=False, temperature=None):
        """
        Runs faster-whisper and writes the raw transcript to srt_path.

        language/initial_prompt/condition_on_previous_text/temperature are
        all overridable (by the transcription review/retry loop in
        __main__) without touching the WHISPER_LANG global, so a retry can
        try a forced language, a vocabulary hint, or different determinism
        settings without restarting the script:

        - condition_on_previous_text defaults to False here (the library
          default is True). Carrying the previous segment's text forward as
          context is the single most common cause of Whisper going
          "wrong" partway through a clip — one garbled or hallucinated
          segment poisons the prompt for every segment after it, and the
          errors compound. Off by default; the retry loop can turn it back
          on if you want cross-segment context for a clean recording.
        - temperature=None uses faster-whisper's own fallback ladder
          (it retries a segment at a higher, more random temperature if the
          low-temperature pass looks low-quality) — this is where run-to-run
          variance mostly comes from. Passing temperature=0.0 forces fully
          deterministic greedy decoding instead.
        """
        lang = language if language is not None else WHISPER_LANG
        print(f"🎙️ Running Whisper Audio Extraction (Model: {self.whisper_model_size or '?'}, "
              f"Language: {lang.upper() if lang else 'Auto'})...")

        transcribe_kwargs = dict(
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=200),
            language=lang,
            beam_size=5,
            best_of=5,
            condition_on_previous_text=condition_on_previous_text,
        )
        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        if temperature is not None:
            transcribe_kwargs["temperature"] = temperature

        segments, _ = self.whisper_model.transcribe(audio_path, **transcribe_kwargs)
        
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments):
                f.write(f"{i+1}\n")
                f.write(f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}\n")
                f.write(f"{segment.text.strip()}\n\n")
        print(f"✅ Subtitle extraction complete! Saved clean chunks to {srt_path}")

def rgb_to_hex(rgb_tuple):
    return '{:02x}{:02x}{:02x}'.format(rgb_tuple[0], rgb_tuple[1], rgb_tuple[2]).upper()

def build_instagram_caption(title, speaker, yt_link, clip_start, clip_end, explanation):
    lines = []
    if title:
        lines.append(f"🎬 {title}")
        lines.append("")
    if explanation:
        lines.append(explanation)
        lines.append("")

    meta_lines = []
    if speaker:
        meta_lines.append(f"🎙️ Speaker: {speaker}")
    if yt_link:
        meta_lines.append(f"🔗 Full Video: {yt_link}")
    if clip_start or clip_end:
        meta_lines.append(f"⏱️ Clip Timestamp: {clip_start} – {clip_end}")

    if meta_lines:
        divider = "─" * 28
        lines.append(divider)
        lines.extend(meta_lines)
        lines.append(divider)

    return "\n".join(lines).strip() + "\n"


def open_in_editor(path):
    """Opens `path` in the OS's default text editor, cross-platform."""
    try:
        if platform.system() == 'Darwin':
            subprocess.call(('open', path))
        elif platform.system() == 'Windows':
            os.startfile(path)
        else:
            subprocess.call(('xdg-open', path))
    except Exception as e:
        print(f"⚠️ Could not auto-launch text editor ({e}). Please open {path} manually.")


def review_and_retry_transcription(director, audio_path, srt_path):
    """
    Runs the raw Whisper transcription, opens it so you can read through it,
    and — if it's wrong — lets you re-run it with different settings as many
    times as you want, BEFORE the translation-intercept step overwrites the
    file with your manual translation. This is the fix for "the transcript
    is wrong": instead of only discovering it's wrong once you're already
    hand-translating garbled text, you get a dedicated checkpoint to retry
    the transcription itself with:
      - a forced/corrected language code (wrong language picked = wrong text)
      - a vocabulary hint (names, jargon, hard words) fed in as a prompt
      - context carryover between segments turned on/off (off is usually the
        fix when errors get worse/weirder as the clip goes on)
      - fully deterministic decoding (removes run-to-run randomness)
      - a different/larger model size, if the current one keeps struggling
    Nothing here re-extracts audio or restarts the pipeline — only Whisper
    itself re-runs, so retries are cheap.
    """
    language = WHISPER_LANG
    initial_prompt = None
    condition_on_previous = False
    temperature = None

    while True:
        director.extract_srt_only(
            audio_path, srt_path,
            language=language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous,
            temperature=temperature,
        )
        print(f"\n📝 Raw transcript ready — opening {srt_path} so you can check it for accuracy.")
        print("   (This is just the review pass — your manual translation happens in the next step.)")
        open_in_editor(srt_path)

        choice = input(
            "\n🔎 Does the transcript look accurate?\n"
            "   [ENTER] = yes, continue to translation\n"
            "   [r]     = re-run transcription with different settings\n"
            "👉 Your choice: "
        ).strip().lower()

        if choice != 'r':
            return

        print("\n🔁 RE-TRANSCRIBE OPTIONS (Enter = leave a given setting unchanged):")

        new_lang = input(f"   🌐 Force a language code (current: {language or 'auto'}, e.g. ur/hi/ar/en): ").strip()
        if new_lang:
            language = new_lang

        hint = input("   💡 Add a vocabulary hint — names, jargon, hard-to-hear words (current: "
                      f"{initial_prompt or 'none'}): ").strip()
        if hint:
            initial_prompt = hint

        ctx_choice = input(
            "   🧠 Carry context between segments? Turning this OFF usually fixes errors that "
            f"compound later in the clip (current: {'ON' if condition_on_previous else 'OFF'}) [y/n]: "
        ).strip().lower()
        if ctx_choice in ('y', 'yes'):
            condition_on_previous = True
        elif ctx_choice in ('n', 'no'):
            condition_on_previous = False

        temp_choice = input(
            "   🎲 Force fully deterministic decoding, no randomness (current: "
            f"{'deterministic' if temperature == 0.0 else 'library default'}) [y/n]: "
        ).strip().lower()
        if temp_choice in ('y', 'yes'):
            temperature = 0.0
        elif temp_choice in ('n', 'no'):
            temperature = None

        model_choice = input(
            f"   🧠 Try a different model size (current: {director.whisper_model_size}, "
            "e.g. large-v3 / distil-large-v3 / medium): "
        ).strip()
        if model_choice:
            director.reload_whisper_model(model_choice)

        print("\n🔄 Re-running transcription with the updated settings...")

def pick_manual_crop_position(director, rgb_frame, initial_pos=None, initial_zoom=1.0):
    """
    Opens a window showing the current frame with a crop box you frame by
    dragging in any direction — click anywhere and drag LEFT/RIGHT or
    UP/DOWN to slide the box. The starting horizontal position defaults to
    the detected face (or the frame center if none is found) as a good
    starting guess for a new cut/new speaker; drag from there to fine-tune.
    Scroll the mouse wheel, or press +/-, to zoom in/out live. ENTER, SPACE,
    or ESC all confirm the current position and zoom — ESC is there so a
    false-positive cut can be dismissed with one tap when no change is
    needed. If initial_pos/initial_zoom are given, the vertical
    position/zoom start there instead of centered/default zoom, so
    re-prompts default to the last vertical choice (horizontal still starts
    from the freshly detected face each time, but can now be dragged from
    there). Returns (cam_x, cam_y, zoom) — cam_x/cam_y in FULL-RESOLUTION
    source-frame coordinates.

    The on-screen preview window is fit to BOTH width and height (capped at
    MAX_PREVIEW_DISPLAY_W x MAX_PREVIEW_DISPLAY_H), so it always fits on
    screen no matter the source's resolution or aspect ratio — a portrait
    or very high-res source previously produced a window taller than the
    screen with WINDOW_AUTOSIZE giving no way to resize or scroll it. The
    window is also created as WINDOW_NORMAL (resizable/draggable) rather
    than WINDOW_AUTOSIZE, as an extra safety net on unusually small
    screens.
    """
    h, w = rgb_frame.shape[:2]
    bgr_full = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    # Do all interaction math in a screen-sized preview, then convert the
    # final result back to full resolution — avoids relying on OpenCV's
    # window-resize-to-mouse-coordinate mapping, which isn't reliable.
    # Fit to BOTH width and height so portrait / oversized sources never
    # produce a preview window bigger than the screen.
    scale = min(MAX_PREVIEW_DISPLAY_W / w, MAX_PREVIEW_DISPLAY_H / h, 1.0)
    display_w = max(1, int(w * scale))
    display_h = max(1, int(h * scale))
    bgr_display = cv2.resize(bgr_full, (display_w, display_h), interpolation=cv2.INTER_AREA)

    MIN_ZOOM, MAX_ZOOM, ZOOM_STEP = 0.6, 2.0, 0.05

    # Starting horizontal position defaults to the detected face in THIS
    # frame (falls back to frame center if none found) — a good starting
    # guess since a new cut usually means a new speaker in a new spot.
    # From here it's fully draggable, same as the vertical axis.
    try:
        face_res = director.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame))
        if face_res.detections:
            largest_face = max(face_res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
            fx = largest_face.bounding_box.origin_x + largest_face.bounding_box.width / 2
        else:
            fx = w / 2
    except Exception:
        fx = w / 2
    start_cx = fx * scale

    # Vertical framing carries over from the previous pick if given.
    start_cy = (initial_pos[1] * scale) if initial_pos is not None else (display_h / 2)

    state = {
        "cx": start_cx,
        "cy": start_cy,
        "zoom": max(MIN_ZOOM, min(initial_zoom, MAX_ZOOM)),
        "dragging": False,
        "drag_start_x": 0,
        "drag_start_y": 0,
        "cx_start": 0,
        "cy_start": 0,
    }

    def current_box():
        cw, ch = director.get_crop_dims(h, w, state["zoom"])
        dcw, dch = max(1, int(cw * scale)), max(1, int(ch * scale))
        min_x1, max_x1 = 0, max(0, display_w - dcw)
        min_y1, max_y1 = 0, max(0, display_h - dch)
        x1 = int(max(min_x1, min(max_x1, state["cx"] - dcw / 2)))
        y1 = int(max(min_y1, min(max_y1, state["cy"] - dch / 2)))
        return x1, y1, dcw, dch

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["drag_start_x"] = x
            state["drag_start_y"] = y
            state["cx_start"] = state["cx"]
            state["cy_start"] = state["cy"]
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            dx = x - state["drag_start_x"]
            dy = y - state["drag_start_y"]
            state["cx"] = state["cx_start"] + dx
            state["cy"] = state["cy_start"] + dy
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            step = ZOOM_STEP if flags > 0 else -ZOOM_STEP
            state["zoom"] = max(MIN_ZOOM, min(state["zoom"] + step, MAX_ZOOM))

    window = "Camera Framing - drag to move (any direction), scroll/+/-=zoom, ENTER/SPACE/ESC=confirm"
    # WINDOW_NORMAL (not WINDOW_AUTOSIZE) so the window can still be
    # resized/dragged by the OS window manager if it doesn't fully fit —
    # a safety net on top of the fit-to-screen sizing above.
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, display_w, display_h)
    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            x1, y1, dcw, dch = current_box()
            canvas = (bgr_display.astype(np.float32) * 0.35).astype(np.uint8)
            canvas[y1:y1+dch, x1:x1+dcw] = bgr_display[y1:y1+dch, x1:x1+dcw]
            cv2.rectangle(canvas, (x1, y1), (x1+dcw, y1+dch), (0, 215, 255), 3)
            cv2.putText(canvas, "Drag to move frame (any direction)  Scroll or +/- =zoom  ENTER/SPACE/ESC=confirm", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Zoom: {int(state['zoom']*100)}%", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)

            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32, 27):
                break
            elif key in (43, 61):  # '+' or '='
                state["zoom"] = max(MIN_ZOOM, min(state["zoom"] + ZOOM_STEP, MAX_ZOOM))
            elif key in (45, 95):  # '-' or '_'
                state["zoom"] = max(MIN_ZOOM, min(state["zoom"] - ZOOM_STEP, MAX_ZOOM))
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)

    x1, y1, dcw, dch = current_box()
    full_x1 = x1 / scale
    full_y1 = y1 / scale
    cw, ch = director.get_crop_dims(h, w, state["zoom"])
    cam_x = full_x1 + cw / 2
    cam_y = full_y1 + ch * 0.45
    return (cam_x, cam_y, state["zoom"])


def pick_gradient_settings(preview_frame_rgb, initial_start_frac, initial_end_frac, initial_max_alpha):
    """
    Simple Crop mode only. Opens a window showing the exact 9:16 framed
    preview (the same crop the video will use) with three live trackbars —
    Start %, End %, Density — so the bottom dark gradient's height/position
    and strength can be dialed in visually instead of guessing constants.
    The preview overlay is recomputed every frame from the current slider
    values using the identical smoothstep math the real render uses, so
    what's shown here is exactly what will render. Drag Density all the way
    to 0 to disable the generated gradient entirely (render_simple_frame /
    build_simple_gradient_overlay treat max_alpha<=0 as "no gradient layer
    at all"). ENTER, SPACE, or ESC all confirm the current slider values
    (also happens if the window is closed). Returns (start_frac, end_frac,
    max_alpha).

    The preview is fit to BOTH width and height (capped at
    MAX_PREVIEW_DISPLAY_W x MAX_PREVIEW_DISPLAY_H) and the window is
    resizable, matching pick_manual_crop_position — the incoming
    preview_frame_rgb is already the 9:16 TARGET_W x TARGET_H canvas, so on
    most screens this fits directly, but capping both dimensions keeps it
    safe on smaller displays too.
    """
    ph, pw = preview_frame_rgb.shape[:2]
    scale = min(MAX_PREVIEW_DISPLAY_W / pw, MAX_PREVIEW_DISPLAY_H / ph, 1.0)
    display_w = max(1, int(pw * scale))
    display_h = max(1, int(ph * scale))
    bgr_base = cv2.cvtColor(
        cv2.resize(preview_frame_rgb, (display_w, display_h), interpolation=cv2.INTER_AREA),
        cv2.COLOR_RGB2BGR
    )

    window = "Dark Gradient Setup - drag sliders, ENTER/SPACE/ESC=confirm"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, display_w, display_h)
    cv2.createTrackbar("Start %", window, int(round(initial_start_frac * 100)), 100, lambda v: None)
    cv2.createTrackbar("End %", window, int(round(initial_end_frac * 100)), 100, lambda v: None)
    cv2.createTrackbar("Density", window, int(round(initial_max_alpha)), 255, lambda v: None)

    s_frac, e_frac, density = initial_start_frac, initial_end_frac, initial_max_alpha

    try:
        while True:
            s_pct = cv2.getTrackbarPos("Start %", window)
            e_pct = cv2.getTrackbarPos("End %", window)
            density = cv2.getTrackbarPos("Density", window)

            # End must sit after Start or the ramp math breaks down.
            if e_pct <= s_pct:
                e_pct = min(100, s_pct + 1)
                cv2.setTrackbarPos("End %", window, e_pct)

            s_frac, e_frac = s_pct / 100.0, e_pct / 100.0

            mask = np.zeros((display_h, display_w), dtype=np.float32)
            start_y = int(display_h * s_frac)
            end_y = int(display_h * e_frac)
            for y in range(display_h):
                if y < start_y:
                    mask[y, :] = 0.0
                elif y > end_y:
                    mask[y, :] = 1.0
                else:
                    progress = (y - start_y) / max(1, (end_y - start_y))
                    mask[y, :] = progress * progress * (3 - 2 * progress)  # smoothstep

            alpha_layer = (mask * (density / 255.0))[:, :, None]
            canvas = (bgr_base.astype(np.float32) * (1.0 - alpha_layer)).astype(np.uint8)

            cv2.putText(canvas, f"Start {s_pct}%  End {e_pct}%  Density {density}/255", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "ENTER / SPACE / ESC = confirm  (Density 0 = gradient off)", (15, display_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32, 27):
                break
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)

    return s_frac, e_frac, density


def build_shadow_mask_from_image(pil_img, target_w, target_h):
    """
    Converts an arbitrary white-to-black (or light-to-dark) gradient/vignette
    graphic — like a flattened Pngtree-style PNG with no real transparency —
    into a pure black shadow overlay: white/light areas become fully
    transparent, black/dark areas become an opaque black shadow, and every
    shade in between maps to a proportional alpha. This is done from
    luminance, ignoring any existing alpha channel in the source file, so it
    works whether the PNG is "transparent" in name only (flattened onto
    white) or has partial actual transparency — either way only the DARK
    part of the graphic ends up affecting the video; the light/white part
    disappears instead of washing the frame out.
    """
    gray = pil_img.convert("L").resize((target_w, target_h), Image.LANCZOS)
    luminance = np.array(gray, dtype=np.uint8)
    alpha = (255 - luminance).astype(np.uint8)  # black (low luminance) -> high alpha
    rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)  # color stays pure black everywhere
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, mode='RGBA')


def pick_overlay_png(director):
    """
    Simple Crop mode only. Asks whether to composite a custom overlay PNG
    (e.g. a pre-made gradient/vignette graphic like the Pngtree-style
    black-to-white fade) on top of every frame, on top of the generated
    dark gradient (if any) and underneath the heading/watermark/subtitles.
    Opens the same file browser used for Static Image foreground selection.

    The chosen image is converted via build_shadow_mask_from_image() into a
    pure black shadow mask — only the DARK part of the graphic becomes a
    shadow; the light/white part is dropped entirely rather than pasted in
    as opaque white — then stretched to fill the full TARGET_W x TARGET_H
    canvas once here, so render_simple_frame can just alpha-composite it
    every frame with no per-frame cost. Sets director.overlay_png_img (or
    leaves it None if skipped/failed).
    """
    print("\n🖼️ OVERLAY PNG: composite a custom shadow graphic (e.g. a gradient/vignette PNG) over every frame?")
    print("   (Only the dark part of the image is used as a shadow — any white/light part is ignored, not pasted in.)")
    choice = input("👉 Add overlay PNG? (y/n): ").strip().lower()
    if choice not in ('y', 'yes'):
        return

    print("📂 Opening the file browser — choose the overlay PNG...")
    overlay_path = pick_image_file_dialog()
    if not overlay_path or not os.path.exists(overlay_path):
        print("⚠️ No valid overlay PNG selected — skipping.")
        return

    try:
        source_pil = Image.open(overlay_path)
        director.overlay_png_img = build_shadow_mask_from_image(source_pil, director.TARGET_W, director.TARGET_H)
        print(f"✅ Overlay PNG loaded as a black shadow mask and will be composited on every frame: {overlay_path}")
    except Exception as e:
        print(f"⚠️ Could not load overlay PNG ({e}) — skipping.")


def isolate_vocals(audio_path, work_dir, model_name="htdemucs"):
    """
    Runs Demucs (Meta's music/vocal source-separation model) on audio_path
    and returns the path to the isolated vocals-only .wav, or None if
    separation failed for any reason (caller should fall back to the
    original mixed audio rather than let the whole render fail over this).

    Tries CUDA first, then falls back to CPU — same "try the fast path,
    fall back to the slow one" pattern as the Whisper model ladder above.
    Requires the `demucs` package: pip install demucs

    Honest limitation: Demucs is trained to separate *singing/melodic
    vocals* from *instrumental* accompaniment. It's genuinely good at
    stripping instrumental background music (oud/qanun/percussion beds,
    ambient pads, etc.) out from under a speaking voice. A background
    nasheed that is ITSELF mostly unaccompanied vocals is a much harder
    "voice vs. voice" separation problem — Demucs has no way to know
    which voice is the "real" one, so results there are hit-or-miss:
    usually quieter/duller, not perfectly gone. There's no clean,
    off-the-shelf tool for reliable speaker-vs-speaker separation the
    way there is for voice-vs-instrument.
    """
    out_dir = os.path.join(work_dir, "demucs_separated")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    for device in ("cuda", "cpu"):
        print(f"🎚️ Running Demucs vocal isolation ({model_name}) on {device}...")
        if model_name == "htdemucs_ft":
            print("   (htdemucs_ft runs 4 model passes — expect several minutes even on a good GPU.")
            print("    First-ever run also downloads ~300-500MB of model weights before it starts —")
            print("    that download's progress bar will print below, same as the processing bar.)")
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "--two-stems", "vocals",
            "-n", model_name,
            "-d", device,
            "--out", out_dir,
            audio_path,
        ]
        try:
            # Deliberately NOT capturing output here — Demucs prints its own
            # download %/processing progress bars, and capture_output=True
            # would swallow them until the process exits, making a slow-but-
            # healthy run look identical to a hung one. Letting it inherit
            # the console means you see exactly what it's doing, live.
            result = subprocess.run(cmd)
            vocals_path = os.path.join(out_dir, model_name, base_name, "vocals.wav")
            if result.returncode == 0 and os.path.exists(vocals_path):
                print(f"✅ Vocal isolation complete ({device}).")
                return vocals_path
            print(f"   ⚠️ Demucs on {device} exited with code {result.returncode} "
                  f"and no vocals.wav was produced — see its output above for details.")
        except FileNotFoundError:
            print("❌ Demucs isn't installed. Run: pip install demucs")
            return None
        except Exception as e:
            print(f"   ⚠️ Demucs error on {device}: {e}")

    print("❌ Vocal isolation failed on every device — continuing with the original mixed audio.")
    return None


if __name__ == "__main__":
    try:
        raw_clip = VideoFileClip(INPUT_PATH)
        director = UltimateDirector()
        director.fps = raw_clip.fps
        director.duration = raw_clip.duration
        director.foreground_mode = FOREGROUND_MODE
        director.output_mode = OUTPUT_MODE
        director.enable_heading = ENABLE_HEADING
        director.enable_dual_subs = ENABLE_DUAL_SUBTITLES

        if OUTPUT_MODE == "studio":
            director.CARD_W = CARD_W
            director.CARD_H = CARD_H
            director.CORNER_RADIUS = CORNER_RADIUS
            director.y_offset = CARD_Y_OFFSET
            director.x_offset = (director.TARGET_W - director.CARD_W) // 2

        if OUTPUT_MODE == "studio" and FOREGROUND_MODE == "image":
            print(f"\n🖼️ Loading static image for the foreground card: {STATIC_IMAGE_PATH}")
            static_pil = Image.open(STATIC_IMAGE_PATH).convert("RGB")
            static_img_rgb = np.array(static_pil)
            img_h, img_w = static_img_rgb.shape[:2]

            print("🖱️  FRAME YOUR IMAGE: drag to move (any direction), scroll wheel (or +/-) to zoom, ENTER/SPACE/ESC to confirm.")
            try:
                icx, icy, izoom = pick_manual_crop_position(director, static_img_rgb, initial_zoom=0.7)
            except Exception as e:
                print(f"⚠️ Manual framing window failed ({e}) — defaulting to centered, 100% zoom.")
                icx, icy, izoom = img_w / 2, img_h / 2, 1.0

            icw, ich = director.compute_crop_dims(img_h, img_w, izoom)

            # Grab a slightly LARGER buffer than the chosen framing so the
            # Ken Burns pan/zoom has room to move without ever revealing
            # empty edges. All Ken Burns math happens against this buffer.
            margin = KEN_BURNS_MARGIN
            mcw = min(img_w, int(icw * margin))
            mch = min(img_h, int(ich * margin))
            mx1 = max(0, min(img_w - mcw, int(icx - mcw / 2)))
            my1 = max(0, min(img_h - mch, int(icy - mch * 0.45)))
            margin_crop = static_img_rgb[my1:my1 + mch, mx1:mx1 + mcw]

            dest_w, dest_h = int(director.CARD_W * margin), int(director.CARD_H * margin)
            # Use a sharper upscale filter when the source photo is smaller
            # than the destination — avoids the soft/"vaseline" look on
            # older or lower-res portraits.
            upscaling = (mcw < dest_w) or (mch < dest_h)
            interp = cv2.INTER_LANCZOS4 if upscaling else cv2.INTER_AREA
            director.static_fg_image_full = cv2.resize(margin_crop, (dest_w, dest_h), interpolation=interp)

            # Build a rich, blurred full-canvas backdrop from the photo
            # itself (cover-fit, not just the tight face crop) so image mode
            # never falls back to a black/empty background behind the card.
            bh, bw = static_img_rgb.shape[:2]
            bscale = max(director.TARGET_W / bw, director.TARGET_H / bh)
            rw, rh = int(math.ceil(bw * bscale)) + 4, int(math.ceil(bh * bscale)) + 4
            bg_cover = cv2.resize(static_img_rgb, (rw, rh), interpolation=cv2.INTER_AREA)
            bx = max(0, (rw - director.TARGET_W) // 2)
            by = max(0, (rh - director.TARGET_H) // 2)
            bg_cover = bg_cover[by:by + director.TARGET_H, bx:bx + director.TARGET_W]
            if bg_cover.shape[0] < director.TARGET_H or bg_cover.shape[1] < director.TARGET_W:
                bg_cover = cv2.resize(bg_cover, (director.TARGET_W, director.TARGET_H))
            bg_blurred_full = cv2.GaussianBlur(bg_cover, (0, 0), sigmaX=45, sigmaY=45)
            director.static_bg_small = cv2.resize(
                bg_blurred_full, (director.TARGET_W // 4, director.TARGET_H // 4), interpolation=cv2.INTER_AREA
            )

            print("✅ Image framing locked in — Ken Burns pan/zoom will animate across the video, "
                  "and the backdrop is built from the photo itself.")

        hex_color_code = "SIMPLE"
        if OUTPUT_MODE == "studio":
            themes = director.generate_theme_profiles(raw_clip)

            if ENABLE_COLOR_SHIFT:
                print("🎨 Multi-Act Grading Mode Configured: Background colors will evolve dynamically!")
                chosen_theme = themes[0]
            else:
                SAMPLES_DIR = os.path.join(PROJECT_DIR, "samples")
                os.makedirs(SAMPLES_DIR, exist_ok=True)
                preview_time = min(4.0, raw_clip.duration * 0.2)
                preview_frame = raw_clip.get_frame(preview_time)

                print(f"\n🖼️ Rendering Premium Theme Archetypes inside '{SAMPLES_DIR}'...")
                for i, theme in enumerate(themes):
                    director.apply_theme_colors(theme["bg"], theme["accent"], theme["banner"])
                    sample_render = director.render_single_frame(preview_frame, preview_time, is_sample=True)
                    cv2.imwrite(os.path.join(SAMPLES_DIR, f"theme_option_{i+1}.png"), cv2.cvtColor(sample_render, cv2.COLOR_RGB2BGR))

                print("\n" + "="*50)
                print("⏸️ THEME SELECTOR MENU")
                for i, t in enumerate(themes):
                    print(f"  [{i+1}] Auto-Generated Theme {i+1}")
                print("="*50)

                while True:
                    choice = input("👉 Select a static grading option (1-5): ").strip()
                    if choice in [str(k) for k in range(1, len(themes)+1)]:
                        chosen_theme = themes[int(choice) - 1]
                        break
                    print("⚠️ Invalid entry.")

                print(f"✅ Sample images have been preserved in: {SAMPLES_DIR}")

            hex_color_code = rgb_to_hex(chosen_theme['accent'])
            director.apply_theme_colors(chosen_theme['bg'], chosen_theme['accent'], chosen_theme['banner'])

        if MANUAL_TRACKING:
            print("\n🖱️  MANUAL CAMERA LOCK: choose the starting framing and zoom.")
            print("    Horizontal starts on the detected face as a guide — click and drag")
            print("    in any direction (left/right and up/down) to fine-tune, scroll the")
            print("    wheel (or press +/-) to zoom.")
            print("    You'll be asked again automatically every time the video cuts to a new shot.")
            crop_preview_time = min(4.0, raw_clip.duration * 0.2)
            crop_preview_frame = raw_clip.get_frame(crop_preview_time)
            ph, pw, _ = crop_preview_frame.shape
            try:
                cam_x, cam_y, zoom = pick_manual_crop_position(director, crop_preview_frame)
                director.manual_cam_pos = (cam_x, cam_y)
                director.manual_zoom = zoom
            except Exception as e:
                print(f"⚠️ Manual positioning window failed ({e}) — defaulting to frame center.")
                director.manual_cam_pos = (pw / 2, ph / 2)
                director.manual_zoom = 1.0
            print(f"✅ Starting position set (x={director.manual_cam_pos[0]:.0f}, y={director.manual_cam_pos[1]:.0f}, zoom={int(director.manual_zoom*100)}%)")

            if OUTPUT_MODE == "simple":
                print("\n🌑 DARK GRADIENT SETUP: drag the sliders to set the gradient's height and density — live preview.")
                print("    (Drag Density all the way to 0 to disable the built-in gradient entirely.)")
                gcw, gch = director.get_crop_dims(ph, pw, director.manual_zoom)
                gx1 = max(0, min(pw - gcw, int(director.manual_cam_pos[0] - gcw / 2)))
                gy1 = max(0, min(ph - gch, int(director.manual_cam_pos[1] - gch * 0.45)))
                gradient_preview_frame = cv2.resize(
                    crop_preview_frame[gy1:gy1 + gch, gx1:gx1 + gcw],
                    (director.TARGET_W, director.TARGET_H),
                    interpolation=cv2.INTER_LINEAR
                )
                try:
                    g_start, g_end, g_alpha = pick_gradient_settings(
                        gradient_preview_frame,
                        director.simple_gradient_start_frac,
                        director.simple_gradient_end_frac,
                        director.simple_gradient_max_alpha,
                    )
                    director.simple_gradient_start_frac = g_start
                    director.simple_gradient_end_frac = g_end
                    director.simple_gradient_max_alpha = g_alpha
                except Exception as e:
                    print(f"⚠️ Gradient setup window failed ({e}) — keeping default gradient settings.")
                print(f"✅ Gradient set (start={int(director.simple_gradient_start_frac*100)}%, "
                      f"end={int(director.simple_gradient_end_frac*100)}%, density={director.simple_gradient_max_alpha}/255)")

                # Optional custom overlay PNG — asked right after the
                # generated-gradient setup so the two darkening layers are
                # configured back-to-back in one setup pass.
                pick_overlay_png(director)
        
        audio_target_path = os.path.join(PROJECT_DIR, "final_clean_vocals.wav")
        srt_target_path = os.path.join(PROJECT_DIR, "subtitles.srt")
        original_srt_backup_path = os.path.join(PROJECT_DIR, "subtitles_original.srt")
        
        audio_temp_path = os.path.join(PROJECT_DIR, "temp_raw_audio.wav")
        raw_clip.audio.write_audiofile(audio_temp_path, fps=16000, logger=None)
        shutil.copy(audio_temp_path, audio_target_path)

        # Full-bandwidth (44.1kHz) copy — used by Demucs vocal isolation (if
        # enabled) and, further below, by the studio audio-enhancement chain.
        # Pulled once here unconditionally so the enhancement chain always
        # has real content above 8kHz to work with, instead of the
        # bandlimited 16kHz Whisper copy above.
        audio_hq_path = os.path.join(PROJECT_DIR, "temp_raw_audio_hq.wav")
        raw_clip.audio.write_audiofile(audio_hq_path, fps=44100, logger=None)

        vocals_path = None
        if ENABLE_VOCAL_ISOLATION:
            vocals_path = isolate_vocals(audio_hq_path, PROJECT_DIR, model_name=DEMUCS_MODEL)
            if vocals_path:
                # audio_target_path feeds BOTH the Whisper transcription
                # step below and the final video's audio track, so
                # overwriting it here means the whole rest of the pipeline
                # (subtitles + final mix) automatically uses the cleaned
                # vocal stem with no other changes needed.
                shutil.copy(vocals_path, audio_target_path)
                print("✅ Background music/nasheed stripped — using the isolated vocal track "
                      "for both subtitles and the final render.")

        if ENABLE_SUBTITLES:
            review_and_retry_transcription(director, audio_target_path, srt_target_path)

            # Preserve a copy of the freshly-transcribed native-language SRT
            # BEFORE the user overwrites srt_target_path with their manual
            # translation below — this backup is what supplies the small
            # original-language dual-subtitle line later.
            if ENABLE_DUAL_SUBTITLES:
                shutil.copy(srt_target_path, original_srt_backup_path)
            
            print("\n" + "🛑"*30)
            print("🛑 TRANSLATION INTERCEPT MODE ACTIVATED")
            print("🛑 1. The script will now open your .srt file in Notepad.")
            print("🛑 2. Type your manual translation normally.")
            print("🛑 3. PRO TIP: If your sentence is getting too long, just type the '|' symbol.")
            print("🛑    The script will automatically clear the screen when it hits '|' and perfectly time the next block.")
            print("🛑"*30)
            
            open_in_editor(srt_target_path)
            
            input("\n🟢 Press ENTER when translation is saved to execute final studio render... ")
            
            raw_subs = read_intercepted_srt(srt_target_path)

            # Attach the matching original-language text to each translated
            # block, by position — the intercept step only edits text in
            # place (splitting a block's text with '|' doesn't add/remove
            # SRT blocks), so the block count and order line up 1:1 with
            # the pre-translation backup.
            if ENABLE_DUAL_SUBTITLES:
                original_subs = read_intercepted_srt(original_srt_backup_path)
                if len(original_subs) != len(raw_subs):
                    print(f"⚠️ Original ({len(original_subs)}) and translated ({len(raw_subs)}) subtitle block counts "
                          f"don't match — dual subtitles will pair by position and may be misaligned near the end.")
                for i, sub in enumerate(raw_subs):
                    sub['original'] = original_subs[i]['text'] if i < len(original_subs) else ''

            director.current_subs = director.semantic_chunker(raw_subs)

        if OUTPUT_MODE == "studio":
            print(f"\n🎨 ENGRAVING TEXT COLOR HEX: #{hex_color_code}")
        
        # --- STUDIO AUDIO ENHANCEMENT ---
        # Runs after audio has been extracted/passed through above (raw
        # extraction, optional Demucs vocal isolation) and before final mux.
        # Writes a NEW file — audio_target_path / audio_hq_path are left
        # intact so the raw (or vocal-isolated) audio stays available for
        # comparison. See audio_enhance.py for the filter chain itself.
        enhanced_audio_path = os.path.join(PROJECT_DIR, "final_enhanced_audio.wav")
        if ENABLE_AUDIO_ENHANCE:
            audio_enhance_source = vocals_path if (ENABLE_VOCAL_ISOLATION and vocals_path) else audio_hq_path
            enhanced_audio_path = audio_enhance.enhance_audio(
                source_audio_path=audio_enhance_source,
                output_audio_path=enhanced_audio_path,
                preenhanced_audio_path=(PREENHANCED_AUDIO_PATH or None),
            )
        else:
            print("🎚️ Audio enhancement disabled — using extracted audio unprocessed.")
            enhanced_audio_path = audio_hq_path

        sequence = [raw_clip.fl_image(director.process_main_frame)]
        final_v = sequence[0].set_audio(AudioFileClip(enhanced_audio_path))
        
        base_name, ext = os.path.splitext(os.path.basename(OUTPUT_PATH))
        if not ext:
            # Without a real extension FFmpeg can't pick an output
            # container and fails with "Invalid argument" / "Unable to
            # choose an output format" — default to .mp4 rather than
            # ever writing an extension-less file.
            print("⚠️ Output filename had no extension — defaulting to .mp4")
            ext = ".mp4"
        final_output_path = os.path.join(PROJECT_DIR, f"{base_name}_{hex_color_code}{ext}")
        
        # Audio DSP now runs upstream, in the dedicated audio_enhance.py
        # chain (declip -> denoise -> highpass -> EQ -> exciter -> de-reverb
        # -> compression -> loudnorm) applied to enhanced_audio_path above.
        # The audio handed to write_videofile here is already that finished
        # file, so this only needs a brick-wall safety limiter against any
        # tiny inter-sample peaks the AAC re-encode itself might introduce —
        # not a second pass of tone-shaping.
        audio_filter_chain = "alimiter=limit=0.97"
        
        final_v.write_videofile(
            final_output_path, 
            codec="h264_nvenc", 
            audio_codec="aac", 
            fps=raw_clip.fps, 
            bitrate="8500k", 
            threads=12, 
            logger="bar",
            ffmpeg_params=[
                "-pix_fmt", "yuv420p", 
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-preset", "p4", 
                "-tune", "hq", 
                "-rc", "vbr", 
                "-cq", "19", 
                "-c:a", "aac", "-af", audio_filter_chain
            ]
        )
        
        print("\n📸 Auto-Generating Smart Thumbnail...")
        best_frame = None
        best_score = 0
        best_time = 0
        test_times = [raw_clip.duration * (i/10) for i in range(2, 8)]
        
        for t in test_times:
            frm = raw_clip.get_frame(t)
            gray = cv2.cvtColor(frm, cv2.COLOR_RGB2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score > best_score:
                best_score = score
                best_frame = frm
                best_time = t
                
        thumb_render = director.render_single_frame(best_frame, best_time, is_sample=True)
        thumb_path = os.path.join(PROJECT_DIR, f"{base_name}_THUMBNAIL.png")
        cv2.imwrite(thumb_path, cv2.cvtColor(thumb_render, cv2.COLOR_RGB2BGR))

        caption_path = None
        if ENABLE_CAPTION:
            if ENABLE_SUBTITLES and director.current_subs:
                CAPTION_EXPLANATION = " ".join(
                    c['text'].strip() for c in director.current_subs if c['text'].strip()
                )
            else:
                print("ℹ️ No subtitle transcript available — caption will skip the explanation section.")

            caption_text = build_instagram_caption(
                VIDEO_TITLE,
                SPEAKER_NAME.replace('|', ' – '),
                YT_LINK,
                CLIP_START,
                CLIP_END,
                CAPTION_EXPLANATION
            )
            caption_path = os.path.join(PROJECT_DIR, f"{base_name}_CAPTION.txt")
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption_text)

        print(f"\n🎬 Master production export finished successfully: {final_output_path}")
        print(f"🖼️ Smart Thumbnail Saved: {thumb_path}")
        if caption_path:
            print(f"📝 Instagram Caption Saved: {caption_path}")
        
    except Exception as e: 
        print(f"❌ ERROR DURING EXECUTION: {e}")