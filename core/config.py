from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "Mídia + Vídeo Automático"
WORK_ROOT = Path(os.getenv("WORK_ROOT", "/tmp/foto_video_saas"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
DEFAULT_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "pt")
WHISPER_KEEP_MODEL_LOADED = os.getenv("WHISPER_KEEP_MODEL_LOADED", "0").strip() == "1"

MAX_VIDEO_MINUTES = float(os.getenv("MAX_VIDEO_MINUTES", "10"))
MAX_INSTAGRAM_DOWNLOAD_MB = max(1, int(os.getenv("MAX_INSTAGRAM_DOWNLOAD_MB", "500")))
OUTPUT_CRF = int(os.getenv("OUTPUT_CRF", "21"))
OUTPUT_PRESET = os.getenv("OUTPUT_PRESET", "veryfast")
OUTPUT_AUDIO_BITRATE = os.getenv("OUTPUT_AUDIO_BITRATE", "160k")
CAPTION_FONT = os.getenv("CAPTION_FONT", "DejaVu Sans")

# Proteção contra picos de RAM/CPU. 1920 preserva 1080x1920 em Reels.
OUTPUT_MAX_LONG_EDGE = max(0, int(os.getenv("OUTPUT_MAX_LONG_EDGE", "1920")))
FFMPEG_THREADS = max(1, int(os.getenv("FFMPEG_THREADS", "1")))
BACKGROUND_BLUR_DIVISOR = max(2, int(os.getenv("BACKGROUND_BLUR_DIVISOR", "4")))

# Detector de transição.
TRANSITION_MAX_SCAN_SECONDS = max(5.0, float(os.getenv("TRANSITION_MAX_SCAN_SECONDS", "60")))

# Limpeza de temporários. Mais curto = menos disco ocupado no Railway.
TEMP_MAX_AGE_HOURS = max(0.25, float(os.getenv("TEMP_MAX_AGE_HOURS", "1.5")))

# OCR: menos amostras e recorte da interface do Reel reduzem custo e ruído.
OCR_MAX_SAMPLES = max(3, int(os.getenv("OCR_MAX_SAMPLES", "8")))
OCR_SAMPLE_STEP_SECONDS = max(0.4, float(os.getenv("OCR_SAMPLE_STEP_SECONDS", "0.9")))
OCR_CROP_TOP_PERCENT = min(35.0, max(0.0, float(os.getenv("OCR_CROP_TOP_PERCENT", "5"))))
OCR_CROP_BOTTOM_PERCENT = min(35.0, max(0.0, float(os.getenv("OCR_CROP_BOTTOM_PERCENT", "8"))))
OCR_CROP_SIDE_PERCENT = min(25.0, max(0.0, float(os.getenv("OCR_CROP_SIDE_PERCENT", "1"))))
