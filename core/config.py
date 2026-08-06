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

MAX_VIDEO_MINUTES = float(os.getenv("MAX_VIDEO_MINUTES", "10"))
MAX_INSTAGRAM_DOWNLOAD_MB = max(1, int(os.getenv("MAX_INSTAGRAM_DOWNLOAD_MB", "500")))
OUTPUT_CRF = int(os.getenv("OUTPUT_CRF", "20"))
OUTPUT_PRESET = os.getenv("OUTPUT_PRESET", "veryfast")
OUTPUT_AUDIO_BITRATE = os.getenv("OUTPUT_AUDIO_BITRATE", "192k")
CAPTION_FONT = os.getenv("CAPTION_FONT", "DejaVu Sans")

# Proteção contra picos de RAM em containers pequenos.
OUTPUT_MAX_LONG_EDGE = max(0, int(os.getenv("OUTPUT_MAX_LONG_EDGE", "1920")))
FFMPEG_THREADS = max(1, int(os.getenv("FFMPEG_THREADS", "1")))
BACKGROUND_BLUR_DIVISOR = max(2, int(os.getenv("BACKGROUND_BLUR_DIVISOR", "4")))

# O detector procura a primeira troca dentro deste intervalo.
TRANSITION_MAX_SCAN_SECONDS = max(5.0, float(os.getenv("TRANSITION_MAX_SCAN_SECONDS", "90")))

# Arquivos temporários são apagados após este tempo.
TEMP_MAX_AGE_HOURS = float(os.getenv("TEMP_MAX_AGE_HOURS", "3"))
