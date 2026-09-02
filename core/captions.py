from __future__ import annotations

import difflib
import functools
import gc
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import regex
from PIL import Image, ImageDraw, ImageFont

from .config import (
    CAPTION_FONT,
    DEFAULT_LANGUAGE,
    OCR_CROP_BOTTOM_PERCENT,
    OCR_CROP_SIDE_PERCENT,
    OCR_CROP_TOP_PERCENT,
    OCR_MAX_SAMPLES,
    OCR_SAMPLE_STEP_SECONDS,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_KEEP_MODEL_LOADED,
    WHISPER_MODEL,
)
from .media import MediaError, require_binary, run_command


@dataclass(frozen=True)
class CaptionEvent:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class CaptionOverlay:
    start: float
    end: float
    path: str


_MODEL = None
_MODEL_LOCK = threading.Lock()


def _get_whisper_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MediaError(
                "A transcrição automática não está instalada. Execute 'pip install -r requirements.txt' "
                "ou use a opção de legenda manual."
            ) from exc
        try:
            _MODEL = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
        except Exception as exc:
            message = str(exc)
            if (
                "ConnectError" in message
                or "LocalEntryNotFoundError" in type(exc).__name__
                or "internet connection" in message.lower()
            ):
                raise MediaError(
                    "Não foi possível baixar o modelo de transcrição. Verifique a conexão de internet "
                    "do servidor e tente novamente. No Railway, a primeira transcrição baixa o modelo "
                    "configurado em WHISPER_MODEL."
                ) from exc
            raise MediaError(f"Não foi possível carregar o modelo Whisper: {message}") from exc
        return _MODEL


def _release_whisper_model() -> None:
    """Libera o modelo da RAM em containers pequenos quando configurado."""
    global _MODEL
    if WHISPER_KEEP_MODEL_LOADED:
        return
    with _MODEL_LOCK:
        _MODEL = None
    gc.collect()



def extract_audio_for_transcription(
    video_path: str | Path,
    output_wav: str | Path,
    duration: float,
) -> None:
    ffmpeg = require_binary("ffmpeg")
    run_command(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ],
        timeout=max(120, int(duration * 5)),
    )


def _clean_word(word: str) -> str:
    return re.sub(r"\s+", " ", word).strip()


def _group_words(words: Iterable, *, max_words: int = 5, max_duration: float = 2.3) -> list[CaptionEvent]:
    events: list[CaptionEvent] = []
    current: list = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = " ".join(_clean_word(item.word) for item in current).strip()
        if text:
            events.append(
                CaptionEvent(
                    start=max(0.0, float(current[0].start)),
                    end=max(float(current[-1].end), float(current[0].start) + 0.12),
                    text=text,
                )
            )
        current = []

    for word in words:
        if word.start is None or word.end is None:
            continue
        cleaned = _clean_word(word.word)
        if not cleaned:
            continue
        if current:
            current_duration = float(word.end) - float(current[0].start)
            sentence_break = bool(re.search(r"[.!?…]$", _clean_word(current[-1].word)))
            if len(current) >= max_words or current_duration > max_duration or sentence_break:
                flush()
        current.append(word)
    flush()
    return events


def transcribe_intro(
    video_path: str | Path,
    work_dir: str | Path,
    duration: float,
    *,
    language: str | None = DEFAULT_LANGUAGE,
) -> tuple[list[CaptionEvent], str]:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    wav_path = work / "intro_audio.wav"
    extract_audio_for_transcription(video_path, wav_path, duration)

    model = _get_whisper_model()
    try:
        segments, info = model.transcribe(
            str(wav_path),
            language=language or None,
            beam_size=2,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        words = []
        full_text: list[str] = []
        for segment in segments:
            if segment.text:
                full_text.append(segment.text.strip())
            if segment.words:
                words.extend(segment.words)
        events = _group_words(words)
        detected_language = getattr(info, "language", None) or language or "desconhecido"
    except Exception as exc:
        raise MediaError(f"Falha durante a transcrição automática: {exc}") from exc
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except OSError:
            pass
        _release_whisper_model()

    clipped = [
        CaptionEvent(event.start, min(event.end, duration), clean_review_text(event.text))
        for event in events
        if event.start < duration and event.end > 0
    ]
    return clipped, " ".join(full_text).strip() + f"\nIdioma: {detected_language}"


def manual_caption(text: str, duration: float) -> list[CaptionEvent]:
    # Mantém todo Unicode, inclusive emojis, ZWJ, tons de pele e seletores de variação.
    cleaned = re.sub(r"[\t\r\f\v ]+", " ", text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        return []
    return [CaptionEvent(0.0, max(0.15, duration), cleaned)]


def _fc_match(pattern: str) -> str | None:
    executable = shutil.which("fc-match")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "-f", "%{file}\n", pattern],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    candidate = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    return candidate if candidate and Path(candidate).exists() else None


def _normal_font_path() -> str:
    candidates = [
        _fc_match(f"{CAPTION_FONT}:style=Bold"),
        _fc_match(CAPTION_FONT),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise MediaError("Não foi encontrada uma fonte para renderizar as legendas.")


def _emoji_font_path() -> str | None:
    candidates = [
        _fc_match("Noto Color Emoji"),
        _fc_match("Segoe UI Emoji"),
        _fc_match("Apple Color Emoji"),
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        r"C:\Windows\Fonts\seguiemj.ttf",
        "/System/Library/Fonts/Apple Color Emoji.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def _is_emoji_cluster(cluster: str) -> bool:
    for char in cluster:
        code = ord(char)
        if (
            0x1F000 <= code <= 0x1FAFF
            or 0x2600 <= code <= 0x27BF
            or 0x2300 <= code <= 0x23FF
            or 0x2B00 <= code <= 0x2BFF
            or code in {0x200D, 0x20E3, 0xFE0F}
        ):
            return True
    return False


@functools.lru_cache(maxsize=512)
def _render_emoji(cluster: str, target_height: int, font_path: str) -> Image.Image:
    # Noto Color Emoji possui uma strike bitmap de 109 px. Outras fontes podem ser escaláveis.
    font = None
    for candidate_size in (target_height, 109, 128, 96, 64, 32):
        try:
            font = ImageFont.truetype(font_path, candidate_size)
            break
        except OSError:
            continue
    if font is None:
        raise MediaError("A fonte de emoji instalada não pôde ser carregada.")

    probe = Image.new("RGBA", (320, 320), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    try:
        bbox = draw.textbbox((0, 0), cluster, font=font, embedded_color=True)
    except TypeError:
        bbox = draw.textbbox((0, 0), cluster, font=font)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    glyph = Image.new("RGBA", (width + 12, height + 12), (0, 0, 0, 0))
    glyph_draw = ImageDraw.Draw(glyph)
    position = (6 - bbox[0], 6 - bbox[1])
    try:
        glyph_draw.text(position, cluster, font=font, embedded_color=True)
    except TypeError:
        glyph_draw.text(position, cluster, font=font, fill="white")

    alpha_bbox = glyph.getchannel("A").getbbox()
    if alpha_bbox:
        glyph = glyph.crop(alpha_bbox)
    if glyph.height != target_height and glyph.height > 0:
        target_width = max(1, int(round(glyph.width * target_height / glyph.height)))
        glyph = glyph.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return glyph


def _clusters(text: str) -> list[str]:
    return regex.findall(r"\X", text)


def _cluster_width(
    cluster: str,
    normal_font: ImageFont.FreeTypeFont,
    emoji_height: int,
    emoji_font_path: str | None,
) -> float:
    if _is_emoji_cluster(cluster) and emoji_font_path:
        return float(_render_emoji(cluster, emoji_height, emoji_font_path).width)
    return float(normal_font.getlength(cluster))


def _token_clusters(text: str) -> list[list[str]]:
    tokens: list[list[str]] = []
    for token in regex.findall(r"\s+|\S+", text):
        tokens.append(_clusters(token))
    return tokens


def _wrap_lines(
    text: str,
    *,
    normal_font: ImageFont.FreeTypeFont,
    emoji_height: int,
    emoji_font_path: str | None,
    max_width: int,
) -> tuple[list[list[str]], list[float]]:
    lines: list[list[str]] = []
    widths: list[float] = []

    paragraphs = text.split("\n")
    for paragraph_index, paragraph in enumerate(paragraphs):
        current: list[str] = []
        current_width = 0.0
        for token in _token_clusters(paragraph):
            token_width = sum(
                _cluster_width(cluster, normal_font, emoji_height, emoji_font_path)
                for cluster in token
            )
            is_space = all(cluster.isspace() for cluster in token)
            if is_space and not current:
                continue

            if current and current_width + token_width > max_width and not is_space:
                while current and current[-1].isspace():
                    removed = current.pop()
                    current_width -= _cluster_width(removed, normal_font, emoji_height, emoji_font_path)
                lines.append(current)
                widths.append(max(0.0, current_width))
                current = []
                current_width = 0.0

            if token_width > max_width and not is_space:
                for cluster in token:
                    width = _cluster_width(cluster, normal_font, emoji_height, emoji_font_path)
                    if current and current_width + width > max_width:
                        lines.append(current)
                        widths.append(current_width)
                        current = []
                        current_width = 0.0
                    current.append(cluster)
                    current_width += width
            else:
                current.extend(token)
                current_width += token_width

        while current and current[-1].isspace():
            removed = current.pop()
            current_width -= _cluster_width(removed, normal_font, emoji_height, emoji_font_path)
        lines.append(current)
        widths.append(max(0.0, current_width))

        if paragraph_index < len(paragraphs) - 1 and not current:
            # Preserva uma linha vazia explícita sem criar vazias infinitas.
            pass

    return lines or [[]], widths or [0.0]


def _render_caption_canvas(
    text: str,
    *,
    width: int,
    height: int,
    font_percent: float,
    position: str,
) -> Image.Image:
    normal_path = _normal_font_path()
    emoji_path = _emoji_font_path()
    requested_size = max(24, int(round(height * max(2.0, min(font_percent, 12.0)) / 100.0)))
    max_text_width = int(round(width * 0.90))

    font_size = requested_size
    while True:
        normal_font = ImageFont.truetype(normal_path, font_size)
        emoji_height = max(18, int(round(font_size * 1.18)))
        lines, line_widths = _wrap_lines(
            text,
            normal_font=normal_font,
            emoji_height=emoji_height,
            emoji_font_path=emoji_path,
            max_width=max_text_width,
        )
        if len(lines) <= 3 or font_size <= 22:
            break
        font_size = max(22, int(round(font_size * 0.90)))

    if len(lines) > 3:
        merged = lines[:2]
        merged.append([cluster for line in lines[2:] for cluster in ([" "] + line)])
        lines = merged
        line_widths = [
            sum(_cluster_width(cluster, normal_font, emoji_height, emoji_path) for cluster in line)
            for line in lines
        ]

    ascent, descent = normal_font.getmetrics()
    line_height = max(int(round(font_size * 1.34)), emoji_height + max(2, descent // 3))
    block_height = line_height * len(lines)
    margin_v = max(20, int(round(height * 0.08)))
    if position == "Centro inferior":
        top = max(0, height - margin_v - block_height)
    elif position == "Centro superior":
        top = margin_v
    else:
        top = max(0, (height - block_height) // 2)

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    outline = max(2, int(round(font_size * 0.075)))

    for line_index, (line, line_width) in enumerate(zip(lines, line_widths)):
        x = (width - line_width) / 2.0
        baseline = top + line_index * line_height + ascent
        for cluster in line:
            if _is_emoji_cluster(cluster) and emoji_path:
                emoji_image = _render_emoji(cluster, emoji_height, emoji_path)
                emoji_top = int(round(baseline - emoji_image.height + max(0, descent * 0.20)))
                emoji_x = int(round(x))
                # Sombra curta para manter legibilidade em fundos claros.
                shadow = Image.new("RGBA", emoji_image.size, (0, 0, 0, 0))
                shadow.putalpha(emoji_image.getchannel("A").point(lambda value: int(value * 0.65)))
                canvas.alpha_composite(shadow, (emoji_x + outline, emoji_top + outline))
                canvas.alpha_composite(emoji_image, (emoji_x, emoji_top))
                x += emoji_image.width
            else:
                if not cluster.isspace():
                    draw.text(
                        (x, baseline),
                        cluster,
                        font=normal_font,
                        fill=(255, 255, 255, 255),
                        stroke_width=outline,
                        stroke_fill=(0, 0, 0, 255),
                        anchor="ls",
                    )
                x += normal_font.getlength(cluster)

    return canvas


def render_caption_overlays(
    events: list[CaptionEvent],
    work_dir: str | Path,
    *,
    width: int,
    height: int,
    font_percent: float,
    position: str,
) -> list[CaptionOverlay]:
    output_dir = Path(work_dir) / "caption_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays: list[CaptionOverlay] = []

    for index, event in enumerate(events):
        canvas = _render_caption_canvas(
            event.text,
            width=width,
            height=height,
            font_percent=font_percent,
            position=position,
        )
        path = output_dir / f"caption_{index:03d}.png"
        canvas.save(path, format="PNG", optimize=True)
        overlays.append(CaptionOverlay(event.start, event.end, str(path)))

    return overlays


_COMMON_PT_ACCENTS = {
    "voce": "você",
    "voces": "vocês",
    "nao": "não",
    "tambem": "também",
    "ninguem": "ninguém",
    "alem": "além",
    "porem": "porém",
    "possivel": "possível",
    "incrivel": "incrível",
}


def _preserve_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def clean_review_text(text: str) -> str:
    """Limpeza conservadora de artefatos comuns de OCR/transcrição.

    Corrige espaços/pontuação e alguns acentos muito inequívocos em português.
    O resultado continua editável na interface antes de qualquer renderização.
    """
    value = (text or "").replace("\u00a0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"([,.;:!?])(?=[A-Za-zÀ-ÿ])", r"\1 ", value)
    value = re.sub(r"([!?.,])\1{2,}", r"\1\1", value)
    value = re.sub(r"\n{3,}", "\n\n", value)

    def fix_word(match):
        original = match.group(0)
        replacement = _COMMON_PT_ACCENTS.get(original.lower())
        return _preserve_case(replacement, original) if replacement else original

    value = re.sub(r"\b[A-Za-zÀ-ÿ]+\b", fix_word, value)
    return value.strip()


# ====================== LEITURA DA LEGENDA QUEIMADA (OCR) ======================

_OCR_LANGUAGE_MAP = {"pt": "por", "en": "eng", "es": "spa"}

# Regiões opcionais. A opção automática usa quase todo o vídeo e deixa o
# consenso espacial/temporal escolher o bloco que realmente se comporta como
# legenda. As regiões manuais são um escape rápido para vídeos fora do padrão.
_OCR_REGION_RANGES = {
    "Parte superior": (0.04, 0.50),
    "Centro": (0.18, 0.80),
    "Parte inferior": (0.46, 0.94),
    "Tela quase inteira": (0.00, 1.00),
}


@dataclass(frozen=True)
class _OcrCandidate:
    text: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


def _ocr_grab_frames(video_path: str | Path, timestamps: list[float]) -> list[tuple[float, object]]:
    """Lê todos os quadros com um único VideoCapture (bem mais barato no Railway)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames: list[tuple[float, object]] = []
    try:
        for time_sec in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append((float(time_sec), frame))
    finally:
        cap.release()
    return frames


def _ocr_crop(frame, region: str):
    """Remove margens e, quando solicitado, limita a busca a uma faixa vertical."""
    height, width = frame.shape[:2]
    top = int(height * OCR_CROP_TOP_PERCENT / 100.0)
    bottom = int(height * (1.0 - OCR_CROP_BOTTOM_PERCENT / 100.0))
    side = int(width * OCR_CROP_SIDE_PERCENT / 100.0)
    right = max(side + 2, width - side)
    bottom = max(top + 2, bottom)
    safe = frame[top:bottom, side:right]

    y0_ratio, y1_ratio = _OCR_REGION_RANGES.get(region, (0.0, 1.0))
    safe_h = safe.shape[0]
    y0 = max(0, min(safe_h - 2, int(round(safe_h * y0_ratio))))
    y1 = max(y0 + 2, min(safe_h, int(round(safe_h * y1_ratio))))
    return safe[y0:y1, :]


def _ocr_variants(frame, region: str = "Automática (recomendado)"):
    """Pré-processamento conservador: nitidez + contraste e binarização adaptativa."""
    import cv2

    frame = _ocr_crop(frame, region)
    height, width = frame.shape[:2]

    # Abaixo de ~900 px letras pequenas perdem muito detalhe; acima de ~1200 px
    # o custo cresce bastante e o ganho costuma ser pequeno.
    if width < 900:
        scale = 900.0 / max(width, 1)
        frame = cv2.resize(frame, (900, max(2, int(round(height * scale)))), interpolation=cv2.INTER_CUBIC)
    elif width > 1200:
        scale = 1200.0 / width
        frame = cv2.resize(frame, (1200, max(2, int(round(height * scale)))), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sharp = cv2.addWeighted(clahe, 1.55, cv2.GaussianBlur(clahe, (0, 0), 1.0), -0.55, 0)

    binary = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    # Tesseract tende a funcionar melhor com fundo majoritariamente claro.
    if float(binary.mean()) < 127.0:
        binary = 255 - binary

    return [sharp, binary]


def _token_is_plausible(token: str) -> bool:
    compact = re.sub(r"\s+", "", token or "")
    if not compact or not re.search(r"[A-Za-zÀ-ÿ0-9]", compact):
        return False
    meaningful = len(re.findall(r"[A-Za-zÀ-ÿ0-9@#'’.,!?%+-]", compact))
    return meaningful / max(len(compact), 1) >= 0.55


def _candidate_from_words(words: list[dict], image_w: int, image_h: int) -> _OcrCandidate | None:
    if not words:
        return None
    text = clean_review_text(" ".join(item["text"] for item in words)).strip()
    flat = re.sub(r"\s+", " ", text)
    letters = len(re.findall(r"[A-Za-zÀ-ÿ]", flat))
    digits = len(re.findall(r"[0-9]", flat))
    tokens = re.findall(r"\S+", flat)
    if letters + digits < 3 or not tokens:
        return None

    weights = [max(1, len(re.sub(r"\W", "", item["text"]))) for item in words]
    confidence = sum(float(item["conf"]) * weight for item, weight in zip(words, weights)) / max(sum(weights), 1)
    x0 = min(item["left"] for item in words)
    y0 = min(item["top"] for item in words)
    x1 = max(item["left"] + item["width"] for item in words)
    y1 = max(item["top"] + item["height"] for item in words)
    return _OcrCandidate(
        text=flat,
        confidence=confidence,
        x0=max(0.0, min(1.0, x0 / max(image_w, 1))),
        y0=max(0.0, min(1.0, y0 / max(image_h, 1))),
        x1=max(0.0, min(1.0, x1 / max(image_w, 1))),
        y1=max(0.0, min(1.0, y1 / max(image_h, 1))),
    )


def _can_merge_candidates(first: _OcrCandidate, second: _OcrCandidate) -> bool:
    if second.y0 < first.y0:
        first, second = second, first
    vertical_gap = max(0.0, second.y0 - first.y1)
    horizontal_overlap = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    overlap_ratio = horizontal_overlap / max(min(first.width, second.width), 1e-6)
    center_distance = abs(first.cx - second.cx)
    return vertical_gap <= max(0.055, 1.7 * max(first.height, second.height)) and (
        overlap_ratio >= 0.16 or center_distance <= 0.22
    )


def _merge_spatial_candidates(parts: list[_OcrCandidate]) -> _OcrCandidate:
    parts = sorted(parts, key=lambda item: (item.y0, item.x0))
    text = clean_review_text(" ".join(item.text for item in parts))
    weights = [max(1, len(re.findall(r"[A-Za-zÀ-ÿ0-9]", item.text))) for item in parts]
    confidence = sum(item.confidence * weight for item, weight in zip(parts, weights)) / max(sum(weights), 1)
    return _OcrCandidate(
        text=text,
        confidence=confidence,
        x0=min(item.x0 for item in parts),
        y0=min(item.y0 for item in parts),
        x1=max(item.x1 for item in parts),
        y1=max(item.y1 for item in parts),
    )


def _ocr_frame_candidates(image, lang: str) -> list[_OcrCandidate]:
    """Retorna blocos de texto com posição, em vez de achatar toda a tela em uma frase."""
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config="--oem 1 --psm 11 -c preserve_interword_spaces=1",
        output_type=Output.DICT,
    )
    image_h, image_w = image.shape[:2]
    lines: dict[tuple[int, int, int], list[dict]] = {}
    for index in range(len(data.get("text", []))):
        token = (data["text"][index] or "").strip()
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if confidence < 52 or not _token_is_plausible(token):
            continue
        key = (int(data["block_num"][index]), int(data["par_num"][index]), int(data["line_num"][index]))
        lines.setdefault(key, []).append(
            {
                "text": token,
                "conf": confidence,
                "left": int(data["left"][index]),
                "top": int(data["top"][index]),
                "width": int(data["width"][index]),
                "height": int(data["height"][index]),
            }
        )

    line_candidates = [
        candidate
        for candidate in (_candidate_from_words(words, image_w, image_h) for words in lines.values())
        if candidate is not None
    ]
    line_candidates.sort(key=lambda item: (item.y0, item.x0))

    # Além de linhas individuais, testa blocos de até 3 linhas próximas. Isso
    # recupera legendas quebradas sem anexar textos distantes do cenário.
    candidates = list(line_candidates)
    total = len(line_candidates)
    for start in range(total):
        parts = [line_candidates[start]]
        for end in range(start + 1, min(total, start + 3)):
            if not _can_merge_candidates(parts[-1], line_candidates[end]):
                break
            parts.append(line_candidates[end])
            candidates.append(_merge_spatial_candidates(parts))

    # Remove duplicatas do mesmo quadro/variante e limita a quantidade para não
    # deixar placas/logos de fundo dominarem o consenso temporal.
    unique: list[_OcrCandidate] = []
    for candidate in sorted(candidates, key=lambda item: _ocr_candidate_score(item.text, item.confidence, item), reverse=True):
        if any(
            _ocr_text_compatibility(candidate.text, old.text) >= 0.90
            and abs(candidate.cy - old.cy) <= 0.035
            and abs(candidate.cx - old.cx) <= 0.08
            for old in unique
        ):
            continue
        unique.append(candidate)
        if len(unique) >= 12:
            break
    return unique


def _word_tokens(value: str) -> list[str]:
    return re.findall(r"\S+", clean_review_text(re.sub(r"\s+", " ", value or "")))


def _token_key(token: str) -> str:
    return re.sub(r"[^a-z0-9à-ÿ]", "", token.lower())


def _merge_caption_candidates(texts: list[str]) -> str:
    """Une apenas prefixos/sufixos com sobreposição literal de pelo menos 2 palavras."""
    clean = [clean_review_text(re.sub(r"\s+", " ", t or "")).strip() for t in texts]
    clean = [t for t in clean if t]
    if not clean:
        return ""

    base = max(clean, key=lambda t: (len(_word_tokens(t)), len(t)))
    base_tokens = _word_tokens(base)

    for candidate in sorted(clean, key=lambda t: (len(_word_tokens(t)), len(t)), reverse=True):
        cand_tokens = _word_tokens(candidate)
        if not cand_tokens or candidate == base:
            continue

        base_keys = [_token_key(t) for t in base_tokens]
        cand_keys = [_token_key(t) for t in cand_tokens]
        base_norm = " ".join(base_keys)
        cand_norm = " ".join(cand_keys)
        if cand_norm and cand_norm in base_norm:
            continue
        if base_norm and base_norm in cand_norm:
            base_tokens = cand_tokens
            continue

        max_overlap = min(len(base_tokens), len(cand_tokens), 12)
        for overlap in range(max_overlap, 1, -1):
            if cand_keys[-overlap:] == base_keys[:overlap]:
                base_tokens = cand_tokens[:-overlap] + base_tokens
                break
            if base_keys[-overlap:] == cand_keys[:overlap]:
                base_tokens = base_tokens + cand_tokens[overlap:]
                break

    return clean_review_text(" ".join(base_tokens))


def _normalized_ocr(value: str) -> str:
    value = clean_review_text(re.sub(r"\s+", " ", value or "")).lower().strip()
    return re.sub(r"[^a-z0-9à-ÿ ]+", "", value)


def _ocr_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _ocr_text_compatibility(a: str, b: str) -> float:
    """Similaridade tolerante a palavra inicial/final perdida e pequenos erros de OCR."""
    na, nb = _normalized_ocr(a), _normalized_ocr(b)
    if not na or not nb:
        return 0.0
    sequence = _ocr_similarity(na, nb)
    ta = [_token_key(t) for t in _word_tokens(a) if _token_key(t)]
    tb = [_token_key(t) for t in _word_tokens(b) if _token_key(t)]
    if not ta or not tb:
        return sequence
    sa, sb = set(ta), set(tb)
    jaccard = len(sa & sb) / max(len(sa | sb), 1)
    containment = len(sa & sb) / max(min(len(sa), len(sb)), 1)
    if " ".join(ta) in " ".join(tb) or " ".join(tb) in " ".join(ta):
        containment = max(containment, 0.92)
    return max(sequence, 0.58 * containment + 0.42 * jaccard)


def _bbox_iou(a: _OcrCandidate, b: _OcrCandidate) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = a.width * a.height + b.width * b.height - inter
    return inter / max(union, 1e-6)


def _ocr_spatial_compatibility(a: _OcrCandidate, b: _OcrCandidate) -> bool:
    return _bbox_iou(a, b) >= 0.10 or (abs(a.cy - b.cy) <= 0.065 and abs(a.cx - b.cx) <= 0.18)


def _ocr_candidate_score(text: str, confidence: float, candidate: _OcrCandidate | None = None) -> float:
    """Qualidade de uma leitura isolada sem recompensar texto aleatório ilimitadamente."""
    flat = clean_review_text(re.sub(r"\s+", " ", text or ""))
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", flat)
    letters = len(re.findall(r"[A-Za-zÀ-ÿ]", flat))
    score = float(confidence) * 0.52 + min(len(words), 12) * 2.4 + min(letters, 90) * 0.045
    if len(words) > 18:
        score -= (len(words) - 18) * 1.8
    if candidate is not None:
        edge = min(candidate.cx, 1.0 - candidate.cx, candidate.cy, 1.0 - candidate.cy)
        score += min(1.0, edge / 0.20) * 4.0
        if len(words) <= 1 and candidate.width < 0.35 and edge < 0.13:
            score -= 14.0
    return score


def _best_frame_candidates(frame, lang: str, region: str) -> list[_OcrCandidate]:
    merged: list[_OcrCandidate] = []
    for variant in _ocr_variants(frame, region):
        for candidate in _ocr_frame_candidates(variant, lang):
            duplicate_index = next(
                (
                    index
                    for index, old in enumerate(merged)
                    if _ocr_text_compatibility(candidate.text, old.text) >= 0.86
                    and _ocr_spatial_compatibility(candidate, old)
                ),
                None,
            )
            if duplicate_index is None:
                merged.append(candidate)
            elif _ocr_candidate_score(candidate.text, candidate.confidence, candidate) > _ocr_candidate_score(
                merged[duplicate_index].text, merged[duplicate_index].confidence, merged[duplicate_index]
            ):
                merged[duplicate_index] = candidate
    return sorted(merged, key=lambda item: _ocr_candidate_score(item.text, item.confidence, item), reverse=True)[:10]


def _choose_ocr_cluster(per_frame: list[list[_OcrCandidate]]) -> tuple[list[_OcrCandidate], int] | None:
    """Escolhe texto estável no MESMO lugar em quadros diferentes."""
    clusters: list[dict] = []
    frames_with_text = sum(bool(items) for items in per_frame)
    for frame_index, candidates in enumerate(per_frame):
        for candidate in candidates:
            best_cluster = None
            best_match = 0.0
            for cluster in clusters:
                representatives = list(cluster["by_frame"].values())
                compatible = [
                    old for old in representatives
                    if _ocr_spatial_compatibility(candidate, old)
                ]
                if not compatible:
                    continue
                match = max(_ocr_text_compatibility(candidate.text, old.text) for old in compatible)
                if match >= 0.48 and match > best_match:
                    best_match = match
                    best_cluster = cluster
            if best_cluster is None:
                clusters.append({"by_frame": {frame_index: candidate}})
            else:
                current = best_cluster["by_frame"].get(frame_index)
                if current is None or _ocr_candidate_score(candidate.text, candidate.confidence, candidate) > _ocr_candidate_score(
                    current.text, current.confidence, current
                ):
                    best_cluster["by_frame"][frame_index] = candidate

    if not clusters:
        return None

    def rank(cluster: dict) -> float:
        items = list(cluster["by_frame"].values())
        persistence = len(items) / max(frames_with_text, 1)
        best = max(items, key=lambda item: _ocr_candidate_score(item.text, item.confidence, item))
        words = len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", best.text))
        avg_conf = sum(item.confidence for item in items) / len(items)
        edge = min(best.cx, 1.0 - best.cx, best.cy, 1.0 - best.cy)
        edge_penalty = 0.0
        if edge < 0.10:
            edge_penalty += 15.0
        if words <= 1 and best.width < 0.35:
            edge_penalty += 12.0
        return persistence * 100.0 + min(words, 14) * 3.0 + avg_conf * 0.24 - edge_penalty

    chosen = max(clusters, key=rank)
    items = list(chosen["by_frame"].values())
    minimum_hits = 1 if frames_with_text <= 1 else max(2, int(round(frames_with_text * 0.28)))
    if len(items) < minimum_hits:
        return None
    return items, frames_with_text


def _select_ocr_language(language: str, available: set[str]) -> str:
    requested = (language or "").strip().lower()
    preferred = _OCR_LANGUAGE_MAP.get(requested)
    if preferred and preferred in available:
        # Se o usuário escolheu explicitamente Português, não mistura inglês.
        # Isso reduz bastante falsos positivos e palavras absurdas.
        return preferred
    if requested:
        return "eng" if "eng" in available else next(iter(available), "eng")
    # Em modo automático, português + inglês cobre a maioria dos Reels usados na operação.
    auto = [code for code in ("por", "eng") if code in available]
    return "+".join(auto) or next(iter(available), "eng")


def read_burned_caption(
    video_path: str | Path,
    duration: float,
    *,
    language: str = "",
    region: str = "Automática (recomendado)",
) -> tuple[list[CaptionEvent], str]:
    """Lê uma legenda FIXA usando consenso de texto + posição em vários quadros.

    A lógica antiga achatava todo o frame em uma frase e favorecia a leitura com
    mais palavras, o que fazia texto de cenário, watermark e ruído entrarem no
    resultado. Agora cada bloco mantém sua posição e só ganha força se reaparece
    no mesmo lugar ao longo de quadros diferentes.
    """
    try:
        import pytesseract
    except ImportError as exc:
        raise MediaError(
            "A leitura da legenda do vídeo requer o pacote 'pytesseract'. "
            "Execute 'pip install -r requirements.txt'."
        ) from exc

    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception as exc:
        raise MediaError(
            "O programa Tesseract OCR não foi encontrado. No Docker/Railway ele é instalado automaticamente."
        ) from exc

    lang = _select_ocr_language(language, available)
    duration = max(0.05, float(duration))
    max_samples = max(3, OCR_MAX_SAMPLES)

    # Evita 0.00s (fade/animação) e também o último instante colado na transição.
    start = min(max(0.10, duration * 0.04), max(0.01, duration * 0.30))
    end = max(start, duration - min(0.10, duration * 0.04))
    sample_count = max(3, min(max_samples, int(duration / OCR_SAMPLE_STEP_SECONDS) + 2))
    if sample_count <= 1 or end <= start + 0.03:
        timestamps = [duration * 0.5]
    else:
        timestamps = [start + (end - start) * index / (sample_count - 1) for index in range(sample_count)]

    frames = _ocr_grab_frames(video_path, timestamps)
    if not frames:
        return [], ""

    per_frame: list[list[_OcrCandidate]] = []
    for _timestamp, frame in frames:
        try:
            per_frame.append(_best_frame_candidates(frame, lang, region))
        except Exception as exc:
            raise MediaError(f"Falha ao ler a legenda do vídeo: {exc}") from exc

    chosen_info = _choose_ocr_cluster(per_frame)
    if chosen_info is None:
        return [], (
            "OCR não encontrou um texto estável com confiança suficiente. "
            "Isso é intencional: é melhor deixar o campo vazio do que inventar uma legenda."
        )

    items, frames_with_text = chosen_info
    # Usa apenas leituras realmente compatíveis com o melhor grupo e funde
    # prefixos/sufixos quando há sobreposição segura.
    candidate_texts = [item.text for item in items]
    merged = _merge_caption_candidates(candidate_texts)
    best_item = max(items, key=lambda item: _ocr_candidate_score(item.text, item.confidence, item))
    if len(_word_tokens(best_item.text)) > len(_word_tokens(merged)):
        merged = best_item.text
    merged = clean_review_text(re.sub(r"\s+", " ", merged)).strip()

    words = len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", merged))
    avg_conf = sum(item.confidence for item in items) / max(len(items), 1)
    persistence = len(items) / max(frames_with_text, 1)

    # Última barreira anti-lixo. Textos muito curtos, encostados na borda e de
    # baixa confiança são quase sempre watermark/logo, não a legenda principal.
    if not merged or avg_conf < 56:
        return [], "OCR rejeitou uma leitura instável/baixa confiança. Revise manualmente."
    if words <= 1 and best_item.width < 0.32 and min(best_item.cx, 1 - best_item.cx, best_item.cy, 1 - best_item.cy) < 0.12:
        return [], "OCR encontrou apenas um provável watermark/logo e preferiu não usá-lo como legenda."

    event = CaptionEvent(start=0.0, end=max(0.15, duration), text=merged)
    summary = (
        f"Legenda fixa detectada: {merged}\n"
        f"Confiança média {avg_conf:.0f}% • apareceu em {len(items)}/{max(frames_with_text, 1)} quadros úteis "
        f"({persistence:.0%}) • região: {region}."
    )
    return [event], summary
