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
    value = (text or "").replace("\u00a0", " ").replace("|", "I")
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


@dataclass(frozen=True)
class _OCRLineReading:
    """Uma linha encontrada por OCR em um quadro, com posição normalizada 0-1."""

    sample_index: int
    text: str
    confidence: float
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


@dataclass(frozen=True)
class _OCRPersistentLine:
    text: str
    confidence: float
    hits: int
    total_samples: int
    first_sample: int
    x: float
    y: float
    w: float
    h: float

    @property
    def persistence(self) -> float:
        return self.hits / max(self.total_samples, 1)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0



def _ocr_grab_frame(video_path: str | Path, time_sec: float):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()



def _ocr_variants(frame):
    """Prepara o quadro para OCR sem destruir a posição original do texto.

    Diferente da versão antiga, o recorte padrão é mínimo. Vídeo baixado do
    Instagram não contém a interface do aplicativo; cortar 5-8% podia amputar
    justamente uma legenda colocada perto da borda.
    """
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    top = int(height * OCR_CROP_TOP_PERCENT / 100.0)
    bottom = int(height * (1.0 - OCR_CROP_BOTTOM_PERCENT / 100.0))
    side = int(width * OCR_CROP_SIDE_PERCENT / 100.0)
    right = max(side + 2, width - side)
    bottom = max(top + 2, bottom)
    frame = frame[top:bottom, side:right]

    height, width = frame.shape[:2]
    # ~1080 px costuma preservar bem texto de Reels sem explodir CPU no Railway.
    target_width = 1080
    if width < 820:
        scale = target_width / max(width, 1)
        frame = cv2.resize(
            frame,
            (target_width, max(2, int(round(height * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
    elif width > 1280:
        scale = 1280.0 / width
        frame = cv2.resize(
            frame,
            (1280, max(2, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Contraste local ajuda texto branco/preto com sombra sem binarizar demais o fundo.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Segunda variante especializada em texto claro, muito comum em Reels.
    white = (gray >= 190).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    isolated = 255 - white
    return [clahe, isolated]



def _ocr_line_quality(text: str) -> bool:
    flat = clean_review_text(re.sub(r"\s+", " ", text or "")).strip()
    if not flat:
        return False
    letters = re.findall(r"[A-Za-zÀ-ÿ]", flat)
    alnum = re.findall(r"[A-Za-zÀ-ÿ0-9]", flat)
    if len(alnum) < 2 or len(letters) < 2:
        return False
    # Evita lixo típico do OCR: uma sequência enorme sem nenhuma palavra útil.
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", flat)
    if not words:
        return False
    return True



def _ocr_lines(image, lang: str, sample_index: int) -> list[_OCRLineReading]:
    import pytesseract
    from pytesseract import Output

    # PSM 11 procura texto esparso em posições variadas; é melhor para overlays
    # do que tratar o quadro inteiro como um único parágrafo (PSM 6).
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config="--oem 1 --psm 11 -c preserve_interword_spaces=1",
        output_type=Output.DICT,
    )
    image_h, image_w = image.shape[:2]
    grouped: dict[tuple[int, int, int], list[dict]] = {}

    for index in range(len(data.get("text", []))):
        token = (data["text"][index] or "").strip()
        try:
            confidence = float(data["conf"][index])
            left = int(data["left"][index])
            top = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
        except (TypeError, ValueError, KeyError):
            continue

        # O limiar individual é propositalmente moderado. A confiança verdadeira
        # vem da repetição temporal; assim não perdemos uma palavra por compressão.
        if not token or confidence < 42:
            continue
        if not re.search(r"[A-Za-zÀ-ÿ0-9]", token):
            continue
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        grouped.setdefault(key, []).append(
            {
                "token": token,
                "confidence": confidence,
                "left": left,
                "top": top,
                "right": left + max(width, 1),
                "bottom": top + max(height, 1),
            }
        )

    results: list[_OCRLineReading] = []
    for items in grouped.values():
        items.sort(key=lambda item: item["left"])
        text = clean_review_text(" ".join(item["token"] for item in items))
        if not _ocr_line_quality(text):
            continue

        left = min(item["left"] for item in items)
        top = min(item["top"] for item in items)
        right = max(item["right"] for item in items)
        bottom = max(item["bottom"] for item in items)
        confidence = sum(item["confidence"] for item in items) / len(items)
        w = max(1, right - left) / max(image_w, 1)
        h = max(1, bottom - top) / max(image_h, 1)

        # Muito pequeno tende a ser logo/marca d'água/ruído. Ainda mantemos um
        # limiar baixo porque a persistência temporal fará a filtragem principal.
        if h < 0.008 or w < 0.018:
            continue

        results.append(
            _OCRLineReading(
                sample_index=sample_index,
                text=text,
                confidence=confidence,
                x=max(0.0, min(1.0, left / max(image_w, 1))),
                y=max(0.0, min(1.0, top / max(image_h, 1))),
                w=max(0.0, min(1.0, w)),
                h=max(0.0, min(1.0, h)),
            )
        )
    return results



def _word_tokens(value: str) -> list[str]:
    return re.findall(r"\S+", clean_review_text(re.sub(r"\s+", " ", value or "")))



def _token_key(token: str) -> str:
    return re.sub(r"[^a-z0-9à-ÿ]", "", token.lower())



def _normalized_ocr(value: str) -> str:
    return " ".join(_token_key(token) for token in _word_tokens(value) if _token_key(token)).strip()



def _ocr_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()



def _token_overlap(a: str, b: str) -> float:
    a_set = {token for token in _normalized_ocr(a).split() if token}
    b_set = {token for token in _normalized_ocr(b).split() if token}
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(1, min(len(a_set), len(b_set)))



def _texts_match(a: str, b: str) -> bool:
    na = _normalized_ocr(a)
    nb = _normalized_ocr(b)
    if not na or not nb:
        return False
    similarity = _ocr_similarity(na, nb)
    overlap = _token_overlap(na, nb)
    if similarity >= 0.58 or overlap >= 0.66:
        return True
    # OCR frequentemente perde a primeira/última palavra de uma mesma frase.
    return (na in nb or nb in na) and min(len(na), len(nb)) >= 5



def _horizontal_overlap(a: _OCRLineReading, b: _OCRLineReading) -> float:
    left = max(a.x, b.x)
    right = min(a.x + a.w, b.x + b.w)
    overlap = max(0.0, right - left)
    return overlap / max(1e-6, min(a.w, b.w))



def _same_line_position(a: _OCRLineReading, b: _OCRLineReading) -> bool:
    vertical_tol = max(0.025, 1.6 * max(a.h, b.h))
    if abs(a.cy - b.cy) > vertical_tol:
        return False
    return _horizontal_overlap(a, b) >= 0.18 or abs(a.cx - b.cx) <= 0.16



def _reading_score(reading: _OCRLineReading) -> float:
    words = len(_word_tokens(reading.text))
    letters = len(re.findall(r"[A-Za-zÀ-ÿ]", reading.text))
    prominence = min(18.0, reading.h * 260.0) + min(10.0, reading.w * 25.0)
    return reading.confidence * 0.28 + words * 5.5 + min(letters, 120) * 0.08 + prominence



def _dedupe_frame_lines(lines: list[_OCRLineReading]) -> list[_OCRLineReading]:
    """Une duplicatas produzidas pelas duas variantes do mesmo quadro."""
    kept: list[_OCRLineReading] = []
    for line in sorted(lines, key=_reading_score, reverse=True):
        duplicate_index = None
        for index, existing in enumerate(kept):
            if _same_line_position(line, existing) and _texts_match(line.text, existing.text):
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(line)
        elif _reading_score(line) > _reading_score(kept[duplicate_index]):
            kept[duplicate_index] = line
    return kept



def _merge_caption_candidates(texts: list[str]) -> str:
    """Recupera prefixos/sufixos perdidos sem inventar palavras entre leituras."""
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

        max_overlap = min(len(base_tokens), len(cand_tokens), 14)
        for overlap in range(max_overlap, 1, -1):
            if cand_keys[-overlap:] == base_keys[:overlap]:
                base_tokens = cand_tokens[:-overlap] + base_tokens
                break
            if base_keys[-overlap:] == cand_keys[:overlap]:
                base_tokens = base_tokens + cand_tokens[overlap:]
                break

    return clean_review_text(" ".join(base_tokens))



def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0



def _cluster_temporal_lines(
    readings: list[_OCRLineReading],
    total_samples: int,
) -> list[_OCRPersistentLine]:
    """Mantém texto que se repete na MESMA posição em vários quadros.

    Esta é a mudança central para o formato do usuário: uma frase fixa do take 1
    sobrevive; texto de cenário, ruído de compressão e OCR aleatório tendem a não
    repetir com conteúdo + posição suficientes.
    """
    clusters: list[list[_OCRLineReading]] = []

    for reading in sorted(readings, key=lambda r: (r.sample_index, r.y, r.x)):
        best_index = None
        best_score = -1.0
        for index, cluster in enumerate(clusters):
            # Um cluster representa uma linha por quadro; evita juntar duas linhas
            # diferentes que o Tesseract encontrou na mesma amostra.
            if any(item.sample_index == reading.sample_index for item in cluster):
                continue
            representative = max(cluster, key=_reading_score)
            if not _same_line_position(reading, representative):
                continue
            if not _texts_match(reading.text, representative.text):
                continue
            score = _ocr_similarity(_normalized_ocr(reading.text), _normalized_ocr(representative.text))
            score += _horizontal_overlap(reading, representative) * 0.25
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            clusters.append([reading])
        else:
            clusters[best_index].append(reading)

    # Com 8 amostras, 3 confirmações são suficientes; com poucas amostras, 2.
    min_hits = 2 if total_samples <= 4 else 3
    persistent: list[_OCRPersistentLine] = []
    for cluster in clusters:
        unique_samples = sorted({item.sample_index for item in cluster})
        hits = len(unique_samples)
        if hits < min_hits:
            continue

        # Texto válido do primeiro take precisa aparecer cedo. Isso bloqueia texto
        # que só surge no take 2 quando a detecção da transição atrasar um pouco.
        first_sample = unique_samples[0]
        if total_samples >= 5 and first_sample > max(2, int(total_samples * 0.38)):
            continue

        text = _merge_caption_candidates([item.text for item in cluster])
        if not _ocr_line_quality(text):
            continue

        persistent.append(
            _OCRPersistentLine(
                text=text,
                confidence=_median([item.confidence for item in cluster]),
                hits=hits,
                total_samples=total_samples,
                first_sample=first_sample,
                x=_median([item.x for item in cluster]),
                y=_median([item.y for item in cluster]),
                w=_median([item.w for item in cluster]),
                h=_median([item.h for item in cluster]),
            )
        )
    return persistent



def _persistent_line_score(line: _OCRPersistentLine) -> float:
    words = len(_word_tokens(line.text))
    letters = len(re.findall(r"[A-Za-zÀ-ÿ]", line.text))
    centrality = max(0.0, 1.0 - abs(line.cx - 0.5) * 1.7)
    prominence = min(24.0, line.h * 300.0) + min(12.0, line.w * 28.0)
    score = (
        line.persistence * 72.0
        + line.confidence * 0.18
        + min(words, 12) * 5.0
        + min(letters, 100) * 0.07
        + prominence
        + centrality * 5.0
    )
    if words == 1 and letters < 8:
        score -= 18.0
    return score



def _lines_are_caption_neighbors(a: _OCRPersistentLine, b: _OCRPersistentLine) -> bool:
    if b.y < a.y:
        a, b = b, a
    vertical_gap = b.y - (a.y + a.h)
    max_gap = max(0.035, 2.3 * max(a.h, b.h))
    if vertical_gap > max_gap:
        return False
    a_left, a_right = a.x, a.x + a.w
    b_left, b_right = b.x, b.x + b.w
    overlap = max(0.0, min(a_right, b_right) - max(a_left, b_left))
    overlap_ratio = overlap / max(1e-6, min(a.w, b.w))
    return overlap_ratio >= 0.12 or abs(a.cx - b.cx) <= 0.18



def _build_caption_groups(lines: list[_OCRPersistentLine]) -> list[list[_OCRPersistentLine]]:
    groups: list[list[_OCRPersistentLine]] = []
    for line in sorted(lines, key=lambda item: (item.cy, item.x)):
        placed = False
        for group in groups:
            if any(_lines_are_caption_neighbors(existing, line) for existing in group):
                group.append(line)
                placed = True
                break
        if not placed:
            groups.append([line])

    # Faz uma segunda passagem para unir grupos que ficaram separados por ordem de inserção.
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            for j in range(i + 1, len(groups)):
                if any(_lines_are_caption_neighbors(a, b) for a in groups[i] for b in groups[j]):
                    groups[i].extend(groups[j])
                    del groups[j]
                    changed = True
                    break
    return groups



def _caption_group_score(group: list[_OCRPersistentLine]) -> float:
    ordered = sorted(group, key=lambda line: (line.cy, line.x))
    text = " ".join(line.text for line in ordered)
    words = len(_word_tokens(text))
    letters = len(re.findall(r"[A-Za-zÀ-ÿ]", text))
    avg_persistence = sum(line.persistence for line in ordered) / len(ordered)
    avg_confidence = sum(line.confidence for line in ordered) / len(ordered)
    median_height = _median([line.h for line in ordered])
    min_x = min(line.x for line in ordered)
    max_x = max(line.x + line.w for line in ordered)
    center_x = (min_x + max_x) / 2.0
    centrality = max(0.0, 1.0 - abs(center_x - 0.5) * 1.6)

    score = (
        avg_persistence * 95.0
        + avg_confidence * 0.15
        + min(words, 18) * 5.8
        + min(letters, 150) * 0.05
        + min(24.0, median_height * 320.0)
        + centrality * 5.0
    )
    if words <= 1:
        score -= 24.0
    elif words == 2:
        score -= 7.0
    # Um "parágrafo" enorme costuma ser OCR de cenário fundido, não overlay.
    if words > 30:
        score -= (words - 30) * 2.0
    return score



def _sample_timestamps(duration: float, max_samples: int) -> list[float]:
    """Amostra o miolo do take 1 e evita o frame de entrada/saída do corte."""
    if duration <= 0.08:
        return [max(0.01, duration * 0.5)]

    # Reserva ~8% no início e ~12% no fim. Texto com animação inicial deixa de
    # dominar; e um corte detectado alguns frames tarde não contamina o OCR.
    start = min(max(0.06, duration * 0.08), max(0.01, duration * 0.30))
    end = max(start + 0.02, duration * 0.88)
    end = min(end, max(start + 0.02, duration - 0.03))

    count_by_duration = max(4, int(duration / max(OCR_SAMPLE_STEP_SECONDS, 0.4)) + 2)
    count = max(4, min(max_samples, count_by_duration))
    if count == 1:
        return [(start + end) / 2.0]
    return [start + (end - start) * index / (count - 1) for index in range(count)]




def _refine_caption_region(
    video_path: str | Path,
    timestamps: list[float],
    chosen: list[_OCRPersistentLine],
    lang: str,
) -> list[str]:
    """Faz uma segunda leitura somente na caixa da legenda escolhida.

    O OCR global localiza a região. Esta etapa usa um recorte apertado e PSM 6
    para recuperar letras de borda que o modo esparso pode perder (ex.: "vem"
    virar "em"). Como a região já foi validada temporalmente, o risco de puxar
    texto aleatório do cenário fica muito menor.
    """
    import pytesseract

    if not chosen:
        return []
    min_x = min(line.x for line in chosen)
    min_y = min(line.y for line in chosen)
    max_x = max(line.x + line.w for line in chosen)
    max_y = max(line.y + line.h for line in chosen)

    # Padding generoso nas laterais para não cortar primeira/última letra.
    x0 = max(0.0, min_x - 0.055)
    x1 = min(1.0, max_x + 0.055)
    y0 = max(0.0, min_y - 0.030)
    y1 = min(1.0, max_y + 0.030)

    refined: list[str] = []
    # 3 quadros bastam: início/meio/fim do miolo validado.
    if len(timestamps) <= 3:
        selected_times = timestamps
    else:
        selected_times = [timestamps[0], timestamps[len(timestamps)//2], timestamps[-1]]

    for timestamp in selected_times:
        frame = _ocr_grab_frame(video_path, timestamp)
        if frame is None:
            continue
        for variant in _ocr_variants(frame):
            h, w = variant.shape[:2]
            left = max(0, min(w - 1, int(round(x0 * w))))
            right = max(left + 2, min(w, int(round(x1 * w))))
            top = max(0, min(h - 1, int(round(y0 * h))))
            bottom = max(top + 2, min(h, int(round(y1 * h))))
            roi = variant[top:bottom, left:right]
            if roi.size == 0:
                continue
            try:
                raw = pytesseract.image_to_string(
                    roi,
                    lang=lang,
                    config="--oem 1 --psm 6 -c preserve_interword_spaces=1",
                )
            except Exception:
                continue
            text = clean_review_text(re.sub(r"\s+", " ", raw or "")).strip()
            if _ocr_line_quality(text):
                refined.append(text)
    return refined


_COMMON_PT_WORDS = {
    "a", "ao", "aos", "as", "com", "da", "das", "de", "do", "dos",
    "e", "ela", "ele", "em", "eu", "isso", "me", "meu", "minha",
    "na", "nas", "no", "nos", "não", "o", "os", "ou", "para", "por",
    "pra", "que", "se", "sem", "só", "te", "tem", "tu", "um", "uma",
    "vai", "vc", "você", "vocês", "tudo", "quando", "como", "mas", "mais",
}


def _refined_text_score(text: str) -> float:
    """Pontua naturalidade de uma frase, sem premiar OCR longo e cheio de lixo.

    A versão V2 dava +8 por qualquer token. Assim, uma leitura como
    ``di engolir tudo ob vate / 7 As ...`` podia vencer uma frase correta
    simplesmente por conter mais pedaços. Aqui o sinal principal passa a ser
    proporção de letras + palavras plausíveis; símbolos soltos e tokens sem
    letras recebem penalidade forte.
    """
    value = clean_review_text(re.sub(r"\s+", " ", text or "")).strip()
    if not value:
        return -999.0

    tokens = _word_tokens(value)
    letters = len(re.findall(r"[A-Za-zÀ-ÿ]", value))
    visible = len(re.sub(r"\s+", "", value))
    letter_ratio = letters / max(visible, 1)

    weird_chars = len(re.findall(r'[^A-Za-zÀ-ÿ0-9\s,.;:!?@#\'"-]', value))
    symbol_only = 0
    suspicious_short = 0
    common_hits = 0
    numeric_only = 0

    for token in tokens:
        key = _token_key(token)
        if not re.search(r"[A-Za-zÀ-ÿ]", token):
            symbol_only += 1
            if re.search(r"\d", token):
                numeric_only += 1
            continue
        if key in _COMMON_PT_WORDS:
            common_hits += 1
        if len(key) <= 2 and key not in _COMMON_PT_WORDS:
            suspicious_short += 1

    # Comprimento conta pouco; qualidade lexical conta muito.
    score = (
        min(len(tokens), 24) * 2.0
        + min(letters, 180) * 0.10
        + letter_ratio * 42.0
        + min(common_hits, 10) * 4.0
        - weird_chars * 9.0
        - symbol_only * 12.0
        - numeric_only * 4.0
        - suspicious_short * 3.5
    )
    return score


def _choose_refined_candidate(candidates: list[str]) -> str:
    """Escolhe a leitura que mais concorda com as outras, não a mais longa.

    O texto do take 1 é fixo, portanto leituras corretas tendem a reaparecer em
    vários frames/variantes. Ruído pode ser comprido, mas costuma ser isolado.
    """
    clean: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        item = clean_review_text(re.sub(r"\s+", " ", item or "")).strip()
        norm = _normalized_ocr(item)
        if not item or not norm or norm in seen:
            continue
        seen.add(norm)
        clean.append(item)

    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]

    def total_score(candidate: str) -> float:
        norm = _normalized_ocr(candidate)
        peers = []
        for other in clean:
            if other == candidate:
                continue
            onorm = _normalized_ocr(other)
            similarity = _ocr_similarity(norm, onorm)
            overlap = _token_overlap(norm, onorm)
            peers.append(max(similarity, overlap))
        peers.sort(reverse=True)
        # As três concordâncias mais fortes bastam e evitam que dezenas de
        # variantes ruins dominem por quantidade.
        consensus = sum(peers[:3]) * 12.0
        return _refined_text_score(candidate) + consensus

    return max(clean, key=total_score)



def _longest_common_token_block(a_tokens: list[str], b_tokens: list[str]) -> tuple[int, int, int]:
    a = [_token_key(t) for t in a_tokens]
    b = [_token_key(t) for t in b_tokens]
    best = (0, 0, 0)
    for i in range(len(a)):
        if not a[i]:
            continue
        for j in range(len(b)):
            if a[i] != b[j] or not b[j]:
                continue
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k] and a[i + k]:
                k += 1
            if k > best[2]:
                best = (i, j, k)
    return best


def _repair_caption_edges(base: str, candidates: list[str]) -> str:
    """Corrige apenas prefixo/sufixo usando uma leitura que concorda no miolo.

    Isso resolve o caso clássico do Tesseract em que as aspas de abertura +
    primeira palavra viram lixo (ex.: ``al en engolir...``), sem concatenar
    sobras incompatíveis no restante da frase.
    """
    base_tokens = _word_tokens(base)
    if len(base_tokens) < 3:
        return base

    result = list(base_tokens)
    for candidate in candidates:
        cand_tokens = _word_tokens(candidate)
        if len(cand_tokens) < 3:
            continue
        i, j, k = _longest_common_token_block(result, cand_tokens)
        if k < 3:
            continue

        base_prefix = result[:i]
        cand_prefix = cand_tokens[:j]
        if 0 < len(cand_prefix) <= 3 and len(base_prefix) <= 3:
            bp = " ".join(base_prefix)
            cp = " ".join(cand_prefix)
            if _refined_text_score(cp) > _refined_text_score(bp) + 7.0:
                result = cand_prefix + result[i:]
                i = len(cand_prefix)

        # Recalcula o bloco após possível troca de prefixo para tratar a borda final.
        i2, j2, k2 = _longest_common_token_block(result, cand_tokens)
        if k2 < 3:
            continue
        base_suffix = result[i2 + k2:]
        cand_suffix = cand_tokens[j2 + k2:]
        if 0 < len(cand_suffix) <= 3 and len(base_suffix) <= 3:
            bs = " ".join(base_suffix)
            cs = " ".join(cand_suffix)
            if _refined_text_score(cs) > _refined_text_score(bs) + 7.0:
                result = result[:i2 + k2] + cand_suffix

    return clean_review_text(" ".join(result)).strip()

def read_burned_caption(
    video_path: str | Path,
    duration: float,
    *,
    language: str = "",
) -> tuple[list[CaptionEvent], str]:
    """Extrai UMA frase fixa do primeiro take por consenso temporal + espacial.

    Regra de produto desta versão:
    - o vídeo possui dois takes;
    - o texto desejado está fixo no take 1;
    - texto só é aceito se reaparecer na mesma região em vários quadros;
    - na dúvida, retorna vazio em vez de inventar uma legenda aleatória.
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

    preferred = _OCR_LANGUAGE_MAP.get((language or "").strip().lower())
    # Se o usuário escolheu Português/Inglês/Espanhol, usa SOMENTE esse idioma.
    # Misturar ``por+eng`` piora palavras portuguesas curtas e aumenta falsos
    # positivos em fundo/roupa. Combinação só é usada no modo automático.
    if preferred and preferred in available:
        lang = preferred
    else:
        fallback = [code for code in ["por", "eng"] if code in available]
        lang = "+".join(fallback) or "eng"

    max_samples = max(4, OCR_MAX_SAMPLES)
    timestamps = _sample_timestamps(float(duration), max_samples)
    all_readings: list[_OCRLineReading] = []
    successful_samples = 0

    for sample_index, timestamp in enumerate(timestamps):
        frame = _ocr_grab_frame(video_path, timestamp)
        if frame is None:
            continue
        successful_samples += 1
        frame_lines: list[_OCRLineReading] = []
        try:
            for variant in _ocr_variants(frame):
                frame_lines.extend(_ocr_lines(variant, lang, sample_index))
        except Exception as exc:
            raise MediaError(f"Falha ao ler o texto fixo do primeiro take: {exc}") from exc
        all_readings.extend(_dedupe_frame_lines(frame_lines))

    if successful_samples < 2 or not all_readings:
        return [], ""

    persistent = _cluster_temporal_lines(all_readings, len(timestamps))
    if not persistent:
        # Falhar fechado é intencional: melhor pedir revisão manual do que puxar
        # uma placa, camiseta ou palavra aleatória do cenário.
        return [], "Nenhum texto fixo se repetiu com confiança suficiente no primeiro take."

    groups = _build_caption_groups(persistent)
    if not groups:
        return [], ""

    chosen = max(groups, key=_caption_group_score)
    chosen = sorted(chosen, key=lambda line: (line.cy, line.x))

    # Evita arrastar logo/arroba muito pequeno que esteja perto da legenda, sem
    # apagar prefixos legítimos como "eu:", "POV:" ou "ela:". Linhas curtas só
    # são descartadas quando também são visualmente bem menores que as demais.
    median_height = _median([line.h for line in chosen])
    strongest_score = max((_persistent_line_score(line) for line in chosen), default=0.0)
    filtered = []
    for line in chosen:
        words = len(_word_tokens(line.text))
        letters = len(re.findall(r"[A-Za-zÀ-ÿ]", line.text))
        line_score = _persistent_line_score(line)

        looks_tiny_auxiliary = (
            len(chosen) > 1
            and line.h < median_height * 0.62
            and words <= 2
            and letters <= 14
        )
        # Em fundo estático o Tesseract pode repetir a MESMA alucinação em todos
        # os frames. Portanto persistência sozinha não basta. Se existe uma ou
        # mais linhas grandes/fortes, descartamos fragmentos curtos que ficaram
        # grudados ao grupo apenas por proximidade espacial.
        weak_fragment = (
            len(chosen) >= 3
            and line_score < strongest_score * 0.74
            and (words <= 2 or line.w < 0.18)
        )
        if not looks_tiny_auxiliary and not weak_fragment:
            filtered.append(line)
    if filtered:
        chosen = filtered

    merged = clean_review_text(" ".join(line.text.strip() for line in chosen if line.text.strip()))
    merged = re.sub(r"\s+", " ", merged).strip()

    # Segunda passada na região já validada: recupera letras cortadas nas bordas
    # sem voltar a varrer o cenário inteiro.
    refined_candidates = _refine_caption_region(video_path, timestamps, chosen, lang)
    if refined_candidates:
        # Não concatena leituras diferentes. O merge textual da V2 podia criar
        # frases Frankenstein juntando sobras incompatíveis de vários frames.
        candidates = [merged] + refined_candidates
        best_candidate = _choose_refined_candidate(candidates)
        if best_candidate and _refined_text_score(best_candidate) >= _refined_text_score(merged) - 6.0:
            merged = best_candidate
        merged = _repair_caption_edges(merged, refined_candidates)

    merged = clean_review_text(re.sub(r"\s+", " ", merged)).strip()
    # Tesseract costuma enxergar a aspa de fechamento e perder a de abertura
    # quando ela encosta na primeira letra. Balanceia apenas aspas duplas, sem
    # inventar pontuação quando nenhuma delas foi reconhecida.
    if merged.endswith('"') and not merged.startswith('"'):
        merged = '"' + merged
    elif merged.startswith('"') and not merged.endswith('"'):
        merged = merged + '"'
    if not merged:
        return [], ""

    hits = max(line.hits for line in chosen)
    avg_conf = sum(line.confidence for line in chosen) / len(chosen)
    event = CaptionEvent(start=0.0, end=max(0.15, duration), text=merged)
    summary = (
        f"Texto fixo detectado em {hits}/{len(timestamps)} amostras "
        f"(confiança OCR média {avg_conf:.0f}%): {merged}\n"
        "O texto foi escolhido por repetição + posição na tela, não por uma leitura isolada."
    )
    return [event], summary

