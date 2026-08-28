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
    """Recorta áreas de UI e cria duas versões leves para o Tesseract."""
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
    # OCR ganha muito pouco acima de ~900 px de largura e custa bem mais CPU.
    target_width = 900
    if width < target_width:
        scale = target_width / max(width, 1)
        frame = cv2.resize(frame, (target_width, int(round(height * scale))), interpolation=cv2.INTER_CUBIC)
    elif width > 1200:
        scale = 1200.0 / width
        frame = cv2.resize(frame, (1200, int(round(height * scale))), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Texto claro, muito comum em Reels/TikTok.
    white = (gray >= 195).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    isolated = 255 - white
    return [gray, isolated]


def _ocr_frame(image, lang: str) -> tuple[str, float]:
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(image, lang=lang, config="--psm 6", output_type=Output.DICT)
    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []
    for index in range(len(data["text"])):
        token = (data["text"][index] or "").strip()
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if not token or confidence < 58:
            continue
        if not re.search(r"[\wÀ-ÿ]", token):
            continue
        key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
        lines.setdefault(key, []).append(token)
        confidences.append(confidence)

    text = clean_review_text("\n".join(" ".join(words) for _key, words in sorted(lines.items())).strip())
    average = sum(confidences) / len(confidences) if confidences else 0.0
    return text, average


def _ocr_candidate_score(text: str, confidence: float) -> float:
    """Pontua OCR favorecendo a frase completa, não só a confiança do Tesseract."""
    flat = clean_review_text(re.sub(r"\s*\n\s*", " ", text or ""))
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", flat)
    letters = len(re.findall(r"[A-Za-zÀ-ÿ]", flat))
    # Uma palavra extra vale mais do que alguns pontos de confiança. Isso evita
    # escolher "aqui hoje..." (95%) no lugar de "vem aqui hoje..." (86%).
    return float(confidence) * 0.35 + len(words) * 6.0 + min(letters, 180) * 0.08


def _ocr_frame_best(frame, lang: str) -> tuple[str, float]:
    """Escolhe a leitura mais completa entre as variantes do mesmo quadro."""
    best_text = ""
    best_conf = 0.0
    best_score = float("-inf")
    for variant in _ocr_variants(frame):
        text, confidence = _ocr_frame(variant, lang)
        if not text:
            continue
        text = clean_review_text(re.sub(r"\s*\n\s*", " ", text))
        score = _ocr_candidate_score(text, confidence)
        if score > best_score:
            best_text, best_conf, best_score = text, confidence, score
    return best_text, best_conf


def _word_tokens(value: str) -> list[str]:
    return re.findall(r"\S+", clean_review_text(re.sub(r"\s+", " ", value or "")))


def _token_key(token: str) -> str:
    return re.sub(r"[^a-z0-9à-ÿ]", "", token.lower())


def _merge_caption_candidates(texts: list[str]) -> str:
    """Une somente sobreposições seguras para recuperar início/fim perdido pelo OCR.

    Ex.: "vem aqui hoje tomar um vinho cmg" +
         "aqui hoje tomar um vinho cmg, não vai rolar nada"
    vira uma única frase completa.
    """
    clean = [clean_review_text(re.sub(r"\s+", " ", t or "")).strip() for t in texts]
    clean = [t for t in clean if t]
    if not clean:
        return ""

    # Começa pela leitura mais completa.
    base = max(clean, key=lambda t: (len(_word_tokens(t)), len(t)))
    base_tokens = _word_tokens(base)

    for candidate in sorted(clean, key=lambda t: (len(_word_tokens(t)), len(t)), reverse=True):
        cand_tokens = _word_tokens(candidate)
        if not cand_tokens or candidate == base:
            continue

        base_keys = [_token_key(t) for t in base_tokens]
        cand_keys = [_token_key(t) for t in cand_tokens]

        # Se uma leitura já está contida na outra, não adiciona nada.
        base_norm = " ".join(base_keys)
        cand_norm = " ".join(cand_keys)
        if cand_norm and cand_norm in base_norm:
            continue
        if base_norm and base_norm in cand_norm:
            base_tokens = cand_tokens
            continue

        max_overlap = min(len(base_tokens), len(cand_tokens), 12)
        merged = False
        for overlap in range(max_overlap, 1, -1):
            # candidato termina onde a base começa -> recupera prefixo
            if cand_keys[-overlap:] == base_keys[:overlap]:
                base_tokens = cand_tokens[:-overlap] + base_tokens
                merged = True
                break
            # base termina onde o candidato começa -> recupera sufixo
            if base_keys[-overlap:] == cand_keys[:overlap]:
                base_tokens = base_tokens + cand_tokens[overlap:]
                merged = True
                break
        if merged:
            continue

    return clean_review_text(" ".join(base_tokens))


def _normalized_ocr(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _ocr_similarity(a: str, b: str) -> float:
    """Similaridade 0-1 entre duas leituras, tolerando o ruído normal do OCR."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# Duas leituras do mesmo "cartão" de legenda quase nunca saem idênticas do
# OCR (ruído de compressão, motion blur etc.). Este limiar decide quando duas
# leituras em sequência ainda contam como "o mesmo texto" versus uma troca
# real de frase. Ajuste para cima se cartões diferentes estiverem sendo
# fundidos em um só; para baixo se um único texto estático estiver sendo
# picotado em vários pedaços.
_OCR_SIMILARITY_THRESHOLD = 0.72


def read_burned_caption(
    video_path: str | Path,
    duration: float,
    *,
    language: str = "",
) -> tuple[list[CaptionEvent], str]:
    """Lê uma legenda fixa do primeiro take e a mantém durante todo o take.

    A V3 não cria cartões temporizados. Ela lê vários quadros, agrupa leituras
    semelhantes, recupera prefixos/sufixos perdidos e devolve UMA única
    ``CaptionEvent`` de 0s até ``duration``.
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
            "O programa Tesseract OCR não foi encontrado. No Docker/Railway ele é instalado "
            "automaticamente; no Windows instale em https://github.com/UB-Mannheim/tesseract/wiki."
        ) from exc

    preferred = _OCR_LANGUAGE_MAP.get((language or "").strip().lower())
    langs = [code for code in [preferred, "por", "eng"] if code and code in available]
    lang = "+".join(dict.fromkeys(langs)) or "eng"

    # Amostra cedo de propósito: palavras no começo da frase podem ficar menos
    # nítidas em quadros posteriores por animação, movimento ou compressão.
    max_samples = max(3, OCR_MAX_SAMPLES)
    uniform_count = max(2, min(max_samples - 2, int(duration / OCR_SAMPLE_STEP_SECONDS) + 1))
    early = [min(max(duration * 0.02, 0.03), max(duration - 0.02, 0.03))]
    if duration > 0.35:
        early.append(min(0.28, duration * 0.10))
    uniform = [duration * (index + 0.5) / uniform_count for index in range(uniform_count)]
    timestamps: list[float] = []
    for value in early + uniform:
        value = max(0.01, min(float(value), max(0.01, duration - 0.01)))
        if all(abs(value - existing) > 0.05 for existing in timestamps):
            timestamps.append(value)
        if len(timestamps) >= max_samples:
            break

    readings: list[tuple[str, float]] = []
    for timestamp in timestamps:
        frame = _ocr_grab_frame(video_path, timestamp)
        if frame is None:
            continue
        try:
            text, confidence = _ocr_frame_best(frame, lang)
        except Exception as exc:
            raise MediaError(f"Falha ao ler a legenda do vídeo: {exc}") from exc
        text = clean_review_text(re.sub(r"\s+", " ", text or ""))
        if text:
            readings.append((text, confidence))

    if not readings:
        return [], ""

    # Agrupamento GLOBAL (não cronológico): como a legenda é fixa, leituras
    # "vem aqui hoje..." e "aqui hoje..." pertencem à mesma frase.
    clusters: list[dict] = []
    for text, confidence in readings:
        norm = _normalized_ocr(text)
        best_cluster = None
        best_similarity = 0.0
        for cluster in clusters:
            similarity = max(_ocr_similarity(norm, item["norm"]) for item in cluster["items"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster
        if best_cluster is not None and best_similarity >= 0.60:
            best_cluster["items"].append({"text": text, "confidence": confidence, "norm": norm})
        else:
            clusters.append({"items": [{"text": text, "confidence": confidence, "norm": norm}]})

    # Prefere a frase vista em mais quadros. Em empate, usa completude + confiança.
    def cluster_rank(cluster: dict) -> tuple[float, float]:
        items = cluster["items"]
        best = max(_ocr_candidate_score(i["text"], i["confidence"]) for i in items)
        return (float(len(items)), best)

    chosen = max(clusters, key=cluster_rank)
    items = chosen["items"]
    candidate_texts = [item["text"] for item in items]
    merged = _merge_caption_candidates(candidate_texts)

    # Se a fusão não acrescentou nada, ainda escolhemos pela pontuação de
    # completude, e não pela confiança isolada do Tesseract.
    best_item = max(items, key=lambda i: _ocr_candidate_score(i["text"], i["confidence"]))
    if len(_word_tokens(best_item["text"])) > len(_word_tokens(merged)):
        merged = best_item["text"]

    merged = clean_review_text(re.sub(r"\s+", " ", merged)).strip()
    if not merged:
        return [], ""

    avg_conf = sum(float(i["confidence"]) for i in items) / max(len(items), 1)
    event = CaptionEvent(start=0.0, end=max(0.15, duration), text=merged)
    summary = (
        f"Legenda fixa detectada (confiança aproximada {avg_conf:.0f}%): {merged}\n"
        "Ela ficará visível durante todo o primeiro take. Revise o texto antes de gerar."
    )
    return [event], summary
