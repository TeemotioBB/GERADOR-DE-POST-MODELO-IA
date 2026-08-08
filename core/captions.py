from __future__ import annotations

import functools
import re
from difflib import SequenceMatcher
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import regex
from PIL import Image, ImageDraw, ImageFont

from .config import CAPTION_FONT, DEFAULT_LANGUAGE, WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL
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
    wav_path = work / "intro_audio.wav"
    extract_audio_for_transcription(video_path, wav_path, duration)

    model = _get_whisper_model()
    try:
        segments, info = model.transcribe(
            str(wav_path),
            language=language or None,
            beam_size=3,
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

    clipped = [
        CaptionEvent(event.start, min(event.end, duration), event.text)
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
    requested_size = max(24, int(round(height * max(2.0, min(font_percent, 9.0)) / 100.0)))
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


def _ocr_regions(frame):
    """Recorta áreas prováveis de legenda e evita bordas com logos/@/UI."""
    height, width = frame.shape[:2]
    x0 = int(width * 0.035)
    x1 = int(width * 0.965)
    regions = [
        ("central", frame[int(height * 0.08):int(height * 0.92), x0:x1], 1.00),
        ("superior", frame[int(height * 0.06):int(height * 0.55), x0:x1], 0.96),
        ("meio", frame[int(height * 0.22):int(height * 0.78), x0:x1], 1.04),
        ("inferior", frame[int(height * 0.45):int(height * 0.94), x0:x1], 0.96),
    ]
    return [(name, crop, weight) for name, crop, weight in regions if crop.size]


def _ocr_variants(frame):
    """Gera versões robustas para texto branco/escuro com contorno em Reels."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    target_width = 1280
    if width < target_width:
        scale = target_width / max(width, 1)
        frame = cv2.resize(
            frame,
            (target_width, max(2, int(round(height * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    white_mask = ((value >= 175) & (saturation <= 125)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    isolated_white = 255 - white_mask

    adaptive = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return [
        ("clahe", clahe),
        ("branco", isolated_white),
        ("adaptativo", adaptive),
    ]


def _clean_ocr_text(text: str) -> str:
    """Remove quebras artificiais e pequenos artefatos sem destruir pontuação."""
    text = (text or "").replace("|", " ").replace("¦", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip(" _-—~")
        if not line:
            continue
        alnum = len(re.findall(r"[\wÀ-ÿ]", line, flags=re.UNICODE))
        printable = len(re.sub(r"\s", "", line))
        if printable and alnum / printable < 0.42:
            continue
        if re.fullmatch(r"(?:[._~\-—]+|\d{1,2})", line):
            continue
        lines.append(line)

    # A posição/line wrap é refeita pelo renderizador; manter quebras do Tesseract
    # costuma gerar linhas erradas. Junta tudo e corrige espaços de pontuação.
    joined = " ".join(lines)
    joined = re.sub(r"\s+([,.;:!?])", r"\1", joined)
    joined = re.sub(r"([¿¡])\s+", r"\1", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def _ocr_frame(image, lang: str, *, psm: int = 6) -> tuple[str, float]:
    import pytesseract
    from pytesseract import Output

    config = f"--oem 1 --psm {int(psm)} -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=Output.DICT)
    words: list[str] = []
    confidences: list[float] = []

    for index in range(len(data["text"])):
        token = (data["text"][index] or "").strip()
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if not token or confidence < 34:
            continue

        # Mantém pontuação útil; rejeita ruído que não contém nenhum caractere legível.
        if not re.search(r"[\wÀ-ÿ0-9.,;:!?%$€£@#'\"()\-+]", token, flags=re.UNICODE):
            continue
        words.append(token)
        confidences.append(confidence)

    text = _clean_ocr_text(" ".join(words))
    average = sum(confidences) / len(confidences) if confidences else 0.0
    return text, average


def _text_quality(text: str) -> float:
    if not text:
        return 0.0
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 0.0
    alnum = len(re.findall(r"[\wÀ-ÿ]", compact, flags=re.UNICODE))
    ratio = alnum / len(compact)
    length_score = min(len(text) / 55.0, 1.0)
    weird_runs = len(re.findall(r"[^\wÀ-ÿ\s.,;:!?%$€£@#'\"()\-+]{2,}", text, flags=re.UNICODE))
    return max(0.0, min(1.0, 0.62 * ratio + 0.38 * length_score - weird_runs * 0.08))


def _select_ocr_result(results: list[dict]) -> tuple[str, float, float]:
    """Escolhe por consenso fuzzy entre frames, não por igualdade exata."""
    if not results:
        return "", 0.0, 0.0

    normalized = [re.sub(r"\s+", " ", item["text"].lower()).strip() for item in results]
    unique_frames = sorted({item["frame"] for item in results})

    best_index = 0
    best_score = -1.0
    best_consensus = 0.0

    for i, item in enumerate(results):
        per_frame: dict[int, float] = {}
        for j, other in enumerate(results):
            if i == j:
                sim = 1.0
            else:
                sim = SequenceMatcher(None, normalized[i], normalized[j]).ratio()
            frame_id = other["frame"]
            per_frame[frame_id] = max(per_frame.get(frame_id, 0.0), sim)

        consensus = sum(per_frame.get(frame_id, 0.0) for frame_id in unique_frames) / max(len(unique_frames), 1)
        confidence = min(max(item["confidence"] / 100.0, 0.0), 1.0)
        quality = _text_quality(item["text"])
        region_weight = item.get("region_weight", 1.0)
        score = (0.47 * consensus + 0.30 * confidence + 0.23 * quality) * region_weight

        if score > best_score:
            best_index = i
            best_score = score
            best_consensus = consensus

    best = results[best_index]
    # Confiança apresentada é composta: OCR + repetição temporal + qualidade textual.
    presented_confidence = 100.0 * min(
        0.45 * (best["confidence"] / 100.0) + 0.40 * best_consensus + 0.15 * _text_quality(best["text"]),
        1.0,
    )
    return best["text"], presented_confidence, best_consensus


def read_burned_caption(
    video_path: str | Path,
    duration: float,
    *,
    language: str = "",
) -> tuple[str, float]:
    """Lê texto queimado com poucos frames e consenso fuzzy.

    Mantém o custo de CPU controlado no Railway: a primeira passada usa apenas
    6 chamadas de OCR (3 frames × 2 variantes). Uma segunda passada curta só
    acontece quando a primeira leitura fica fraca.

    OCR tradicional não identifica emojis de forma confiável; por isso o app
    carrega o texto detectado em um campo editável antes da geração.
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
    langs = [code for code in [preferred, "por", "eng"] if code and code in available]
    lang = "+".join(dict.fromkeys(langs)) or "eng"

    duration = max(float(duration), 0.05)
    frames = []
    for frame_index, fraction in enumerate((0.24, 0.50, 0.76)):
        frame = _ocr_grab_frame(video_path, duration * fraction)
        if frame is not None:
            frames.append((frame_index, frame))

    results: list[dict] = []

    def run_pass(*, fallback: bool = False) -> None:
        for frame_index, frame in frames:
            central = _ocr_regions(frame)[0][1]
            variants = _ocr_variants(central)
            chosen = variants[1:] if fallback else variants[:2]
            psm = 11 if fallback else 6
            for _variant_name, variant in chosen:
                try:
                    text, confidence = _ocr_frame(variant, lang, psm=psm)
                except Exception as exc:
                    raise MediaError(f"Falha ao ler a legenda do vídeo: {exc}") from exc
                if len(re.findall(r"[\wÀ-ÿ]", text, flags=re.UNICODE)) < 3:
                    continue
                results.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "frame": frame_index,
                        "region_weight": 1.0,
                    }
                )

    run_pass(fallback=False)
    text, confidence, consensus = _select_ocr_result(results)

    # Só gasta CPU extra quando a primeira leitura realmente parece ruim.
    if not text or confidence < 58.0 or consensus < 0.58:
        run_pass(fallback=True)
        text, confidence, _consensus = _select_ocr_result(results)

    return text, confidence
