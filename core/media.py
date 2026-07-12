from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps, UnidentifiedImageError


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    codec_name: str | None = None
    rotation: int = 0


@dataclass(frozen=True)
class PreparedIntroMedia:
    kind: Literal["image", "video"]
    input_path: str
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    duration: float | None = None


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(
            f"O programa '{name}' não foi encontrado. Instale o FFmpeg e deixe "
            f"'{name}' disponível no PATH."
        )
    return path


def run_command(command: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        if len(details) > 5000:
            details = details[-5000:]
        raise MediaError(f"Falha ao executar FFmpeg/FFprobe:\n{details}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError("O processamento excedeu o tempo limite.") from exc


def _parse_rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 30.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        den = float(denominator)
        return float(numerator) / den if den else 30.0
    return float(value)


def _stream_rotation(stream: dict[str, Any]) -> int:
    rotation: float = 0.0
    tags = stream.get("tags") or {}
    if tags.get("rotate") not in {None, ""}:
        try:
            rotation = float(tags["rotate"])
        except (TypeError, ValueError):
            rotation = 0.0

    for side_data in stream.get("side_data_list") or []:
        if side_data.get("rotation") not in {None, ""}:
            try:
                rotation = float(side_data["rotation"])
            except (TypeError, ValueError):
                pass
            break

    normalized = int(round(rotation)) % 360
    return normalized


def probe_video(path: str | Path) -> VideoInfo:
    ffprobe = require_binary("ffprobe")
    result = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        timeout=60,
    )
    data: dict[str, Any] = json.loads(result.stdout)
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video_stream:
        raise MediaError("O arquivo enviado não possui uma faixa de vídeo válida.")

    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration_raw = video_stream.get("duration") or data.get("format", {}).get("duration")
    if duration_raw in {None, "N/A"}:
        raise MediaError("Não foi possível identificar a duração do vídeo.")

    fps = _parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    fps = min(max(fps, 12.0), 60.0)

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    rotation = _stream_rotation(video_stream)
    if rotation in {90, 270}:
        width, height = height, width

    return VideoInfo(
        duration=float(duration_raw),
        width=width,
        height=height,
        fps=fps,
        has_audio=audio_stream is not None,
        codec_name=video_stream.get("codec_name"),
        rotation=rotation,
    )


def scaled_even_dimensions(width: int, height: int, max_long_edge: int = 0) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise MediaError("A mídia possui dimensões inválidas.")

    target_w, target_h = width, height
    longest = max(width, height)
    if max_long_edge > 0 and longest > max_long_edge:
        ratio = max_long_edge / longest
        target_w = max(2, int(round(width * ratio)))
        target_h = max(2, int(round(height * ratio)))

    output_w = target_w if target_w % 2 == 0 else target_w + 1
    output_h = target_h if target_h % 2 == 0 else target_h + 1
    return output_w, output_h


def prepare_photo(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_long_edge: int = 0,
) -> tuple[int, int, int, int]:
    """Corrige EXIF, converte para RGB, limita resolução e garante dimensões pares."""

    source = Path(input_path)
    target = Path(output_path)
    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            original_w, original_h = image.size
            output_w, output_h = scaled_even_dimensions(original_w, original_h, max_long_edge)

            target_w, target_h = original_w, original_h
            if (output_w, output_h) != (original_w, original_h):
                ratio = min(output_w / original_w, output_h / original_h)
                target_w = max(2, int(round(original_w * ratio)))
                target_h = max(2, int(round(original_h * ratio)))
                target_w = min(target_w, output_w)
                target_h = min(target_h, output_h)
                image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)

            if (output_w, output_h) != image.size:
                canvas = Image.new("RGB", (output_w, output_h))
                x = (output_w - image.width) // 2
                y = (output_h - image.height) // 2
                canvas.paste(image, (x, y))
                image = canvas

            image.save(target, format="PNG", optimize=True)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MediaError("Não foi possível abrir a imagem enviada.") from exc

    return original_w, original_h, output_w, output_h


def prepare_intro_media(
    input_path: str | Path,
    work_dir: str | Path,
    *,
    max_long_edge: int = 0,
) -> PreparedIntroMedia:
    """Reconhece automaticamente se a mídia inicial é imagem ou vídeo."""

    source = Path(input_path)
    if not source.exists():
        raise MediaError("A mídia inicial enviada não foi encontrada.")

    # Imagens são tentadas primeiro. Isso evita tratar JPEG/PNG como stream de vídeo.
    try:
        with Image.open(source) as image:
            image.verify()
        prepared_path = Path(work_dir) / "intro_image.png"
        original_w, original_h, output_w, output_h = prepare_photo(
            source,
            prepared_path,
            max_long_edge=max_long_edge,
        )
        return PreparedIntroMedia(
            kind="image",
            input_path=str(prepared_path),
            original_width=original_w,
            original_height=original_h,
            output_width=output_w,
            output_height=output_h,
        )
    except (UnidentifiedImageError, OSError, ValueError):
        pass

    try:
        info = probe_video(source)
    except MediaError as exc:
        raise MediaError("A primeira mídia precisa ser uma imagem ou um vídeo válido.") from exc

    output_w, output_h = scaled_even_dimensions(info.width, info.height, max_long_edge)
    return PreparedIntroMedia(
        kind="video",
        input_path=str(source),
        original_width=info.width,
        original_height=info.height,
        output_width=output_w,
        output_height=output_h,
        duration=info.duration,
    )


def escape_filter_path(path: str | Path) -> str:
    """Escapa caminho para uso dentro de filtros do FFmpeg."""
    value = str(Path(path).resolve()).replace("\\", "/")
    value = value.replace(":", r"\:").replace("'", r"\'")
    return value
