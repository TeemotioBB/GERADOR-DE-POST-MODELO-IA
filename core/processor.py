from __future__ import annotations

import math
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .captions import (
    CaptionEvent,
    CaptionOverlay,
    manual_caption,
    read_burned_caption,
    render_caption_overlays,
    transcribe_intro,
)
from .config import (
    BACKGROUND_BLUR_DIVISOR,
    FFMPEG_THREADS,
    MAX_VIDEO_MINUTES,
    OUTPUT_AUDIO_BITRATE,
    OUTPUT_CRF,
    OUTPUT_MAX_LONG_EDGE,
    OUTPUT_PRESET,
    TEMP_MAX_AGE_HOURS,
    WORK_ROOT,
)
from .media import MediaError, PreparedIntroMedia, prepare_intro_media, probe_video, require_binary, run_command
from .transition import TransitionResult, detect_intro_end

ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class ProcessResult:
    output_path: str
    report: str
    transition_seconds: float


def cleanup_old_jobs() -> None:
    cutoff = time.time() - TEMP_MAX_AGE_HOURS * 3600
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    for item in WORK_ROOT.iterdir():
        try:
            if item.is_dir() and item.stat().st_mtime < cutoff:
                shutil.rmtree(item, ignore_errors=True)
        except OSError:
            continue


def _safe_progress(progress: ProgressFn | None, value: float, description: str) -> None:
    if progress:
        progress(max(0.0, min(1.0, value)), description)


def _fit_filter(label: str, width: int, height: int, fps: float, mode: str) -> tuple[str, str]:
    fps_text = f"{fps:.6f}".rstrip("0").rstrip(".")
    common_tail = f"setsar=1,settb=AVTB,fps={fps_text},format=yuv420p"

    if mode == "Preencher a tela (pode cortar bordas)":
        chain = (
            f"[{label}]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},{common_tail}[cont]"
        )
        return chain, "cont"

    if mode == "Barras pretas":
        chain = (
            f"[{label}]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,{common_tail}[cont]"
        )
        return chain, "cont"

    bg_width = max(2, (width // BACKGROUND_BLUR_DIVISOR) // 2 * 2)
    bg_height = max(2, (height // BACKGROUND_BLUR_DIVISOR) // 2 * 2)
    chain = (
        f"[{label}]split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale={bg_width}:{bg_height}:force_original_aspect_ratio=increase,"
        f"crop={bg_width}:{bg_height},gblur=sigma=12,"
        f"scale={width}:{height}:flags=bilinear[bg];"
        f"[fgsrc]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,{common_tail}[cont]"
    )
    return chain, "cont"


def _render_normalized_intro(
    media: PreparedIntroMedia,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: float,
    duration: float,
) -> None:
    """Cria um MP4 sem áudio contendo somente a nova mídia do primeiro take.

    A normalização em uma etapa separada evita que timestamps, rotação, VFR,
    codecs de celular ou o uso de ``-stream_loop`` interfiram na concatenação
    final e façam o vídeo original aparecer no lugar da mídia enviada.
    """
    ffmpeg = require_binary("ffmpeg")
    fps_text = f"{fps:.6f}".rstrip("0").rstrip(".")

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if media.kind == "image":
        command.extend([
            "-loop",
            "1",
            "-framerate",
            fps_text,
            "-i",
            media.input_path,
        ])
    else:
        # Repete somente o vídeo novo quando ele for menor que o primeiro take.
        command.extend(["-stream_loop", "-1", "-i", media.input_path])

    filter_graph = (
        f"[0:v]trim=duration={duration:.6f},setpts=PTS-STARTPTS,"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,settb=AVTB,fps={fps_text},format=yuv420p[vintro]"
    )

    command.extend([
        "-filter_threads",
        str(FFMPEG_THREADS),
        "-filter_complex_threads",
        str(FFMPEG_THREADS),
        "-filter_complex",
        filter_graph,
        "-map",
        "[vintro]",
        "-t",
        f"{duration:.6f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        OUTPUT_PRESET,
        "-crf",
        str(OUTPUT_CRF),
        "-pix_fmt",
        "yuv420p",
        "-threads",
        str(FFMPEG_THREADS),
        "-movflags",
        "+faststart",
        str(output_path),
    ])

    run_command(command, timeout=max(180, int(duration * 20)))
    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise MediaError("Não foi possível preparar o vídeo enviado para o primeiro take.")

    normalized = probe_video(output_path)
    if normalized.duration + 0.08 < duration:
        raise MediaError(
            "O vídeo enviado para o primeiro take terminou antes do esperado durante a preparação."
        )


def _build_filter_complex(
    *,
    width: int,
    height: int,
    fps: float,
    duration: float,
    transition: float,
    has_continuation: bool,
    fit_mode: str,
    caption_overlays: list[CaptionOverlay],
    has_audio: bool,
) -> tuple[str, str, str | None]:
    fps_text = f"{fps:.6f}".rstrip("0").rstrip(".")
    chains: list[str] = []

    intro_duration = transition if has_continuation else duration
    chains.append(
        f"[0:v]trim=duration={intro_duration:.6f},setpts=PTS-STARTPTS,"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,settb=AVTB,fps={fps_text},format=yuv420p[introbase]"
    )

    intro_label = "introbase"
    for index, overlay in enumerate(caption_overlays):
        input_index = 2 + index
        source_label = f"caption{index}src"
        output_label = f"introcaption{index}"
        chains.append(f"[{input_index}:v]format=rgba[{source_label}]")
        # A imagem da legenda entra em loop no FFmpeg. Sem shortest/trim, o filtro
        # overlay pode prolongar o primeiro take indefinidamente e congelar seu
        # último quadro, impedindo a concatenação com a continuação.
        chains.append(
            f"[{intro_label}][{source_label}]overlay=0:0:"
            f"shortest=1:eof_action=pass:repeatlast=0:"
            f"enable='between(t,{overlay.start:.3f},{overlay.end:.3f})'[{output_label}]"
        )
        intro_label = output_label

    # Garantia adicional: independentemente das camadas aplicadas, o primeiro
    # segmento termina exatamente no segundo da troca.
    chains.append(
        f"[{intro_label}]trim=duration={intro_duration:.6f},"
        f"setpts=PTS-STARTPTS[introready]"
    )
    intro_label = "introready"

    if has_continuation:
        # Margem de segurança de ~2 quadros: vídeos de celular (VFR) podem ter
        # timestamps levemente imprecisos e deixar 1-2 quadros da cena antiga
        # "vazarem" no início da continuação. Pular 0.07s do vídeo original a
        # partir da troca elimina isso sem efeito visível no take de continuação.
        safety_margin = 0.07
        continuation_start = min(transition + safety_margin, max(transition, duration - 0.05))
        chains.append(
            f"[1:v]trim=start={continuation_start:.6f},setpts=PTS-STARTPTS[contsrc]"
        )
        fit_chain, cont_label = _fit_filter("contsrc", width, height, fps, fit_mode)
        chains.append(fit_chain)
        chains.append(f"[{intro_label}][{cont_label}]concat=n=2:v=1:a=0[vout]")
    else:
        chains.append(f"[{intro_label}]null[vout]")

    audio_label: str | None = None
    if has_audio:
        chains.append("[1:a:0]asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0[aout]")
        audio_label = "aout"

    return ";".join(chains), "vout", audio_label


def _resolve_transition(
    video_path: str,
    video_duration: float,
    transition_mode: str,
    manual_seconds: float | None,
) -> tuple[float, TransitionResult | None, bool]:
    if transition_mode == "Sem vídeo de continuação":
        return video_duration, None, False

    if transition_mode == "Informar o segundo manualmente":
        if manual_seconds is None or not math.isfinite(float(manual_seconds)):
            raise MediaError("Digite o segundo em que começa o vídeo de continuação.")
        transition = float(manual_seconds)
        if transition <= 0 or transition >= video_duration:
            raise MediaError(
                f"O segundo da transição deve ser maior que 0 e menor que {video_duration:.2f}s."
            )
        return transition, None, True

    detected = detect_intro_end(video_path)
    if detected.seconds is None:
        return video_duration, detected, False
    transition = min(max(detected.seconds, 0.1), video_duration - 0.05)
    return transition, detected, True



def process_video(
    *,
    photo_path: str,
    video_path: str,
    transition_mode: str,
    manual_transition_seconds: float | None,
    caption_mode: str,
    manual_caption_text: str,
    caption_position: str,
    caption_font_percent: float,
    continuation_fit_mode: str,
    language: str,
    progress: ProgressFn | None = None,
) -> ProcessResult:
    """Gera o vídeo final. ``photo_path`` aceita imagem ou vídeo por compatibilidade de API."""

    cleanup_old_jobs()
    require_binary("ffmpeg")
    require_binary("ffprobe")

    if not photo_path:
        raise MediaError("Envie a nova mídia da personagem: uma imagem ou um vídeo.")
    if not video_path:
        raise MediaError("Envie o vídeo original com áudio e, quando existir, o trecho de continuação.")

    job_dir = WORK_ROOT / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=False)
    output_path = job_dir / "video_pronto.mp4"
    normalized_intro_path = job_dir / "primeiro_take_normalizado.mp4"

    _safe_progress(progress, 0.04, "Lendo os arquivos...")
    info = probe_video(video_path)
    if info.duration <= 0:
        raise MediaError("A duração do vídeo original é inválida.")
    if info.duration > MAX_VIDEO_MINUTES * 60:
        raise MediaError(
            f"O vídeo original tem {info.duration / 60:.1f} minutos. O limite configurado é "
            f"{MAX_VIDEO_MINUTES:.0f} minutos."
        )

    intro_media = prepare_intro_media(
        photo_path,
        job_dir,
        max_long_edge=OUTPUT_MAX_LONG_EDGE,
    )
    if intro_media.kind == "video" and (intro_media.duration or 0) <= 0:
        raise MediaError("O vídeo enviado como mídia inicial possui duração inválida.")

    output_w = intro_media.output_width
    output_h = intro_media.output_height

    _safe_progress(progress, 0.16, "Identificando onde termina o primeiro take...")
    transition, detection, has_continuation = _resolve_transition(
        video_path,
        info.duration,
        transition_mode,
        manual_transition_seconds,
    )
    intro_duration = transition if has_continuation else info.duration

    caption_events: list[CaptionEvent] = []
    transcript_summary = ""
    if caption_mode == "Transcrever o áudio automaticamente":
        if not info.has_audio:
            raise MediaError("O vídeo original não possui áudio para transcrever. Use legenda manual ou sem legenda.")
        _safe_progress(progress, 0.30, "Transcrevendo o áudio do primeiro take...")
        caption_events, transcript_summary = transcribe_intro(
            video_path,
            job_dir,
            intro_duration,
            language=language or None,
        )
        if not caption_events:
            transcript_summary = "O Whisper não encontrou fala clara no primeiro trecho."
    elif caption_mode == "Copiar o texto escrito no vídeo original":
        _safe_progress(progress, 0.30, "Lendo a legenda escrita no primeiro take...")
        caption_events, transcript_summary = read_burned_caption(
            video_path,
            intro_duration,
            language=language or "pt",
        )
        if not caption_events:
            raise MediaError(
                "Não foi possível ler nenhuma legenda escrita no primeiro take. "
                "Use 'Usar um texto fixo' e digite a legenda manualmente."
            )
    elif caption_mode == "Usar um texto fixo":
        caption_events = manual_caption(manual_caption_text, intro_duration)
        if not caption_events:
            raise MediaError("Digite o texto da legenda manual.")

    caption_overlays: list[CaptionOverlay] = []
    if caption_events:
        caption_overlays = render_caption_overlays(
            caption_events,
            job_dir,
            width=output_w,
            height=output_h,
            font_percent=float(caption_font_percent),
            position=caption_position,
        )

    media_word = "vídeo" if intro_media.kind == "video" else "imagem"
    _safe_progress(progress, 0.46, f"Preparando o {media_word} enviado para o primeiro take...")
    _render_normalized_intro(
        intro_media,
        normalized_intro_path,
        width=output_w,
        height=output_h,
        fps=info.fps,
        duration=intro_duration,
    )

    _safe_progress(progress, 0.60, "Juntando o novo primeiro take ao vídeo original...")
    filter_complex, video_label, audio_label = _build_filter_complex(
        width=output_w,
        height=output_h,
        fps=info.fps,
        duration=info.duration,
        transition=transition,
        has_continuation=has_continuation,
        fit_mode=continuation_fit_mode,
        caption_overlays=caption_overlays,
        has_audio=info.has_audio,
    )

    ffmpeg = require_binary("ffmpeg")
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-filter_threads",
        str(FFMPEG_THREADS),
        "-filter_complex_threads",
        str(FFMPEG_THREADS),
    ]
    # O input 0 é sempre o trecho já normalizado da nova mídia. O original
    # permanece exclusivamente no input 1 e só é usado após a transição.
    command.extend(["-i", str(normalized_intro_path)])
    command.extend(["-i", str(video_path)])
    for overlay in caption_overlays:
        command.extend([
            "-loop",
            "1",
            "-framerate",
            f"{info.fps:.6f}",
            "-i",
            overlay.path,
        ])
    command.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            f"[{video_label}]",
        ]
    )
    if audio_label:
        command.extend(["-map", f"[{audio_label}]"])

    command.extend(
        [
            "-t",
            f"{info.duration:.6f}",
            "-c:v",
            "libx264",
            "-preset",
            OUTPUT_PRESET,
            "-crf",
            str(OUTPUT_CRF),
            "-pix_fmt",
            "yuv420p",
            "-threads",
            str(FFMPEG_THREADS),
            "-movflags",
            "+faststart",
        ]
    )
    if audio_label:
        command.extend(["-c:a", "aac", "-b:a", OUTPUT_AUDIO_BITRATE])
    else:
        command.append("-an")
    command.extend(["-max_muxing_queue_size", "2048", str(output_path)])

    timeout = max(240, int(info.duration * 25))
    run_command(command, timeout=timeout)
    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise MediaError("O FFmpeg terminou sem produzir um vídeo válido.")

    _safe_progress(progress, 0.96, "Validando o vídeo final...")
    final_info = probe_video(output_path)

    detection_line = ""
    if detection:
        detection_line = f"\n- Detecção: {detection.message} Confiança aproximada: {detection.confidence:.0%}."
    if has_continuation:
        structure_line = (
            f"- Nova mídia e legenda: 0s até {transition:.2f}s.\n"
            f"- Vídeo original de continuação: {transition:.2f}s até {info.duration:.2f}s."
        )
    else:
        structure_line = f"- A nova mídia permanece durante os {info.duration:.2f}s do vídeo."

    media_line = (
        f"- Mídia inicial reconhecida automaticamente: **{media_word}**.\n"
        "- O primeiro take foi normalizado separadamente antes da junção, garantindo que o vídeo original só apareça após a troca."
    )
    loop_line = ""
    if intro_media.kind == "video":
        source_duration = intro_media.duration or 0.0
        if source_duration + 0.05 < intro_duration:
            loop_line = (
                f"\n- O vídeo inicial tinha {source_duration:.2f}s e foi repetido automaticamente "
                f"para preencher {intro_duration:.2f}s."
            )
        elif source_duration > intro_duration + 0.05:
            loop_line = (
                f"\n- O vídeo inicial tinha {source_duration:.2f}s e foi cortado em "
                f"{intro_duration:.2f}s para coincidir com a troca do take."
            )

    size_note = ""
    if max(intro_media.original_width, intro_media.original_height) > max(output_w, output_h) + 1:
        size_note = (
            f"\n- Resolução otimizada: a mídia era {intro_media.original_width}×{intro_media.original_height}; "
            f"a saída ficou {output_w}×{output_h}, preservando a proporção."
        )
    elif (intro_media.original_width, intro_media.original_height) != (output_w, output_h):
        size_note = (
            f"\n- Compatibilidade H.264: a mídia era {intro_media.original_width}×{intro_media.original_height}; "
            f"a saída ficou {output_w}×{output_h} com dimensões pares."
        )

    transcript_line = ""
    if transcript_summary:
        transcript_line = f"\n\n**Transcrição identificada**\n{transcript_summary.strip()}"

    report = (
        "### Vídeo gerado\n"
        f"- Saída: {final_info.width}×{final_info.height}, {final_info.fps:.2f} FPS, "
        f"{final_info.duration:.2f}s.\n"
        f"{media_line}\n"
        f"{structure_line}\n"
        f"- Áudio original: {'mantido do começo ao fim' if info.has_audio else 'o arquivo não possuía áudio'}."
        f"{loop_line}{detection_line}{size_note}{transcript_line}"
    )

    _safe_progress(progress, 1.0, "Pronto.")
    return ProcessResult(str(output_path), report, transition)
