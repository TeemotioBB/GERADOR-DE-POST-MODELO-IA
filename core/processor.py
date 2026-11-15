from __future__ import annotations

import math
import json
import shutil
import time
import uuid
import logging
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
from .media import MediaError, PreparedIntroMedia, prepare_intro_media, probe_video, require_binary, run_command, scaled_even_dimensions
from .transition import TransitionResult, detect_intro_end

ProgressFn = Callable[[float, str], None]
logger = logging.getLogger(__name__)


# Perfis definidos no servidor, nunca aceitos como parâmetros de FFmpeg arbitrários.
QUALITY_STANDARD = "Padrão (1080p)"
QUALITY_ECONOMY = "Econômico (720p)"


def export_settings(source_width: int, source_height: int, source_fps: float,
                    quality_profile: str = QUALITY_STANDARD) -> tuple[int, int, float, int, str]:
    """Preserva o formato do vídeo original; limita resolução/fps/custo por perfil."""
    if quality_profile not in {QUALITY_STANDARD, QUALITY_ECONOMY}:
        raise MediaError("Escolha um perfil de exportação válido.")
    requested_edge = 1280 if quality_profile == QUALITY_ECONOMY else 1920
    cap = min(requested_edge, OUTPUT_MAX_LONG_EDGE) if OUTPUT_MAX_LONG_EDGE else requested_edge
    width, height = scaled_even_dimensions(source_width, source_height, cap)
    fps = min(source_fps, 30.0)  # reduz render de 60fps a 30fps, suficiente para o modo econômico
    crf = 23 if quality_profile == QUALITY_ECONOMY else OUTPUT_CRF
    bitrate = "128k" if quality_profile == QUALITY_ECONOMY else OUTPUT_AUDIO_BITRATE
    return width, height, fps, crf, bitrate



@dataclass(frozen=True)
class ProcessResult:
    output_path: str
    report: str
    transition_seconds: float


_last_cleanup = 0.0


def cleanup_old_jobs(*, force: bool = False) -> None:
    """Limpeza oportunista: sem thread, timer ou tráfego que mantenha o Railway ativo."""
    global _last_cleanup
    now = time.monotonic()
    if not force and now - _last_cleanup < 900:
        return
    _last_cleanup = now
    cutoff = time.time() - TEMP_MAX_AGE_HOURS * 3600
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    for item in WORK_ROOT.iterdir():
        try:
            if item.stat().st_mtime >= cutoff:
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        except OSError:
            continue

    # Uploads e prévias também passam pelo cache interno do Gradio. Removê-los
    # somente em eventos de uso, sem timers, para permitir Railway Serverless.
    import os
    cache_root = Path(os.getenv("GRADIO_TEMP_DIR", "/tmp/gradio"))
    if cache_root.is_dir():
        cache_hours = max(2.0, float(os.getenv("GRADIO_CACHE_MAX_AGE_HOURS", "6")))
        cache_cutoff = time.time() - 3600 * cache_hours
        for item in cache_root.iterdir():
            try:
                if item.stat().st_mtime < cache_cutoff:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
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
    audio_input_index: int = 1,
    caption_input_start: int = 2,
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
        input_index = caption_input_start + index
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
        # O áudio é mantido durante todos os takes; o vídeo da continuação
        # é cortado exatamente na transição para evitar dessincronização.
        chains.append(
            f"[1:v]trim=start={transition:.6f},setpts=PTS-STARTPTS[contsrc]"
        )
        fit_chain, cont_label = _fit_filter("contsrc", width, height, fps, fit_mode)
        chains.append(fit_chain)
        chains.append(f"[{intro_label}][{cont_label}]concat=n=2:v=1:a=0[vout]")
    else:
        chains.append(f"[{intro_label}]null[vout]")

    audio_label: str | None = None
    if has_audio:
        chains.append(
            f"[{audio_input_index}:a:0]asetpts=PTS-STARTPTS,"
            f"aresample=async=1:first_pts=0[aout]"
        )
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



def _caption_events_from_override(
    rows: list[dict] | None,
    text: str,
    duration: float,
) -> list[CaptionEvent]:
    """A legenda revisada é SEMPRE única e fixa durante todo o primeiro take."""
    del rows  # tempos antigos não são mais usados para legenda fixa
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    return [CaptionEvent(0.0, max(0.15, duration), cleaned)]


def _validar_audio_externo(audio_path: str, duracao_video: float) -> None:
    """Valida uma faixa de áudio separada, sem re-encodar nem tocar mídia na etapa de upload."""
    arquivo = Path(audio_path)
    if not arquivo.is_file():
        raise MediaError("O arquivo de áudio de recuperação não foi encontrado.")
    if arquivo.stat().st_size > 200 * 1024 * 1024:
        raise MediaError("O áudio de recuperação excede 200 MB.")
    cmd = [
        require_binary("ffprobe"), "-v", "error", "-show_streams",
        "-show_format", "-of", "json", str(arquivo),
    ]
    try:
        data = json.loads(run_command(cmd, timeout=60).stdout)
        faixas = data.get("streams", [])
        if not any(s.get("codec_type") == "audio" for s in faixas):
            raise MediaError("O arquivo de recuperação não contém uma faixa de áudio.")
        duracao = float(data.get("format", {}).get("duration", "0"))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise MediaError("Não foi possível verificar a duração do áudio adicional.") from exc
    if not math.isfinite(duracao) or duracao <= 0:
        raise MediaError("O áudio de recuperação possui duração inválida.")
    if duracao < duracao_video - 0.1:
        raise MediaError(
            f"O áudio de recuperação tem {duracao:.1f}s e o vídeo tem {duracao_video:.1f}s. "
            "Envie uma faixa que cubra toda a duração do vídeo."
        )


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
    caption_events_override: list[dict] | None = None,
    progress: ProgressFn | None = None,
    quality_profile: str = QUALITY_STANDARD,
    audio_recovery_path: str | None = None,
) -> ProcessResult:
    """Gera o vídeo final. ``photo_path`` aceita imagem ou vídeo por compatibilidade de API."""

    started_at = time.monotonic()
    cleanup_old_jobs()
    require_binary("ffmpeg")
    require_binary("ffprobe")

    if not photo_path:
        raise MediaError("Envie a nova mídia da personagem: uma imagem ou um vídeo.")
    if not video_path:
        raise MediaError("Envie o vídeo original com áudio e, quando existir, o trecho de continuação.")

    _safe_progress(progress, 0.04, "Lendo os arquivos...")
    info = probe_video(video_path)
    if info.duration <= 0:
        raise MediaError("A duração do vídeo original é inválida.")
    recuperar_audio = not info.has_audio and bool(audio_recovery_path)
    if not info.has_audio and not recuperar_audio:
        raise MediaError(
            "O vídeo original não contém faixa de áudio. Na seção 'Recuperar áudio', "
            "envie um arquivo de áudio com a mesma duração do Reel, "
            "ou envie o original em MP4 com som."
        )
    if recuperar_audio:
        _validar_audio_externo(str(audio_recovery_path), info.duration)
    if info.duration > MAX_VIDEO_MINUTES * 60:
        raise MediaError(
            f"O vídeo original tem {info.duration / 60:.1f} minutos. O limite configurado é "
            f"{MAX_VIDEO_MINUTES:.0f} minutos."
        )

    # Não cria diretórios temporários antes de validar o original e o áudio.
    job_dir = WORK_ROOT / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=False)
    output_path = job_dir / "video_pronto.mp4"
    normalized_intro_path = job_dir / "primeiro_take_normalizado.mp4"

    output_w, output_h, output_fps, output_crf, output_bitrate = export_settings(
        info.width, info.height, info.fps, quality_profile
    )
    intro_media = prepare_intro_media(
        photo_path,
        job_dir,
        max_long_edge=max(output_w, output_h),
    )
    if intro_media.kind == "video" and (intro_media.duration or 0) <= 0:
        raise MediaError("O vídeo enviado como mídia inicial possui duração inválida.")


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
    if caption_events_override is not None and caption_mode != "Sem legenda":
        caption_events = _caption_events_from_override(
            caption_events_override, manual_caption_text, intro_duration
        )
        transcript_summary = "Texto revisado pelo usuário antes da geração."
    elif caption_mode == "Transcrever o áudio automaticamente":
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

    # Regra do produto: a legenda é uma única frase fixa durante todo o primeiro take.
    # OCR/Whisper podem produzir vários eventos internos, mas eles são unidos antes do render.
    if caption_events and caption_mode in {
        "Transcrever o áudio automaticamente",
        "Copiar o texto escrito no vídeo original",
    }:
        combined = " ".join(
            " ".join(event.text.split()) for event in caption_events if event.text.strip()
        ).strip()
        caption_events = [CaptionEvent(0.0, max(0.15, intro_duration), combined)] if combined else []

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
        fps=output_fps,
        duration=intro_duration,
    )

    _safe_progress(progress, 0.60, "Juntando o novo primeiro take ao vídeo original...")
    filter_complex, video_label, audio_label = _build_filter_complex(
        width=output_w,
        height=output_h,
        fps=output_fps,
        duration=info.duration,
        transition=transition,
        has_continuation=has_continuation,
        fit_mode=continuation_fit_mode,
        caption_overlays=caption_overlays,
        has_audio=True,
        audio_input_index=2 if recuperar_audio else 1,
        caption_input_start=3 if recuperar_audio else 2,
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
    if recuperar_audio:
        command.extend(["-i", str(audio_recovery_path)])
    for overlay in caption_overlays:
        command.extend([
            "-loop",
            "1",
            "-framerate",
            f"{output_fps:.6f}",
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
            str(output_crf),
            "-pix_fmt",
            "yuv420p",
            "-threads",
            str(FFMPEG_THREADS),
            "-movflags",
            "+faststart",
        ]
    )
    if audio_label:
        command.extend(["-c:a", "aac", "-b:a", output_bitrate])
    else:
        command.append("-an")
    command.extend(["-max_muxing_queue_size", "2048", str(output_path)])

    timeout = max(240, int(info.duration * 25))
    run_command(command, timeout=timeout)
    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise MediaError("O FFmpeg terminou sem produzir um vídeo válido.")

    _safe_progress(progress, 0.96, "Validando o vídeo final...")
    final_info = probe_video(output_path)
    if not final_info.has_audio:
        # Nunca anunciar sucesso caso a saída tenha perdido a faixa do original.
        output_path.unlink(missing_ok=True)
        raise MediaError(
            "A exportação terminou, mas a faixa de áudio original não foi "
            "encontrada no MP4 final. O vídeo foi rejeitado para evitar "
            "um download sem som. Tente novamente ou envie o arquivo original."
        )

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

    size_note = (
        f"\n- Formato de saída baseado no original: {output_w}×{output_h}, "
        "adaptando a nova mídia com margens pretas quando necessário."
    )

    transcript_line = ""
    if transcript_summary:
        transcript_line = f"\n\n**Transcrição identificada**\n{transcript_summary.strip()}"

    audio_report = (
        "Áudio de recuperação enviado pelo usuário, confirmado no MP4 final"
        if recuperar_audio else
        "Áudio original detectado e confirmado no MP4 final"
    )
    report = (
        "### Vídeo gerado\n"
        f"- Saída: {final_info.width}×{final_info.height}, {final_info.fps:.2f} FPS, "
        f"{final_info.duration:.2f}s.\n"
        f"{media_line}\n"
        f"{structure_line}\n"
        f"- Áudio: {audio_report} (AAC, {output_bitrate})."
        f"{loop_line}{detection_line}{size_note}{transcript_line}"
    )

    # O Gradio entrega o MP4 final. Intermediários de vários MB não precisam
    # permanecer em disco até o próximo usuário abrir a ferramenta.
    for temporary in job_dir.iterdir():
        if temporary != output_path:
            if temporary.is_dir():
                shutil.rmtree(temporary, ignore_errors=True)
            else:
                temporary.unlink(missing_ok=True)
    logger.info("render_complete duration=%.2fs output_mb=%.2f quality=%s",
                time.monotonic() - started_at,
                output_path.stat().st_size / (1024 * 1024), quality_profile)
    _safe_progress(progress, 1.0, "Pronto.")
    return ProcessResult(str(output_path), report, transition)
