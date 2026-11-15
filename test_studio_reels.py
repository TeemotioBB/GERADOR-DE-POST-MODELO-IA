"""Testes locais: python -m pytest -q (FFmpeg necessário para integração)."""
from pathlib import Path
import shutil
import subprocess

import pytest
from PIL import Image

from core.media import MediaError, probe_video
from core.processor import (
    QUALITY_ECONOMY, QUALITY_STANDARD, _build_filter_complex,
    export_settings, process_video,
)


def test_export_sizes_and_settings():
    assert export_settings(1080, 1920, 60, QUALITY_STANDARD)[:3] == (1080, 1920, 30)
    assert export_settings(1080, 1920, 60, QUALITY_ECONOMY) == (720, 1280, 30, 23, "128k")
    assert export_settings(270, 480, 25, QUALITY_ECONOMY)[:3] == (270, 480, 25)
    assert export_settings(1920, 1080, 30, QUALITY_ECONOMY)[:2] == (1280, 720)


def test_reject_unknown_quality():
    with pytest.raises(MediaError):
        export_settings(1080, 1920, 30, "sinal arbitrário")


def test_sync_no_hidden_video_margin():
    graph, video, audio = _build_filter_complex(
        width=270, height=480, fps=30, duration=4, transition=2,
        has_continuation=True, fit_mode="Barras pretas", caption_overlays=[], has_audio=True,
    )
    assert "trim=start=2.000000" in graph
    assert "2.070000" not in graph
    assert "asetpts=PTS-STARTPTS" in graph
    assert (video, audio) == ("vout", "aout")


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg requerido")
def test_complete_render_both_takes(tmp_path, monkeypatch):
    # Cria um vídeo pequeno, com um corte em 2s e áudio; dispensa Instagram e Whisper.
    original = tmp_path / "original.mp4"
    photo = tmp_path / "intro.png"
    Image.new("RGB", (480, 270), (80, 150, 80)).save(photo)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=270x480:r=30:d=2",
        "-f", "lavfi", "-i", "color=c=blue:s=270x480:r=30:d=2",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=4",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2:a:0", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(original),
    ], check=True, timeout=50)
    import core.processor as processor
    monkeypatch.setattr(processor, "WORK_ROOT", tmp_path / "outputs")
    result = process_video(
        photo_path=str(photo), video_path=str(original),
        transition_mode="Informar o segundo manualmente", manual_transition_seconds=2,
        caption_mode="Usar um texto fixo", manual_caption_text="LEGENDA DE TESTE",
        caption_position="Centro", caption_font_percent=4.6,
        continuation_fit_mode="Barras pretas", language="pt",
        quality_profile=QUALITY_ECONOMY,
    )
    output = Path(result.output_path)
    assert output.exists() and output.stat().st_size > 1024
    details = probe_video(output)
    assert details.width == 270 and details.height == 480
    assert details.has_audio and 3.8 <= details.duration <= 4.1
    assert len(list(output.parent.iterdir())) == 1, "Arquivos temporários intermediários removidos"
