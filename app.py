"""Studio Reels 2.0 — interface em três etapas, baseada no motor do projeto original.

Sem agendadores, requests de telemetria ou limpeza em background: compatível
com a suspensão Serverless do Railway (quando ativada no painel).
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import uuid
from pathlib import Path

import cv2
import gradio as gr
import uvicorn
from fastapi import FastAPI, Header
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from core.captions import CaptionEvent, clean_review_text, read_burned_caption, render_caption_overlays, transcribe_intro
from core.config import APP_NAME, MAX_INSTAGRAM_DOWNLOAD_MB, MAX_VIDEO_MINUTES, WORK_ROOT
from core.media import MediaError, probe_video, scaled_even_dimensions
from core.processor import (
    QUALITY_ECONOMY, QUALITY_STANDARD, cleanup_old_jobs, export_settings, process_video,
)
from core.transition import detect_intro_end
import url_import

# Funciona sem daemon/cron. Limpeza também ocorre ao abrir a interface.
cleanup_old_jobs(force=True)

CSS = r"""
:root { --studio-blue: #4458e8; --studio-ink: #18233b; --studio-muted: #65718b; }
.gradio-container { max-width: 1320px !important; margin: auto !important; padding: 18px 24px 60px !important; }
#studio-header { background: linear-gradient(110deg,#121c33,#263768 63%,#465ad9);
  border-radius: 22px; padding: 27px 30px; margin-bottom: 18px; color:#fff; }
#studio-header .eyebrow { color:#aabaff; font-size: .72rem; letter-spacing:.16em; font-weight:800; margin-bottom:9px; }
#studio-header h1 { color:#fff !important; font-size:2rem; line-height:1.12; letter-spacing:-.04em; margin:0 0 9px; }
#studio-header p { color:#d1daef !important; margin:0; line-height:1.5; font-size:.95rem; }
#studio-header .status-pill {display:inline-block; margin-top:16px; padding:6px 12px; border-radius:20px;
 background:rgba(255,255,255,.13);color:#edf2ff;font-size:.74rem;}
.studio-panel { border:1px solid #e2e7f0 !important; background:#fff !important;
 border-radius:17px !important; padding:16px !important; box-shadow: 0 5px 22px rgba(20,36,78,.045); }
.studio-section { font-weight:800; font-size:1.02rem; margin:0 0 6px; color:#192942; letter-spacing:-.015em; }
.studio-sub {font-size:.82rem; color:#6b7790; margin:0 0 12px; line-height:1.5;}
.studio-index {display:inline-block; color:white; background:var(--studio-blue); border-radius:9px;
 padding:4px 9px; font-size:.78rem; margin-right:8px; }
#generate-btn { min-height:54px !important; font-size:1rem !important; font-weight:800 !important;
 border-radius:12px !important; background:linear-gradient(100deg,#3e50dc,#6575fa) !important; border:0 !important; }
#analyze-btn, #import-btn {font-weight:700; min-height:43px;}
#preview-panel {position:sticky; top:16px; align-self:flex-start;}
#preview-panel .gradio-image {max-width:430px;margin:auto;}
#result-panel .gradio-video {max-width:430px;margin:auto;}
#download-output {margin-top:6px;}
#source-tabs {margin-top:0;}
#source-tabs [role=tablist] {border-radius:9px !important; overflow:hidden;}
.muted-caption {color:#6b7790;font-size:.79rem;}
.gradio-container textarea, .gradio-container input, .gradio-container select {border-radius:10px !important;}
@media (max-width:850px) {
 .gradio-container{padding:10px 12px 42px !important;}
 #studio-header{padding:22px 19px; border-radius:16px;}
 #studio-header h1{font-size:1.65rem;}
 #preview-panel{position:static;}
 .studio-panel{padding:11px !important;}
}
"""

_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _language_code(label: str) -> str:
    return {"Português": "pt", "Detectar automaticamente": "", "Inglês": "en", "Espanhol": "es"}.get(label, "pt")


def _validate_original(path: str):
    info = probe_video(path)
    if info.duration <= 0 or info.duration > MAX_VIDEO_MINUTES * 60:
        raise MediaError(f"O vídeo original deve ter no máximo {MAX_VIDEO_MINUTES:g} minutos.")
    return info


def transition_visibility(choice: str):
    return gr.update(visible=choice == "Informar o segundo manualmente")


def caption_review_visibility(choice: str):
    return (gr.update(visible=choice != "Sem legenda"), None,
            "Método alterado. Clique em **Reanalisar** para aplicar ao vídeo atual.")


def importar_video_url(url: str, permitir_sem_audio: bool = False) -> str:
    if not (url or "").strip():
        raise gr.Error("Cole um link válido do Instagram.")
    cleanup_old_jobs()
    job_id = uuid.uuid4().hex
    try:
        caminho, _ = url_import.baixar_video(
            url=url.strip(), pasta_destino=str(WORK_ROOT),
            identificador=job_id, limite_mb=MAX_INSTAGRAM_DOWNLOAD_MB,
            permitir_sem_audio=permitir_sem_audio,
        )
        _validate_original(caminho)
        return caminho
    except (url_import.VideoImportError, MediaError) as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(f"Não foi possível importar esse vídeo: {exc}") from exc


def handle_url_import(url: str, permitir_sem_audio: bool):
    caminho = importar_video_url(url, permitir_sem_audio)
    info = _validate_original(caminho)
    audio_status = "🔊 Faixa de áudio detectada" if info.has_audio else "⚠️ VÍDEO SEM ÁUDIO — anexe uma faixa em Recuperar áudio antes de gerar"
    return (f"✅ Importado: **{info.duration:.1f}s** · {info.width}×{info.height} · {audio_status}. Analisando...",
            caminho, gr.update(value="Detectar automaticamente"), gr.update(value=None, visible=False))


def handle_local_video(path: str | None):
    if not path:
        return ("Envie um arquivo para iniciar.", None, gr.update(value="Detectar automaticamente"),
                gr.update(value=None, visible=False))
    try:
        info = _validate_original(path)
        audio_status = "🔊 Faixa de áudio detectada" if info.has_audio else "⚠️ VÍDEO SEM ÁUDIO — anexe uma faixa em Recuperar áudio antes de gerar"
        return (f"✅ Enviado: **{info.duration:.1f}s** · {info.width}×{info.height} · {audio_status}. Analisando...",
                path, gr.update(value="Detectar automaticamente"), gr.update(value=None, visible=False))
    except MediaError as exc:
        raise gr.Error(str(exc)) from exc


def _analysis_work_dir() -> Path:
    directory = WORK_ROOT / f"analysis_{uuid.uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def analyze_source(video_path, caption_source, language_label,
                   requested_transition_mode, manual_transition_seconds):
    if not video_path:
        raise gr.Error("Adicione o vídeo original primeiro.")
    cleanup_old_jobs()
    try:
        info = _validate_original(video_path)
    except MediaError as exc:
        raise gr.Error(str(exc)) from exc

    if requested_transition_mode == "Informar o segundo manualmente":
        try:
            transition = float(manual_transition_seconds)
        except (TypeError, ValueError) as exc:
            raise gr.Error("Informe o segundo em que começa o segundo take.") from exc
        if not 0 < transition < info.duration:
            raise gr.Error(f"O segundo deve estar entre 0 e {info.duration:.2f}s.")
        has_continuation = True
        message = f"Troca manual: **{transition:.2f}s**."
    elif requested_transition_mode == "Sem vídeo de continuação":
        transition = info.duration
        has_continuation = False
        message = "Vídeo tratado sem segundo take."
    else:
        try:
            detected = detect_intro_end(video_path)
        except MediaError as exc:
            raise gr.Error(str(exc)) from exc
        if detected.seconds is None:
            raise gr.Error(
                "Não detectei a transição com confiança. Abra as opções avançadas, "
                "informe o segundo manualmente e clique em Reanalisar."
            )
        transition = min(max(float(detected.seconds), 0.1), info.duration - 0.05)
        has_continuation = True
        message = f"Troca detectada: **{transition:.2f}s** · confiança aproximada {detected.confidence:.0%}."

    duration = transition if has_continuation else info.duration
    language = _language_code(language_label)
    events: list[CaptionEvent] = []
    summary = ""
    if caption_source == "Copiar texto escrito no vídeo":
        try:
            events, summary = read_burned_caption(video_path, duration, language=language or "pt")
        except MediaError as exc:
            raise gr.Error(str(exc)) from exc
    elif caption_source == "Transcrever o áudio":
        if not info.has_audio:
            raise gr.Error("O original não possui áudio para transcrição.")
        directory = _analysis_work_dir()
        try:
            events, summary = transcribe_intro(video_path, directory, duration, language=language or None)
        except MediaError as exc:
            raise gr.Error(str(exc)) from exc
        finally:
            shutil.rmtree(directory, ignore_errors=True)
    elif caption_source not in {"Digitar manualmente", "Sem legenda"}:
        raise gr.Error("Modo de legenda inválido.")

    text = clean_review_text(
        " ".join(" ".join(event.text.split()) for event in events if event.text.strip())
    ).strip()
    timings = [{"start": 0.0, "end": duration}] if text else []
    if caption_source == "Copiar texto escrito no vídeo" and not text:
        note = "⚠️ Não reconheci uma frase fixa. Você pode digitar ou corrigir o texto abaixo."
    elif caption_source == "Transcrever o áudio" and not text:
        note = "⚠️ Não encontrei fala clara. Digite manualmente ou escolha sem legenda."
    elif caption_source == "Digitar manualmente":
        note = "Digite uma frase para o primeiro take."
    elif caption_source == "Sem legenda":
        note = "Legenda desativada."
    else:
        note = "Texto identificado. Revise antes de gerar."

    state = {
        "video_path": video_path, "duration": info.duration,
        "transition": transition, "has_continuation": has_continuation,
        "caption_source": caption_source, "source_width": info.width,
        "source_height": info.height, "source_fps": info.fps,
    }
    audio_note = "🔊 Áudio original detectado." if info.has_audio else (
        "⚠️ **O arquivo original está sem faixa de áudio.** "
        "Para gerar um vídeo com som, importe outro arquivo com áudio."
    )
    details = f"✅ **Análise concluída.** {message} {audio_note} {note}"
    if summary and caption_source in {"Copiar texto escrito no vídeo", "Transcrever o áudio"}:
        line = summary.splitlines()[0] if summary.splitlines() else ""
        if line:
            details += f"\n\n<small>{line}</small>"
    return (
        details, gr.update(value=text, visible=caption_source != "Sem legenda"),
        timings, state,
        gr.update(value="Informar o segundo manualmente" if has_continuation else "Sem vídeo de continuação"),
        gr.update(value=transition if has_continuation else None, visible=has_continuation),
    )


def auto_analyze(video_path, caption_source, language_label, transition_mode, manual_transition_seconds):
    if not video_path:
        return ("Envie um vídeo para começar.", gr.update(value=""), [], None,
                gr.update(value="Detectar automaticamente"), gr.update(value=None, visible=False))
    return analyze_source(video_path, caption_source, language_label, transition_mode, manual_transition_seconds)


def _load_preview_base(path: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        pass
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise MediaError("Não foi possível abrir a mídia para a prévia.")
    try:
        ok, frame = capture.read()
        if not ok:
            raise MediaError("Não foi possível ler o primeiro quadro da nova mídia.")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


def preview_image(intro_media_path, analysis_state, reviewed_text, caption_source,
                  caption_position, caption_size, quality_profile, request: gr.Request):
    """Prévia em memória + imagem pequena; nunca executa o render completo."""
    if not intro_media_path or not analysis_state:
        return None, "Envie o original e a nova mídia para visualizar."
    try:
        output_w, output_h, _, _, _ = export_settings(
            int(analysis_state["source_width"]), int(analysis_state["source_height"]),
            float(analysis_state["source_fps"]), quality_profile,
        )
        w, h = scaled_even_dimensions(output_w, output_h, 720)
        incoming = _load_preview_base(intro_media_path)
        # Mesmo filtro do render: contain + preenchimento preto.
        incoming.thumbnail((w, h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 255))
        canvas.paste(incoming, ((w-incoming.width)//2, (h-incoming.height)//2))
        token = hashlib.sha256((request.session_hash if request and request.session_hash else "anonymous").encode()).hexdigest()[:20]
        preview_dir = WORK_ROOT / f"preview_{token}"
        preview_dir.mkdir(parents=True, exist_ok=True)
        if caption_source != "Sem legenda" and (reviewed_text or "").strip():
            overlays = render_caption_overlays(
                [CaptionEvent(0.0, 1.0, reviewed_text.strip())], preview_dir,
                width=w, height=h, font_percent=float(caption_size), position=caption_position,
            )
            if overlays:
                with Image.open(overlays[0].path) as overlay:
                    canvas = Image.alpha_composite(canvas, overlay.convert("RGBA"))
            for overlay in overlays:
                Path(overlay.path).unlink(missing_ok=True)
        # Um arquivo por atualização, sem crescer indefinidamente na sessão.
        for old in preview_dir.glob("preview_*.jpg"):
            old.unlink(missing_ok=True)
        output_path = preview_dir / f"preview_{uuid.uuid4().hex}.jpg"
        canvas.convert("RGB").save(output_path, "JPEG", quality=85, optimize=True)
        return str(output_path), f"Prévia rápida · {output_w}×{output_h} na exportação."
    except (MediaError, OSError, ValueError, KeyError) as exc:
        raise gr.Error(f"Erro ao montar prévia: {exc}") from exc


def generate_final(intro_media_path, video_path, analysis_state, caption_timings,
                   reviewed_text, caption_source, transition_mode, manual_transition_seconds,
                   continuation_fit_mode, caption_position, caption_font_percent,
                   language_label, quality_profile, audio_recovery_path, progress=gr.Progress()):
    if not intro_media_path or not video_path:
        raise gr.Error("Adicione o original e a nova mídia antes de gerar.")
    if not analysis_state or analysis_state.get("video_path") != video_path:
        raise gr.Error("Analise o vídeo original antes de gerar.")
    if analysis_state.get("caption_source") != caption_source:
        raise gr.Error("O modo de legenda mudou. Clique em Reanalisar antes de gerar.")
    if caption_source != "Sem legenda" and not (reviewed_text or "").strip():
        raise gr.Error("Preencha o texto ou selecione 'Sem legenda'.")

    if transition_mode == "Detectar automaticamente":
        if analysis_state.get("has_continuation"):
            effective_mode = "Informar o segundo manualmente"
            effective_seconds = float(analysis_state["transition"])
        else:
            effective_mode = "Sem vídeo de continuação"
            effective_seconds = None
    else:
        effective_mode = transition_mode
        effective_seconds = manual_transition_seconds

    try:
        result = process_video(
            photo_path=intro_media_path, video_path=video_path,
            transition_mode=effective_mode, manual_transition_seconds=effective_seconds,
            caption_mode="Sem legenda" if caption_source == "Sem legenda" else "Usar um texto fixo",
            manual_caption_text=reviewed_text or "", caption_position=caption_position,
            caption_font_percent=float(caption_font_percent), continuation_fit_mode=continuation_fit_mode,
            language=_language_code(language_label), caption_events_override=list(caption_timings or []),
            progress=lambda value, description: progress(value, desc=description),
            quality_profile=quality_profile,
            audio_recovery_path=audio_recovery_path,
        )
    except MediaError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(f"Erro durante a geração: {exc}") from exc
    # Um player + um botão de download, sem player HTML paralelo.
    return result.output_path, result.output_path, result.report


def reset_project():
    return (None, "", "", None, "Envie o vídeo original para começar.", "", [], None,
            None, None, "", None, None, "", QUALITY_STANDARD, "Copiar texto escrito no vídeo",
            "Detectar automaticamente", gr.update(value=None, visible=False), False, None)


with gr.Blocks(title="Studio Reels | Editor de vídeo", analytics_enabled=False) as demo:
    gr.HTML("""
    <header id="studio-header">
      <div class="eyebrow">STUDIO REELS · VERSÃO 2.0.3</div>
      <h1>Crie seu vídeo em poucos passos.</h1>
      <p>Importe o original, escolha a nova mídia, revise o texto e exporte.<br>
         Sem configurações complicadas nem renderizações desnecessárias.</p>
      <span class="status-pill">● Prévia rápida &nbsp; · &nbsp; Exportação otimizada</span>
    </header>
    """)

    video_path_state = gr.State(None)
    analysis_state = gr.State(None)
    caption_timings_state = gr.State([])

    with gr.Row(equal_height=False):
        with gr.Column(scale=6, min_width=330):
            with gr.Group(elem_classes=["studio-panel"]):
                gr.HTML('<div class="studio-section"><span class="studio-index">01</span> Vídeo original</div>'
                        '<p class="studio-sub">Escolha uma origem. O sistema analisa automaticamente após receber o arquivo.</p>')
                with gr.Tabs(elem_id="source-tabs"):
                    with gr.Tab("Importar link"):
                        video_url = gr.Textbox(label="Link do Instagram", placeholder="https://www.instagram.com/reel/...", lines=1)
                        import_button = gr.Button("Importar e analisar", variant="primary", elem_id="import-btn")
                        permitir_sem_audio = gr.Checkbox(
                            label="Usar uma faixa de áudio separada se o Instagram não fornecer som",
                            value=False,
                            info="Ative somente se a importação comum falhou sem áudio. Será obrigatório anexar o áudio antes da exportação.",
                        )
                    with gr.Tab("Enviar arquivo"):
                        local_video = gr.File(label="Arquivo MP4 ou outro vídeo", file_types=["video"], type="filepath")
                import_status = gr.Markdown("Envie o vídeo original para começar.")
                with gr.Accordion("Recuperar áudio — caso o Instagram importe sem som", open=False):
                    audio_recovery = gr.File(
                        label="Faixa de áudio autorizada (MP3, M4A, WAV ou MP4 com som)",
                        file_types=["audio", "video"], type="filepath",
                    )
                    gr.Markdown(
                        "O áudio deve cobrir toda a duração do Reel e começa no segundo 0. "
                        "Se o arquivo original já tiver som, ele será preservado e este anexo será ignorado."
                    )

            with gr.Group(elem_classes=["studio-panel"]):
                gr.HTML('<div class="studio-section"><span class="studio-index">02</span> Personalizar</div>'
                        '<p class="studio-sub">Sua nova foto ou vídeo entra no primeiro take.</p>')
                intro_media = gr.File(label="Nova foto ou vídeo", file_types=["image", "video"], type="filepath")
                caption_source = gr.Dropdown(
                    choices=["Copiar texto escrito no vídeo", "Transcrever o áudio", "Digitar manualmente", "Sem legenda"],
                    value="Copiar texto escrito no vídeo", label="Como obter a legenda?",
                )
                analysis_result = gr.Markdown("A análise aparecerá aqui.")
                reviewed_text = gr.Textbox(label="Texto do primeiro take", placeholder="Revise ou escreva sua frase...", lines=3)
                with gr.Row():
                    caption_size = gr.Slider(2.0, 12.0, value=4.6, step=0.1, label="Tamanho do texto")
                    caption_position = gr.Dropdown(
                        ["Centro", "Centro inferior", "Centro superior"], value="Centro", label="Posição",
                    )
                analyze_button = gr.Button("↻ Reanalisar original", variant="secondary", elem_id="analyze-btn")

            with gr.Accordion("Opções avançadas", open=False):
                transition_mode = gr.Dropdown(
                    ["Detectar automaticamente", "Informar o segundo manualmente", "Sem vídeo de continuação"],
                    value="Detectar automaticamente", label="Troca dos takes",
                )
                manual_transition = gr.Number(label="Segundo da troca", minimum=0.01, visible=False)
                continuation_fit = gr.Dropdown(
                    ["Manter inteiro com fundo desfocado", "Barras pretas", "Preencher a tela (pode cortar bordas)"],
                    value="Manter inteiro com fundo desfocado", label="Como enquadrar a continuação?",
                )
                language = gr.Dropdown(
                    ["Português", "Detectar automaticamente", "Inglês", "Espanhol"],
                    value="Português", label="Idioma do reconhecimento",
                )
                quality_profile = gr.Radio(
                    [QUALITY_STANDARD, QUALITY_ECONOMY], value=QUALITY_STANDARD,
                    label="Qualidade da exportação", info="Econômico: até 720×1280, no máximo 30 FPS.",
                )

            with gr.Group(elem_classes=["studio-panel"]):
                gr.HTML('<div class="studio-section"><span class="studio-index">03</span> Exportar</div>'
                        '<p class="studio-sub">A prévia usa somente uma imagem. A renderização completa começa apenas no botão abaixo.</p>')
                generate_button = gr.Button("Gerar vídeo final", variant="primary", elem_id="generate-btn")
                reset_button = gr.Button("Limpar projeto / começar outro", variant="secondary")

        with gr.Column(scale=5, min_width=310, elem_id="preview-panel"):
            with gr.Group(elem_classes=["studio-panel"]):
                gr.HTML('<div class="studio-section">Pré-visualização</div>'
                        '<p class="studio-sub">Atualiza quando você troca a mídia ou ajusta o texto, sem renderizar o MP4.</p>')
                preview_output = gr.Image(label="Como ficará o primeiro take", interactive=False, height=390)
                preview_status = gr.Markdown("Adicione sua nova mídia para ver a prévia.")
                preview_button = gr.Button("Atualizar prévia", variant="secondary")
            with gr.Group(elem_classes=["studio-panel"], elem_id="result-panel"):
                gr.HTML('<div class="studio-section">Seu vídeo final</div>')
                output_video = gr.Video(label="Assista ao resultado", format="mp4", interactive=False, buttons=["download"])
                download_output = gr.File(label="Baixar MP4", interactive=False, elem_id="download-output")
                report = gr.Markdown()

    # Importação analisa uma vez; editar legenda e fazer prévia não rodam OCR/Whisper.
    analysis_outputs = [analysis_result, reviewed_text, caption_timings_state,
                        analysis_state, transition_mode, manual_transition]
    analysis_inputs = [video_path_state, caption_source, language, transition_mode, manual_transition]

    import_button.click(handle_url_import, [video_url, permitir_sem_audio], [import_status, video_path_state,
                                                         transition_mode, manual_transition],
                        api_visibility="private").success(auto_analyze, analysis_inputs, analysis_outputs,
                                                          api_visibility="private")
    local_video.change(handle_local_video, local_video, [import_status, video_path_state,
                                                         transition_mode, manual_transition],
                       api_visibility="private").success(auto_analyze, analysis_inputs, analysis_outputs,
                                                         api_visibility="private")
    analyze_button.click(analyze_source, analysis_inputs, analysis_outputs, api_visibility="private")
    caption_source.change(caption_review_visibility, caption_source,
                          [reviewed_text, analysis_state, analysis_result], api_visibility="private")
    transition_mode.change(transition_visibility, transition_mode, manual_transition,
                           api_visibility="private")

    preview_inputs = [intro_media, analysis_state, reviewed_text, caption_source,
                      caption_position, caption_size, quality_profile]
    for component in (intro_media, reviewed_text, caption_position, caption_size, quality_profile):
        component.change(preview_image, preview_inputs, [preview_output, preview_status],
                         trigger_mode="always_last", show_progress="hidden", api_visibility="private",
                         concurrency_limit=1, concurrency_id="preview")
    analysis_state.change(preview_image, preview_inputs, [preview_output, preview_status],
                          show_progress="hidden", api_visibility="private",
                          concurrency_limit=1, concurrency_id="preview")
    preview_button.click(preview_image, preview_inputs, [preview_output, preview_status],
                         show_progress="minimal", api_visibility="private", concurrency_limit=1, concurrency_id="preview")
    generate_button.click(
        generate_final,
        inputs=[intro_media, video_path_state, analysis_state, caption_timings_state,
                reviewed_text, caption_source, transition_mode, manual_transition,
                continuation_fit, caption_position, caption_size, language, quality_profile, audio_recovery],
        outputs=[output_video, download_output, report],
        api_visibility="private", concurrency_limit=1, concurrency_id="render",
    )
    reset_button.click(
        reset_project, inputs=[],
        outputs=[local_video, video_url, import_status, video_path_state, analysis_result,
                 reviewed_text, caption_timings_state, analysis_state, intro_media,
                 preview_output, preview_status, output_video, download_output,
                 report, quality_profile, caption_source, transition_mode, manual_transition,
                 permitir_sem_audio, audio_recovery],
        api_visibility="private",
    )

# Uma tarefa pesada por vez. Não é um processo separado nem mantém rede ativa.
demo.queue(max_size=3, default_concurrency_limit=1)
fastapi_app = FastAPI(title=APP_NAME)


@fastapi_app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@fastapi_app.get("/api/info")
def api_info():
    return {"name": "Studio Reels 2.0.3", "status": "online", "preview": True,
            "url_import": True, "api_import_enabled": bool(os.getenv("IMPORT_API_TOKEN", ""))}


def _authorized_api(auth_header: str | None) -> bool:
    configured = os.getenv("IMPORT_API_TOKEN", "").strip()
    if not configured:
        return False  # desativada por padrão
    provided = (auth_header or "").removeprefix("Bearer ").strip()
    return secrets.compare_digest(configured, provided)


@fastapi_app.post("/api/import-video")
def api_import_video(data: dict, authorization: str | None = Header(default=None)):
    if not _authorized_api(authorization):
        return JSONResponse({"erro": "API desativada ou token inválido."}, status_code=401)
    url = str(data.get("url", "")).strip()
    if not url:
        return JSONResponse({"erro": "URL não fornecida"}, status_code=400)
    try:
        caminho = importar_video_url(url)
        return {"caminho": caminho}
    except gr.Error as exc:
        return JSONResponse({"erro": str(exc)}, status_code=400)


@fastapi_app.get("/midia/{job_id}.mp4")
def get_job_video(job_id: str, authorization: str | None = Header(default=None)):
    # Compatibilidade da rota antiga, mas agora fechada por padrão.
    if not _authorized_api(authorization):
        return JSONResponse({"erro": "Não autorizado."}, status_code=401)
    if not _JOB_ID_RE.fullmatch(job_id):
        return JSONResponse({"erro": "Identificador inválido."}, status_code=400)
    path = WORK_ROOT / job_id / "video_pronto.mp4"
    if not path.is_file():
        return JSONResponse({"erro": "Vídeo não encontrado ou expirado."}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename="video_pronto.mp4")


username = os.getenv("APP_USERNAME", "").strip()
password = os.getenv("APP_PASSWORD", "").strip()
auth = (username, password) if username and password else None

app = gr.mount_gradio_app(
    fastapi_app, demo, path="/", allowed_paths=[str(WORK_ROOT.resolve())],
    max_file_size=os.getenv("MAX_UPLOAD_SIZE", "200mb"),
    auth=auth, show_error=True, css=CSS,
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate"),
    enable_monitoring=False,
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")
