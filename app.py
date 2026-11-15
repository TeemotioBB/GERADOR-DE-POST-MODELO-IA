from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path

import cv2
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from core.captions import CaptionEvent, clean_review_text, read_burned_caption, render_caption_overlays, transcribe_intro
from core.config import APP_NAME, MAX_INSTAGRAM_DOWNLOAD_MB, WORK_ROOT
from core.media import MediaError, probe_video
from core.processor import cleanup_old_jobs, process_video
from core.transition import detect_intro_end
import url_import

cleanup_old_jobs()

CSS = """
.gradio-container {max-width: 1120px !important; margin: 0 auto !important;}
.hero-v2 {
  padding: 22px 24px;
  border: 1px solid var(--border-color-primary);
  border-radius: 20px;
  margin-bottom: 14px;
}
.hero-v2 h1 {margin: 0 0 6px 0 !important; font-size: 1.8rem !important;}
.hero-v2 p {margin: 0 !important; opacity: .78;}
.step-card {
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 18px !important;
  padding: 14px !important;
  margin-bottom: 12px !important;
}
.step-title {font-size: 1.02rem; font-weight: 700; margin-bottom: 2px;}
.step-help {font-size: .9rem; opacity: .72; margin-bottom: 8px;}
#analyze-btn, #preview-btn, #generate-btn {min-height: 48px; font-weight: 700;}
#generate-btn {font-size: 1.03rem;}
.status-ok {padding: 10px 12px; border-radius: 12px;}
.small-note {font-size: .88rem; opacity: .75; line-height: 1.45;}
.result-card {margin-top: 10px;}
@media (max-width: 700px) {
  .hero-v2 {padding: 18px 16px;}
  .step-card {padding: 10px !important;}
}
"""

_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _language_code(label: str) -> str:
    return {
        "Português": "pt",
        "Detectar automaticamente": "",
        "Inglês": "en",
        "Espanhol": "es",
    }.get(label, "pt")


def _video_player_html(output_path: str) -> str:
    job_id = Path(output_path).parent.name
    if not _JOB_ID_RE.match(job_id):
        return ""
    url = f"/midia/{job_id}.mp4"
    return f"""
    <div style="margin-top:10px">
      <video controls playsinline preload="metadata" style="width:100%;max-height:680px;border-radius:14px;background:#000" src="{url}"></video>
      <div class="small-note" style="margin-top:8px">
        Se o player acima não abrir no seu celular, <a href="{url}" target="_blank" rel="noopener"><b>abra o vídeo em uma nova aba</b></a>.
      </div>
    </div>
    """


def video_input_visibility(choice: str):
    local = choice == "Enviar arquivo"
    return gr.update(visible=local), gr.update(visible=not local)


def transition_visibility(choice: str):
    return gr.update(visible=choice == "Informar o segundo manualmente")


def caption_review_visibility(choice: str):
    return gr.update(visible=choice != "Sem legenda")


def importar_video_url(url: str) -> str:
    if not (url or "").strip():
        raise gr.Error("Cole um link válido.")
    cleanup_old_jobs()
    job_id = uuid.uuid4().hex
    try:
        caminho, _nome = url_import.baixar_video(
            url=url.strip(),
            pasta_destino=str(WORK_ROOT),
            identificador=job_id,
            limite_mb=MAX_INSTAGRAM_DOWNLOAD_MB,
        )
        return caminho
    except url_import.VideoImportError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(f"Falha ao importar o vídeo: {exc}") from exc


def handle_url_import(url: str):
    caminho = importar_video_url(url)
    info = probe_video(caminho)
    status = f"✅ Vídeo importado • {info.duration:.1f}s • {info.width}×{info.height}"
    return status, caminho


def handle_local_video(path: str | None):
    if not path:
        return "", None
    try:
        info = probe_video(path)
        return f"✅ Vídeo carregado • {info.duration:.1f}s • {info.width}×{info.height}", path
    except MediaError as exc:
        raise gr.Error(str(exc)) from exc


def _analysis_work_dir() -> Path:
    path = WORK_ROOT / f"analysis_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def analyze_source(
    video_path: str | None,
    caption_source: str,
    language_label: str,
    requested_transition_mode: str,
    manual_transition_seconds: float | None,
):
    if not video_path:
        raise gr.Error("Adicione o vídeo original primeiro.")

    cleanup_old_jobs()
    try:
        info = probe_video(video_path)
    except MediaError as exc:
        raise gr.Error(str(exc)) from exc

    # Regra desta operação: o formato normal possui EXATAMENTE dois takes.
    # Portanto uma falha do detector nunca pode virar silenciosamente
    # "sem continuação", pois isso destruiria a estrutura do vídeo.
    if requested_transition_mode == "Informar o segundo manualmente":
        if manual_transition_seconds is None:
            raise gr.Error("Informe o segundo em que começa o take 2.")
        try:
            transition = float(manual_transition_seconds)
        except (TypeError, ValueError) as exc:
            raise gr.Error("O segundo da troca é inválido.") from exc
        if transition <= 0 or transition >= info.duration:
            raise gr.Error(
                f"O segundo da troca deve ser maior que 0 e menor que {info.duration:.2f}s."
            )
        has_continuation = True
        transition_mode = "Informar o segundo manualmente"
        transition_message = f"Troca manual definida em {transition:.2f}s."
    elif requested_transition_mode == "Sem vídeo de continuação":
        transition = info.duration
        has_continuation = False
        transition_mode = "Sem vídeo de continuação"
        transition_message = "Modo excepcional: vídeo tratado sem take 2."
    else:
        try:
            detected = detect_intro_end(video_path)
        except MediaError as exc:
            raise gr.Error(str(exc)) from exc
        if detected.seconds is None:
            raise gr.Error(
                "Não encontrei a troca entre os 2 takes com segurança. "
                "Abra Configurações avançadas, escolha 'Informar o segundo manualmente', "
                "digite o instante em que começa o take 2 e clique em Analisar vídeo novamente."
            )
        transition = min(max(float(detected.seconds), 0.1), info.duration - 0.05)
        has_continuation = True
        transition_mode = "Informar o segundo manualmente"
        transition_message = f"Troca detectada em {transition:.2f}s ({detected.confidence:.0%} de confiança)."

    intro_duration = transition if has_continuation else info.duration
    language = _language_code(language_label)
    events: list[CaptionEvent] = []
    summary = ""

    if caption_source == "Copiar texto escrito no vídeo":
        try:
            events, summary = read_burned_caption(
                video_path,
                intro_duration,
                language=language or "pt",
            )
        except MediaError as exc:
            raise gr.Error(str(exc)) from exc
    elif caption_source == "Transcrever o áudio":
        if not info.has_audio:
            raise gr.Error("O vídeo não possui áudio para transcrever.")
        work_dir = _analysis_work_dir()
        try:
            events, summary = transcribe_intro(
                video_path,
                work_dir,
                intro_duration,
                language=language or None,
            )
        except MediaError as exc:
            raise gr.Error(str(exc)) from exc
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
    elif caption_source in {"Digitar manualmente", "Sem legenda"}:
        events = []
    else:
        raise gr.Error("Escolha uma opção de legenda válida.")

    # A legenda do produto é sempre UMA frase fixa no primeiro take.
    text = clean_review_text(
        " ".join(" ".join(event.text.split()) for event in events if event.text.strip())
    ).strip()
    timings = [{"start": 0.0, "end": intro_duration}] if text else []

    if caption_source == "Copiar texto escrito no vídeo" and not text:
        caption_note = (
            "⚠️ Não encontrei um texto FIXO repetido com confiança. "
            "Preferi não puxar texto aleatório; corrija/digite abaixo."
        )
    elif caption_source == "Transcrever o áudio" and not text:
        caption_note = "⚠️ Não encontrei fala clara. Você pode digitar o texto manualmente."
    elif caption_source == "Sem legenda":
        caption_note = "Sem legenda selecionada."
    elif caption_source == "Digitar manualmente":
        caption_note = "Digite o texto que deseja usar antes de pré-visualizar."
    else:
        caption_note = "✅ Texto fixo detectado por repetição em vários frames. Revise antes de gerar."

    state = {
        "video_path": video_path,
        "duration": info.duration,
        "transition": transition,
        "has_continuation": has_continuation,
        "caption_source": caption_source,
    }

    details = f"### Análise concluída\n**{transition_message}**\n\n{caption_note}"
    if summary and caption_source in {"Copiar texto escrito no vídeo", "Transcrever o áudio"}:
        first_line = summary.splitlines()[0] if summary.splitlines() else ""
        if first_line:
            details += f"\n\n<span style='opacity:.72'>{first_line}</span>"

    return (
        details,
        gr.update(value=text, visible=caption_source != "Sem legenda"),
        timings,
        state,
        gr.update(value=transition_mode),
        gr.update(value=transition if has_continuation else None, visible=has_continuation),
    )


def _load_preview_base(path: str) -> Image.Image:
    source = Path(path)
    try:
        with Image.open(source) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        pass

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise MediaError("Não foi possível abrir a nova mídia para pré-visualização.")
    try:
        ok, frame = cap.read()
        if not ok:
            raise MediaError("Não foi possível ler o primeiro quadro da nova mídia.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    finally:
        cap.release()


def preview_image(
    intro_media_path: str | None,
    reviewed_text: str,
    caption_source: str,
    caption_position: str,
    caption_size: float,
):
    if not intro_media_path:
        raise gr.Error("Envie a nova foto ou vídeo antes de pré-visualizar.")

    try:
        base = _load_preview_base(intro_media_path)
    except MediaError as exc:
        raise gr.Error(str(exc)) from exc

    max_edge = 720
    if max(base.size) > max_edge:
        ratio = max_edge / max(base.size)
        base = base.resize(
            (max(2, int(base.width * ratio)), max(2, int(base.height * ratio))),
            Image.Resampling.LANCZOS,
        )

    preview_dir = WORK_ROOT / f"preview_{uuid.uuid4().hex}"
    preview_dir.mkdir(parents=True, exist_ok=False)
    output_path = preview_dir / "preview.png"

    try:
        final = base.convert("RGBA")
        if caption_source != "Sem legenda" and (reviewed_text or "").strip():
            # A prévia usa a legenda inteira; no vídeo ela fica fixa durante todo o primeiro take.
            fixed_text = reviewed_text.strip()
            overlays = render_caption_overlays(
                [CaptionEvent(0.0, 1.0, fixed_text)],
                preview_dir,
                width=final.width,
                height=final.height,
                font_percent=float(caption_size),
                position=caption_position,
            )
            if overlays:
                with Image.open(overlays[0].path) as overlay:
                    final = Image.alpha_composite(final, overlay.convert("RGBA"))
        final.convert("RGB").save(output_path, format="PNG", optimize=True)
    except Exception as exc:
        shutil.rmtree(preview_dir, ignore_errors=True)
        raise gr.Error(f"Não foi possível criar a prévia: {exc}") from exc

    return str(output_path), "Prévia rápida criada. O vídeo final mantém o áudio e a continuação."


def generate_final(
    intro_media_path: str | None,
    video_path: str | None,
    analysis_state: dict | None,
    caption_timings: list | None,
    reviewed_text: str,
    caption_source: str,
    transition_mode: str,
    manual_transition_seconds: float | None,
    continuation_fit_mode: str,
    caption_position: str,
    caption_font_percent: float,
    language_label: str,
    progress=gr.Progress(),
):
    if not intro_media_path:
        raise gr.Error("Envie a nova foto ou vídeo.")
    if not video_path:
        raise gr.Error("Adicione o vídeo original.")
    if not analysis_state or analysis_state.get("video_path") != video_path:
        raise gr.Error("Clique em 'Analisar vídeo' antes de gerar. Isso evita processamento duplicado.")
    if analysis_state.get("caption_source") != caption_source:
        raise gr.Error("Você mudou o modo de legenda. Clique em 'Analisar vídeo' novamente.")

    if caption_source != "Sem legenda" and not (reviewed_text or "").strip():
        raise gr.Error("Revise ou digite o texto da legenda antes de gerar.")

    language = _language_code(language_label)
    # A análise já calculou a transição. No modo automático da tela, reutilizamos
    # o segundo detectado para não analisar o mesmo vídeo duas vezes.
    if transition_mode == "Detectar automaticamente":
        if analysis_state.get("has_continuation"):
            effective_transition_mode = "Informar o segundo manualmente"
            effective_manual = float(analysis_state["transition"])
        else:
            effective_transition_mode = "Sem vídeo de continuação"
            effective_manual = None
    else:
        effective_transition_mode = transition_mode
        effective_manual = manual_transition_seconds

    if caption_source == "Sem legenda":
        processor_caption_mode = "Sem legenda"
        override = None
    else:
        # O texto já foi analisado e revisado. Não rodamos OCR/Whisper novamente.
        processor_caption_mode = "Usar um texto fixo"
        override = list(caption_timings or [])

    try:
        result = process_video(
            photo_path=intro_media_path,
            video_path=video_path,
            transition_mode=effective_transition_mode,
            manual_transition_seconds=effective_manual,
            caption_mode=processor_caption_mode,
            manual_caption_text=reviewed_text or "",
            caption_position=caption_position,
            caption_font_percent=float(caption_font_percent),
            continuation_fit_mode=continuation_fit_mode,
            language=language,
            caption_events_override=override,
            progress=lambda value, description: progress(value, desc=description),
        )
    except MediaError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(f"Erro inesperado durante a geração: {exc}") from exc

    return result.output_path, result.report, _video_player_html(result.output_path)


with gr.Blocks(title=APP_NAME) as demo:
    gr.HTML(
        """
        <div class="hero-v2">
          <h1>🎬 Mídia + Vídeo Automático</h1>
          <p>Importe o original, revise o texto, veja uma prévia rápida e só então renderize o vídeo final.</p>
        </div>
        """
    )

    video_path_state = gr.State(value=None)
    analysis_state = gr.State(value=None)
    caption_timings_state = gr.State(value=[])

    with gr.Group(elem_classes=["step-card"]):
        gr.HTML('<div class="step-title">1. Vídeo original</div><div class="step-help">Envie o arquivo ou cole o link do Reel/vídeo.</div>')
        input_mode = gr.Radio(
            choices=["Enviar arquivo", "Importar por URL"],
            value="Importar por URL",
            label="Origem",
        )
        local_video = gr.File(
            label="Vídeo original",
            file_types=["video"],
            type="filepath",
            visible=False,
        )
        with gr.Group(visible=True) as url_group:
            with gr.Row():
                video_url = gr.Textbox(
                    label="Link do vídeo",
                    placeholder="https://www.instagram.com/reel/...",
                    scale=4,
                )
                import_button = gr.Button("📥 Importar", variant="secondary", scale=1)
        import_status = gr.Markdown()

    with gr.Group(elem_classes=["step-card"]):
        gr.HTML('<div class="step-title">2. Analisar conteúdo</div><div class="step-help">Detecta a troca entre os 2 takes e procura somente o texto fixo do primeiro take.</div>')
        caption_source = gr.Radio(
            choices=[
                "Copiar texto escrito no vídeo",
                "Transcrever o áudio",
                "Digitar manualmente",
                "Sem legenda",
            ],
            value="Copiar texto escrito no vídeo",
            label="Como obter a legenda?",
        )
        analyze_button = gr.Button("🔎 ANALISAR VÍDEO", variant="primary", elem_id="analyze-btn")
        analysis_result = gr.Markdown()
        reviewed_text = gr.Textbox(
            label="Texto detectado — revise antes de gerar",
            placeholder="O texto aparecerá aqui. Você pode corrigir qualquer palavra ou digitar manualmente.",
            lines=4,
            visible=True,
        )
        with gr.Row():
            caption_size = gr.Slider(
                minimum=2.0,
                maximum=12.0,
                value=4.6,
                step=0.1,
                label="Tamanho do texto no primeiro take",
                info="Ajuste aqui e clique em Pré-visualizar para conferir antes de gerar.",
            )
            caption_position = gr.Dropdown(
                choices=["Centro", "Centro inferior", "Centro superior"],
                value="Centro",
                label="Posição do texto",
            )

    with gr.Group(elem_classes=["step-card"]):
        gr.HTML('<div class="step-title">3. Nova mídia e prévia</div><div class="step-help">A prévia é uma imagem rápida e não gasta uma renderização completa.</div>')
        intro_media = gr.File(
            label="Nova foto ou vídeo do primeiro take",
            file_types=["image", "video"],
            type="filepath",
        )
        preview_button = gr.Button("👁️ PRÉ-VISUALIZAR", variant="secondary", elem_id="preview-btn")
        with gr.Row(equal_height=False):
            preview_output = gr.Image(label="Prévia", interactive=False)
            preview_status = gr.Markdown()

    with gr.Accordion("⚙️ Configurações avançadas", open=False):
        with gr.Row():
            transition_mode = gr.Dropdown(
                choices=[
                    "Detectar automaticamente",
                    "Informar o segundo manualmente",
                    "Sem vídeo de continuação",
                ],
                value="Detectar automaticamente",
                label="Troca do primeiro take",
            )
            manual_transition = gr.Number(
                label="Segundo da troca",
                minimum=0.01,
                visible=False,
            )
        continuation_fit = gr.Dropdown(
            choices=[
                "Manter inteiro com fundo desfocado",
                "Barras pretas",
                "Preencher a tela (pode cortar bordas)",
            ],
            value="Manter inteiro com fundo desfocado",
            label="Encaixe da continuação",
        )
        language = gr.Dropdown(
            choices=["Português", "Detectar automaticamente", "Inglês", "Espanhol"],
            value="Português",
            label="Idioma",
        )

    generate_button = gr.Button("✨ GERAR VÍDEO FINAL", variant="primary", elem_id="generate-btn")

    with gr.Group(elem_classes=["step-card", "result-card"]):
        gr.HTML('<div class="step-title">Resultado</div>')
        output_video = gr.Video(label="Vídeo pronto", format="mp4", interactive=False)
        fallback_player = gr.HTML()
        report = gr.Markdown()

    input_mode.change(
        fn=video_input_visibility,
        inputs=input_mode,
        outputs=[local_video, url_group],
    )
    local_video.change(
        fn=handle_local_video,
        inputs=local_video,
        outputs=[import_status, video_path_state],
    )
    import_button.click(
        fn=handle_url_import,
        inputs=video_url,
        outputs=[import_status, video_path_state],
    )
    caption_source.change(
        fn=caption_review_visibility,
        inputs=caption_source,
        outputs=reviewed_text,
    )
    transition_mode.change(
        fn=transition_visibility,
        inputs=transition_mode,
        outputs=manual_transition,
    )
    analyze_button.click(
        fn=analyze_source,
        inputs=[video_path_state, caption_source, language, transition_mode, manual_transition],
        outputs=[
            analysis_result,
            reviewed_text,
            caption_timings_state,
            analysis_state,
            transition_mode,
            manual_transition,
        ],
    )
    preview_button.click(
        fn=preview_image,
        inputs=[intro_media, reviewed_text, caption_source, caption_position, caption_size],
        outputs=[preview_output, preview_status],
    )
    generate_button.click(
        fn=generate_final,
        inputs=[
            intro_media,
            video_path_state,
            analysis_state,
            caption_timings_state,
            reviewed_text,
            caption_source,
            transition_mode,
            manual_transition,
            continuation_fit,
            caption_position,
            caption_size,
            language,
        ],
        outputs=[output_video, report, fallback_player],
        api_name="generate",
    )


demo.queue(max_size=4, default_concurrency_limit=1)

fastapi_app = FastAPI(title=APP_NAME)


@fastapi_app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@fastapi_app.get("/midia/{job_id}.mp4")
def get_job_video(job_id: str):
    if not _JOB_ID_RE.match(job_id):
        return JSONResponse({"erro": "identificador inválido"}, status_code=400)

    path = (WORK_ROOT / job_id / "video_pronto.mp4").resolve()
    try:
        path.relative_to(Path(WORK_ROOT).resolve())
    except ValueError:
        return JSONResponse({"erro": "caminho inválido"}, status_code=400)

    if not path.exists():
        return JSONResponse(
            {"erro": "Vídeo não encontrado ou já removido pela limpeza automática."},
            status_code=404,
        )

    return FileResponse(
        path,
        media_type="video/mp4",
        filename="video_pronto.mp4",
        content_disposition_type="inline",
    )


@fastapi_app.get("/api/info")
def api_info():
    return {
        "name": APP_NAME,
        "status": "online",
        "preview": True,
        "review_before_render": True,
        "url_import": True,
    }


@fastapi_app.post("/api/import-video")
def api_import_video(data: dict):
    url = (data.get("url") or "").strip()
    if not url:
        return JSONResponse({"erro": "URL não fornecida"}, status_code=400)
    try:
        caminho = importar_video_url(url)
        return {"caminho": caminho}
    except gr.Error as exc:
        return JSONResponse({"erro": str(exc)}, status_code=400)


username = os.getenv("APP_USERNAME", "").strip()
password = os.getenv("APP_PASSWORD", "").strip()
auth = (username, password) if username and password else None

app = gr.mount_gradio_app(
    fastapi_app,
    demo,
    path="/",
    allowed_paths=[str(Path(WORK_ROOT).resolve())],
    max_file_size=os.getenv("MAX_UPLOAD_SIZE", "500mb"),
    auth=auth,
    show_error=True,
    css=CSS,
)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")
