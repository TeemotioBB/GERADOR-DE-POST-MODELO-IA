from __future__ import annotations

import os
import uuid
import json
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from core.config import APP_NAME, WORK_ROOT
from core.captions import read_burned_caption
from core.media import MediaError, probe_video
from core.processor import cleanup_old_jobs, process_video
from core.transition import detect_intro_end
import url_import

cleanup_old_jobs()

CSS = """
.gradio-container {max-width: 1180px !important;}
.hero {padding: 18px 20px; border: 1px solid var(--border-color-primary); border-radius: 18px;}
.hero h1 {margin-bottom: 6px !important;}
.small-note {font-size: 0.92rem; opacity: 0.82;}
#generate-btn {min-height: 48px; font-weight: 700;}
"""

# Cache de vídeos importados por URL (para evitar re-análise)
IMPORTED_VIDEOS = {}
GENERATED_VIDEOS = {}


def transition_visibility(choice: str):
    return gr.update(visible=choice == "Informar o segundo manualmente")


def caption_visibility(choice: str):
    is_copy = choice == "Copiar o texto escrito no vídeo original"
    is_manual = choice == "Usar um texto fixo"
    visible = is_copy or is_manual
    if is_copy:
        label = "Texto detectado — revise antes de gerar (emojis podem ser adicionados aqui)"
        placeholder = "Clique em LER TEXTO DO VÍDEO, confira a frase e corrija/adicione emojis se necessário."
    else:
        label = "Texto da legenda (emojis permitidos 😍🔥✨)"
        placeholder = "Digite a frase com texto e emojis..."
    return (
        gr.update(visible=visible, label=label, placeholder=placeholder),
        gr.update(visible=is_copy),
        gr.update(visible=is_copy),
    )


def video_input_visibility(choice: str):
    """Mostra/esconde os campos de upload/URL baseado na escolha."""
    if choice == "Carregar arquivo local":
        return gr.update(visible=True), gr.update(visible=False)
    else:  # "Importar por URL"
        return gr.update(visible=False), gr.update(visible=True)


def analyze_video(video_path: str | None):
    if not video_path:
        raise gr.Error("Envie o vídeo original primeiro.")
    try:
        info = probe_video(video_path)
        detected = detect_intro_end(video_path)
    except MediaError as exc:
        raise gr.Error(str(exc)) from exc

    if detected.seconds is None:
        text = (
            "### Análise do vídeo original\n"
            f"- Duração: {info.duration:.2f}s\n"
            f"- Tamanho original: {info.width}×{info.height}\n"
            f"- FPS: {info.fps:.2f}\n"
            f"- Áudio: {'sim' if info.has_audio else 'não'}\n"
            f"- Resultado: {detected.message}\n\n"
            "Você pode escolher **Sem vídeo de continuação** ou informar o segundo manualmente."
        )
        return text, None

    text = (
        "### Análise do vídeo original\n"
        f"- Duração: {info.duration:.2f}s\n"
        f"- Tamanho original: {info.width}×{info.height}\n"
        f"- FPS: {info.fps:.2f}\n"
        f"- Áudio: {'sim' if info.has_audio else 'não'}\n"
        f"- Troca provável do take: **{detected.seconds:.2f}s**\n"
        f"- Confiança aproximada: {detected.confidence:.0%}\n\n"
        "A geração automática usará esse ponto. Se estiver errado, selecione o modo manual."
    )
    return text, detected.seconds


def importar_video_url(url: str):
    """Importa vídeo por URL e retorna o caminho local."""
    if not url or not url.strip():
        raise gr.Error("Cole um link válido.")
    
    job_id = uuid.uuid4().hex
    try:
        caminho, nome = url_import.baixar_video(
            url=url,
            pasta_destino=str(WORK_ROOT),
            identificador=job_id,
            limite_mb=500,  # Ajuste conforme seu limite
        )
        IMPORTED_VIDEOS[job_id] = {"path": caminho, "nome": nome}
        return caminho
    except url_import.VideoImportError as e:
        raise gr.Error(str(e)) from e
    except Exception as e:
        raise gr.Error(f"Falha ao importar: {e}") from e


def read_caption_for_review(
    video_path: str | None,
    transition_mode: str,
    manual_transition_seconds: float | None,
    language_label: str,
):
    if not video_path:
        raise gr.Error("Envie ou importe o vídeo original primeiro.")

    language_map = {
        "Português": "pt",
        "Detectar automaticamente": "",
        "Inglês": "en",
        "Espanhol": "es",
    }

    try:
        info = probe_video(video_path)
        if transition_mode == "Sem vídeo de continuação":
            intro_duration = info.duration
        elif transition_mode == "Informar o segundo manualmente":
            if manual_transition_seconds is None:
                raise MediaError("Informe o segundo da transição antes de ler o texto.")
            intro_duration = float(manual_transition_seconds)
            if intro_duration <= 0 or intro_duration > info.duration:
                raise MediaError(f"A transição precisa ficar entre 0 e {info.duration:.2f}s.")
        else:
            detected = detect_intro_end(video_path)
            intro_duration = detected.seconds if detected.seconds is not None else info.duration

        text, confidence = read_burned_caption(
            video_path,
            intro_duration,
            language=language_map.get(language_label, "pt"),
        )
    except MediaError as exc:
        raise gr.Error(str(exc)) from exc

    if not text:
        raise gr.Error(
            "Não consegui identificar um texto confiável. Você ainda pode digitá-lo manualmente no campo de revisão."
        )

    status = (
        f"✅ Texto detectado com confiança aproximada de **{confidence:.0f}%**. "
        "Confira antes de gerar. Emojis do vídeo precisam ser adicionados nesse campo, pois OCR tradicional não os reconhece com segurança."
    )
    return text, gr.update(value=status, visible=True)


def generate_video(
    intro_media_path: str | None,
    video_path: str | None,
    transition_mode: str,
    manual_transition_seconds: float | None,
    caption_mode: str,
    manual_caption_text: str,
    caption_position: str,
    caption_font_percent: float,
    continuation_fit_mode: str,
    language_label: str,
    progress=gr.Progress(),
):
    language_map = {
        "Português": "pt",
        "Detectar automaticamente": "",
        "Inglês": "en",
        "Espanhol": "es",
    }

    try:
        result = process_video(
            photo_path=intro_media_path or "",
            video_path=video_path or "",
            transition_mode=transition_mode,
            manual_transition_seconds=manual_transition_seconds,
            caption_mode=caption_mode,
            manual_caption_text=manual_caption_text or "",
            caption_position=caption_position,
            caption_font_percent=float(caption_font_percent),
            continuation_fit_mode=continuation_fit_mode,
            language=language_map.get(language_label, "pt"),
            progress=lambda value, description: progress(value, desc=description),
        )
    except MediaError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(f"Erro inesperado: {exc}") from exc

    return result.output_path, result.report, result.transition_seconds


with gr.Blocks(title=APP_NAME) as demo:
    gr.Markdown(
        """
        <div class="hero">
          <h1>Mídia + Vídeo Automático</h1>
          <p>Use uma foto ou um vídeo no primeiro take, recrie a legenda com emojis, mantenha o áudio original e preserve o vídeo de continuação.</p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            intro_media = gr.File(
                label="1. Nova mídia da personagem (foto ou vídeo)",
                file_types=["image", "video"],
                type="filepath",
            )
            
            # === NOVA SEÇÃO: Escolher entre upload local ou URL ===
            video_input_mode = gr.Radio(
                choices=[
                    "Carregar arquivo local",
                    "Importar por URL",
                ],
                value="Carregar arquivo local",
                label="Como adicionar o vídeo original?",
            )
            
            # Arquivo local
            video_upload = gr.File(
                label="2. Vídeo original com áudio e possível continuação",
                file_types=["video"],
                type="filepath",
                visible=True,
            )
            
            # URL
            with gr.Group(visible=False) as video_url_group:
                gr.Markdown("**Cole o link do vídeo** (Instagram, TikTok, YouTube, Twitter/X, etc.)")
                video_url = gr.Textbox(
                    label="URL do vídeo",
                    placeholder="https://www.instagram.com/reel/...",
                    lines=2,
                )
                import_button = gr.Button("📥 Importar vídeo", variant="secondary")
                import_status = gr.Markdown(visible=False)
            
            # Armazena o caminho do vídeo (local ou importado)
            video_path_state = gr.State(value=None)
            
            gr.Markdown(
                "A proporção final segue a primeira mídia. Se ela for um vídeo curto, ele será repetido até a troca do take; se for longo, será cortado no ponto da troca.",
                elem_classes=["small-note"],
            )

            analyze_button = gr.Button("Analisar troca do take", variant="secondary")
            analysis_result = gr.Markdown()
            gr.Markdown(
                "**Dica**: se aparecer um pedacinho da modelo original no resultado, use o modo "
                "manual e informe o segundo exato da troca (o botão acima mostra o segundo detectado).",
                elem_classes=["small-note"],
            )

        with gr.Column(scale=1):
            transition_mode = gr.Radio(
                choices=[
                    "Detectar automaticamente",
                    "Informar o segundo manualmente",
                    "Sem vídeo de continuação",
                ],
                value="Detectar automaticamente",
                label="Onde termina o primeiro take?",
            )
            manual_transition = gr.Number(
                label="Segundo em que começa o vídeo de continuação",
                value=5.0,
                minimum=0.01,
                visible=False,
            )

            continuation_fit = gr.Radio(
                choices=[
                    "Manter inteiro com fundo desfocado",
                    "Barras pretas",
                    "Preencher a tela (pode cortar bordas)",
                ],
                value="Manter inteiro com fundo desfocado",
                label="Como encaixar a continuação no tamanho da primeira mídia?",
            )

            caption_mode = gr.Radio(
                choices=[
                    "Transcrever o áudio automaticamente",
                    "Copiar o texto escrito no vídeo original",
                    "Usar um texto fixo",
                    "Sem legenda",
                ],
                value="Transcrever o áudio automaticamente",
                label="Legenda do primeiro take",
            )
            manual_caption = gr.Textbox(
                label="Texto da legenda (emojis permitidos 😍🔥✨)",
                placeholder="Digite a frase com texto e emojis...",
                lines=4,
                visible=False,
            )
            read_caption_button = gr.Button("🔎 LER TEXTO DO VÍDEO", variant="secondary", visible=False)
            ocr_status = gr.Markdown(visible=False)

            with gr.Row():
                caption_position = gr.Dropdown(
                    choices=["Centro", "Centro inferior", "Centro superior"],
                    value="Centro",
                    label="Posição",
                )
                caption_size = gr.Slider(
                    minimum=2.5,
                    maximum=8.0,
                    value=4.6,
                    step=0.1,
                    label="Tamanho da fonte (% da altura)",
                )

            language = gr.Dropdown(
                choices=["Português", "Detectar automaticamente", "Inglês", "Espanhol"],
                value="Português",
                label="Idioma da transcrição",
            )

    generate_button = gr.Button("GERAR VÍDEO", variant="primary", elem_id="generate-btn")

    with gr.Row(equal_height=False):
        output_video = gr.Video(
            label="Prévia do vídeo pronto",
            format="mp4",
            autoplay=False,
            buttons=["download"],
            height=640,
        )
        with gr.Column():
            download_file = gr.DownloadButton("⬇️ BAIXAR MP4", value=None, variant="primary")
            iphone_action = gr.HTML()
            gr.Markdown(
                "No iPhone, o download comum do Safari costuma ir para **Arquivos**. "
                "Use **Abrir vídeo no iPhone** e depois Compartilhar → **Salvar Vídeo** para mandar à galeria.",
                elem_classes=["small-note"],
            )
            report = gr.Markdown()
            transition_used = gr.Number(label="Segundo da transição usado", interactive=False)

    # === LÓGICA DE EVENTOS ===
    
    # Ao mudar modo de entrada, mostra/esconde campos
    video_input_mode.change(
        fn=video_input_visibility,
        inputs=video_input_mode,
        outputs=[video_upload, video_url_group],
    )
    
    # Quando arquivo local é selecionado, armazena no state
    video_upload.change(
        fn=lambda x: x,
        inputs=video_upload,
        outputs=video_path_state,
    )
    
    # Quando URL é importada
    def handle_url_import(url):
        caminho = importar_video_url(url)
        return (
            gr.update(value="✅ Vídeo importado com sucesso!", visible=True),
            caminho,
        )
    
    import_button.click(
        fn=handle_url_import,
        inputs=video_url,
        outputs=[import_status, video_path_state],
    )
    
    # Análise do vídeo
    def analyze_wrapper(video_path):
        return analyze_video(video_path)

    analyze_button.click(
        fn=analyze_wrapper,
        inputs=video_path_state,
        outputs=[analysis_result, manual_transition],
    )

    # Pré-leitura da legenda escrita, para o usuário corrigir antes de renderizar.
    read_caption_button.click(
        fn=read_caption_for_review,
        inputs=[video_path_state, transition_mode, manual_transition, language],
        outputs=[manual_caption, ocr_status],
    )

    # Geração
    def generate_wrapper(
        intro_media_path,
        video_path,
        transition_mode,
        manual_transition_seconds,
        caption_mode,
        manual_caption_text,
        caption_position,
        caption_font_percent,
        continuation_fit_mode,
        language_label,
        progress=gr.Progress(),
    ):
        output_path, result_report, transition_seconds = generate_video(
            intro_media_path,
            video_path,
            transition_mode,
            manual_transition_seconds,
            caption_mode,
            manual_caption_text,
            caption_position,
            caption_font_percent,
            continuation_fit_mode,
            language_label,
            progress,
        )
        token = uuid.uuid4().hex
        GENERATED_VIDEOS[token] = output_path
        iphone_html = (
            f'<a href="/iphone-video/{token}" target="_blank" '
            'style="display:block;text-align:center;padding:13px 16px;border-radius:10px;'
            'font-weight:700;text-decoration:none;border:1px solid currentColor;margin-top:10px;">'
            '🍎 ABRIR VÍDEO NO IPHONE / SALVAR EM FOTOS</a>'
        )
        return output_path, output_path, iphone_html, result_report, transition_seconds

    generate_button.click(
        fn=generate_wrapper,
        inputs=[
            intro_media,
            video_path_state,
            transition_mode,
            manual_transition,
            caption_mode,
            manual_caption,
            caption_position,
            caption_size,
            continuation_fit,
            language,
        ],
        outputs=[output_video, download_file, iphone_action, report, transition_used],
        api_name="generate",
    )

    # Outros eventos
    transition_mode.change(
        fn=transition_visibility,
        inputs=transition_mode,
        outputs=manual_transition,
    )
    caption_mode.change(
        fn=caption_visibility,
        inputs=caption_mode,
        outputs=[manual_caption, read_caption_button, ocr_status],
    )


demo.queue(max_size=8, default_concurrency_limit=1)

fastapi_app = FastAPI(title=APP_NAME)


@fastapi_app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@fastapi_app.get("/iphone-video/{token}")
def iphone_video(token: str):
    path_value = GENERATED_VIDEOS.get(token)
    if not path_value:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado ou expirado.")

    path = Path(path_value).resolve()
    work_root = Path(WORK_ROOT).resolve()
    if work_root not in path.parents or not path.exists() or not path.is_file():
        GENERATED_VIDEOS.pop(token, None)
        raise HTTPException(status_code=404, detail="Vídeo não encontrado ou expirado.")

    return FileResponse(
        path=str(path),
        media_type="video/mp4",
        filename="video_pronto.mp4",
        content_disposition_type="inline",
        headers={"Cache-Control": "no-store"},
    )


@fastapi_app.get("/api/info")
def api_info():
    return {
        "name": APP_NAME,
        "work_root": str(WORK_ROOT),
        "status": "online",
        "intro_media": ["image", "video"],
        "emoji_captions": True,
        "url_import": True,
    }


@fastapi_app.post("/api/import-video")
def api_import_video(data: dict):
    """API para importar vídeo por URL (para integração com outros apps)."""
    url = (data.get("url") or "").strip()
    if not url:
        return {"erro": "URL não fornecida"}, 400
    
    job_id = uuid.uuid4().hex
    try:
        caminho, nome = url_import.baixar_video(
            url=url,
            pasta_destino=str(WORK_ROOT),
            identificador=job_id,
            limite_mb=500,
        )
        IMPORTED_VIDEOS[job_id] = {"path": caminho, "nome": nome}
        return {"id": job_id, "caminho": caminho, "nome": nome}
    except url_import.VideoImportError as e:
        return {"erro": str(e)}, 400
    except Exception as e:
        return {"erro": f"Falha ao importar: {e}"}, 500


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
