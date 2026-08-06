from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from core.config import APP_NAME, MAX_INSTAGRAM_DOWNLOAD_MB, MAX_VIDEO_MINUTES, WORK_ROOT
from core.instagram_import import InstagramImportError, download_instagram_video
from core.media import MediaError, probe_video
from core.processor import cleanup_old_jobs, process_video
from core.transition import detect_intro_end

cleanup_old_jobs()

CSS = """
.gradio-container {max-width: 1180px !important;}
.hero {padding: 18px 20px; border: 1px solid var(--border-color-primary); border-radius: 18px;}
.hero h1 {margin-bottom: 6px !important;}
.small-note {font-size: 0.92rem; opacity: 0.82;}
#generate-btn {min-height: 48px; font-weight: 700;}
#import-instagram-btn {min-height: 44px; font-weight: 700;}
.import-box {padding: 14px; border: 1px solid var(--border-color-primary); border-radius: 14px;}
"""


def transition_visibility(choice: str):
    return gr.update(visible=choice == "Informar o segundo manualmente")


def caption_visibility(choice: str):
    return gr.update(visible=choice == "Usar um texto fixo")



def import_instagram_video(url: str, progress=gr.Progress()):
    clean_url = (url or "").strip()
    if not clean_url:
        raise gr.Error("Cole o link do Reels do Instagram.")

    cleanup_old_jobs()
    import_id = uuid.uuid4().hex
    import_dir = Path(WORK_ROOT) / f"instagram_{import_id}"
    import_dir.mkdir(parents=True, exist_ok=True)

    try:
        progress(0.02, desc="Preparando a importação...")
        video_path, display_name = download_instagram_video(
            clean_url,
            import_dir,
            import_id,
            max_mb=MAX_INSTAGRAM_DOWNLOAD_MB,
            progress_callback=lambda value, description: progress(
                0.05 + 0.72 * max(0.0, min(1.0, value)),
                desc=description,
            ),
        )

        progress(0.82, desc="Conferindo vídeo e áudio...")
        info = probe_video(video_path)
        if info.duration <= 0:
            raise MediaError("A duração do vídeo importado é inválida.")
        if info.duration > MAX_VIDEO_MINUTES * 60:
            raise MediaError(
                f"O vídeo importado tem {info.duration / 60:.1f} minutos. "
                f"O limite configurado é {MAX_VIDEO_MINUTES:.0f} minutos."
            )

        progress(1.0, desc="Reels importado e pronto para editar.")
        status = (
            "✅ **Reels importado com sucesso.**  \n"
            f"Arquivo: `{display_name}`  \n"
            f"Duração: {info.duration:.2f}s · {info.width}×{info.height} · "
            f"Áudio: {'sim' if info.has_audio else 'não'}  \n"
            "O vídeo já foi colocado no campo 2. Agora você pode analisar ou gerar diretamente."
        )
        return video_path, status, ""
    except (InstagramImportError, MediaError) as exc:
        shutil.rmtree(import_dir, ignore_errors=True)
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        shutil.rmtree(import_dir, ignore_errors=True)
        raise gr.Error(f"Falha inesperada ao importar o Reels: {exc}") from exc


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
            # O nome photo_path é mantido internamente para não quebrar integrações antigas,
            # mas agora esse campo aceita tanto imagem quanto vídeo.
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
            video = gr.File(
                label="2. Vídeo original com áudio e possível continuação",
                file_types=["video"],
                type="filepath",
            )
            with gr.Group(elem_classes=["import-box"]):
                gr.Markdown(
                    "**Ou importe o vídeo direto pelo link do Instagram**",
                    elem_classes=["small-note"],
                )
                instagram_url = gr.Textbox(
                    label="Link do Reels",
                    placeholder="https://www.instagram.com/reel/XXXXXXXXXXX/",
                    lines=1,
                )
                import_instagram_button = gr.Button(
                    "IMPORTAR VÍDEO PELO LINK",
                    variant="secondary",
                    elem_id="import-instagram-btn",
                )
                import_status = gr.Markdown()
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
                lines=3,
                visible=False,
            )

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
        output_video = gr.Video(label="Vídeo pronto")
        with gr.Column():
            report = gr.Markdown()
            transition_used = gr.Number(label="Segundo da transição usado", interactive=False)

    transition_mode.change(
        fn=transition_visibility,
        inputs=transition_mode,
        outputs=manual_transition,
    )
    caption_mode.change(
        fn=caption_visibility,
        inputs=caption_mode,
        outputs=manual_caption,
    )
    import_instagram_button.click(
        fn=import_instagram_video,
        inputs=instagram_url,
        outputs=[video, import_status, instagram_url],
        api_name="import_instagram",
    )
    analyze_button.click(
        fn=analyze_video,
        inputs=video,
        outputs=[analysis_result, manual_transition],
    )
    generate_button.click(
        fn=generate_video,
        inputs=[
            intro_media,
            video,
            transition_mode,
            manual_transition,
            caption_mode,
            manual_caption,
            caption_position,
            caption_size,
            continuation_fit,
            language,
        ],
        outputs=[output_video, report, transition_used],
        api_name="generate",
    )


demo.queue(max_size=8, default_concurrency_limit=1)

fastapi_app = FastAPI(title=APP_NAME)


@fastapi_app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@fastapi_app.get("/api/info")
def api_info():
    return {
        "name": APP_NAME,
        "work_root": str(WORK_ROOT),
        "status": "online",
        "intro_media": ["image", "video"],
        "emoji_captions": True,
        "instagram_import": True,
    }


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
