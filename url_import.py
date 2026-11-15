"""API pública do importador (compatibilidade com app.py e integrações antigas).

Lógica de download centralizada em core.instagram_import. Evita manter duas
cópias do mesmo código de cookies, validação e tratamento de erros.
"""
from core.instagram_import import (
    InstagramImportError,
    baixar_video_instagram,
    normalizar_url_instagram,
)

VideoImportError = InstagramImportError


def baixar_video(url: str, pasta_destino: str, identificador: str,
                 limite_mb: int = 200) -> tuple[str, str]:
    return baixar_video_instagram(
        url=url, pasta_destino=pasta_destino,
        identificador=identificador, limite_mb=limite_mb,
    )
