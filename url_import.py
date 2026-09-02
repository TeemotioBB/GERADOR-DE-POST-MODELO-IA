"""Camada de compatibilidade para importação por URL.

A implementação real vive em ``core.instagram_import``. Manter apenas um
código de download evita que duas cópias quase idênticas se desalinhem.
"""
from __future__ import annotations

from core.instagram_import import InstagramImportError, baixar_video_instagram

VideoImportError = InstagramImportError


def baixar_video(
    url: str,
    pasta_destino: str,
    identificador: str,
    limite_mb: int = 200,
) -> tuple[str, str]:
    return baixar_video_instagram(
        url=url,
        pasta_destino=pasta_destino,
        identificador=identificador,
        limite_mb=limite_mb,
    )
