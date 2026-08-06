#!/usr/bin/env python3
"""
Importação genérica de vídeos por URL usando yt-dlp.

Aceita URLs de Instagram, TikTok, YouTube, Twitter/X e outros sites.
Implementa fallback com cookies para casos onde o servidor está bloqueado.

Variáveis aceitas:
- COOKIES_B64: conteúdo de cookies.txt codificado em Base64
- COOKIES: conteúdo bruto do cookies.txt no formato Netscape
- FORCE_IPV4: "1" para forçar IPv4
"""

from __future__ import annotations

import base64
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp


class VideoImportError(RuntimeError):
    """Erro esperado e seguro para mostrar ao usuário."""


def normalizar_url(url: str) -> str:
    """Valida e normaliza a URL."""
    url = (url or "").strip()
    if not url:
        raise VideoImportError("Cole o link do vídeo.")

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise VideoImportError("Link inválido.") from exc

    if parsed.scheme not in {"http", "https"}:
        raise VideoImportError("Use um link válido (http ou https).")

    host = (parsed.hostname or "").lower()
    if not host:
        raise VideoImportError("Use um link válido.")

    return url


def _criar_arquivo_cookies_temporario() -> str | None:
    """Cria cookies.txt temporário a partir das variáveis de ambiente."""
    conteudo: str | None = None

    cookies_b64 = os.environ.get("COOKIES_B64", "").strip()
    cookies_raw = os.environ.get("COOKIES", "").strip()

    if cookies_b64:
        try:
            conteudo = base64.b64decode(cookies_b64, validate=True).decode("utf-8")
        except Exception as exc:
            raise VideoImportError(
                "A variável COOKIES_B64 está inválida. Gere novamente o Base64 do cookies.txt."
            ) from exc
    elif cookies_raw:
        conteudo = cookies_raw.replace("\\n", "\n")

    if not conteudo:
        return None

    if "# Netscape HTTP Cookie File" not in conteudo and "\t." not in conteudo:
        raise VideoImportError(
            "Os cookies não parecem estar no formato Netscape cookies.txt."
        )

    arquivo = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix="_cookies.txt",
        delete=False,
    )
    try:
        arquivo.write(conteudo)
        if not conteudo.endswith("\n"):
            arquivo.write("\n")
        return arquivo.name
    finally:
        arquivo.close()


def _erro_pede_autenticacao(mensagem: str) -> bool:
    """Detecta se o erro indica bloqueio/autenticação necessária."""
    texto = mensagem.lower()
    sinais = (
        "empty media response",
        "login",
        "log in",
        "cookies",
        "authentication",
        "main webpage is locked",
        "requested content is not available",
        "not available in your country",
        "access denied",
        "forbidden",
    )
    return any(sinal in texto for sinal in sinais)


def _opcoes_yt_dlp(
    template: str,
    limite_bytes: int,
    limite_mb: int,
    cookiefile: str | None,
) -> dict:
    """Configura opções do yt-dlp."""
    
    def rejeitar_grande(info, *, incomplete=False):
        del incomplete
        tamanho = info.get("filesize") or info.get("filesize_approx")
        if tamanho and tamanho > limite_bytes:
            return f"O vídeo ultrapassa o limite de {limite_mb} MB."
        return None

    opcoes = {
        "outtmpl": template,
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "socket_timeout": 35,
        "retries": 3,
        "fragment_retries": 3,
        "extractor_retries": 2,
        "match_filter": rejeitar_grande,
        "restrictfilenames": True,
        "cachedir": False,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/150.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    }

    if cookiefile:
        opcoes["cookiefile"] = cookiefile

    if os.environ.get("FORCE_IPV4", "").strip() == "1":
        opcoes["source_address"] = "0.0.0.0"

    return opcoes


def _executar_download(
    url: str,
    template: str,
    limite_bytes: int,
    limite_mb: int,
    cookiefile: str | None,
) -> dict:
    """Executa o download com yt-dlp."""
    opcoes = _opcoes_yt_dlp(
        template=template,
        limite_bytes=limite_bytes,
        limite_mb=limite_mb,
        cookiefile=cookiefile,
    )
    with yt_dlp.YoutubeDL(opcoes) as ydl:
        return ydl.extract_info(url, download=True)


def baixar_video(
    url: str,
    pasta_destino: str,
    identificador: str,
    limite_mb: int = 200,
) -> tuple[str, str]:
    """
    Baixa um vídeo por URL e retorna (caminho_do_arquivo, nome_para_download).
    
    Suporta: Instagram, TikTok, YouTube, Twitter/X e muitos outros.
    Implementa fallback com cookies para IPs de servidor bloqueados.
    """
    url_limpa = normalizar_url(url)
    pasta = Path(pasta_destino)
    pasta.mkdir(parents=True, exist_ok=True)

    prefixo = pasta / f"{identificador}_video"
    template = str(prefixo) + ".%(ext)s"
    limite_bytes = int(limite_mb) * 1024 * 1024
    cookiefile: str | None = None
    info: dict | None = None

    try:
        # Primeiro tenta anonimamente
        try:
            info = _executar_download(
                url_limpa, template, limite_bytes, limite_mb, cookiefile=None
            )
        except yt_dlp.utils.DownloadError as primeiro_erro:
            mensagem = str(primeiro_erro)
            
            if "limite" in mensagem.lower() or "too large" in mensagem.lower():
                raise VideoImportError(
                    f"O vídeo ultrapassa o limite de {limite_mb} MB."
                ) from primeiro_erro

            if not _erro_pede_autenticacao(mensagem):
                raise VideoImportError(
                    "Não foi possível importar esse vídeo. Confirme o link e tente novamente."
                ) from primeiro_erro

            # Servidor bloqueado. Tenta com cookies.
            cookiefile = _criar_arquivo_cookies_temporario()
            if not cookiefile:
                raise VideoImportError(
                    "O servidor bloqueou a importação. Configure a variável COOKIES_B64 "
                    "e tente novamente."
                ) from primeiro_erro

            try:
                info = _executar_download(
                    url_limpa, template, limite_bytes, limite_mb, cookiefile=cookiefile
                )
            except yt_dlp.utils.DownloadError as segundo_erro:
                mensagem2 = str(segundo_erro)
                if "limite" in mensagem2.lower() or "too large" in mensagem2.lower():
                    raise VideoImportError(
                        f"O vídeo ultrapassa o limite de {limite_mb} MB."
                    ) from segundo_erro
                if _erro_pede_autenticacao(mensagem2):
                    raise VideoImportError(
                        "Os cookies expiraram ou foram recusados. "
                        "Exporte um cookies.txt novo e atualize COOKIES_B64."
                    ) from segundo_erro
                raise VideoImportError(
                    "O servidor recusou esse download mesmo com autenticação. "
                    "Tente novamente mais tarde."
                ) from segundo_erro

        titulo = ((info or {}).get("title") or (info or {}).get("id") or "video").strip()

    finally:
        if cookiefile:
            try:
                os.remove(cookiefile)
            except OSError:
                pass

    # Procura o arquivo MP4 baixado
    candidatos = sorted(pasta.glob(f"{identificador}_video.*"))
    arquivo = next((p for p in candidatos if p.suffix.lower() == ".mp4"), None)
    if arquivo is None:
        arquivo = next(
            (
                p
                for p in candidatos
                if p.is_file() and p.suffix.lower() not in {".part", ".ytdl"}
            ),
            None,
        )

    if arquivo is None or not arquivo.exists():
        raise VideoImportError(
            "O download terminou, mas o arquivo de vídeo não foi encontrado."
        )

    if arquivo.stat().st_size > limite_bytes:
        try:
            arquivo.unlink()
        except OSError:
            pass
        raise VideoImportError(
            f"O vídeo ultrapassa o limite de {limite_mb} MB."
        )

    # Gera nome para download (sanitizado)
    nome = re.sub(r"[^A-Za-z0-9._-]+", "_", titulo).strip("._")[:80]
    nome = nome or "video_importado"
    return str(arquivo), f"{nome}.mp4"
