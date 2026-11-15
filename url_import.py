#!/usr/bin/env python3
"""Importação de Reels públicos do Instagram usando yt-dlp.

O Instagram pode devolver uma resposta vazia para IPs de datacenter, mesmo
quando o Reels é público. Este módulo tenta primeiro sem autenticação e, se
necessário, repete usando cookies armazenados em variável do Railway.

Variáveis aceitas:
- INSTAGRAM_COOKIES_B64: conteúdo de cookies.txt codificado em Base64 (recomendado)
- INSTAGRAM_COOKIES: conteúdo bruto do cookies.txt no formato Netscape
- INSTAGRAM_FORCE_IPV4: "1" para forçar IPv4
"""

from __future__ import annotations

import base64
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp


class InstagramImportError(RuntimeError):
    """Erro esperado e seguro para mostrar ao usuário."""


_DOMINIOS = {
    "instagram.com",
    "www.instagram.com",
    "m.instagram.com",
}
_PADRAO_REEL = re.compile(r"^/(?:reel|reels|p)/[A-Za-z0-9_-]+/?$")


def normalizar_url_instagram(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise InstagramImportError("Cole o link do Reels.")

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise InstagramImportError("Link do Instagram inválido.") from exc

    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in _DOMINIOS:
        raise InstagramImportError("Use um link válido do Instagram.")

    caminho = parsed.path.rstrip("/")
    if not _PADRAO_REEL.match(caminho):
        raise InstagramImportError(
            "Esse link não parece ser de um Reels ou publicação do Instagram."
        )

    # Remove ?igsh=, utm e demais parâmetros de rastreamento.
    return f"https://www.instagram.com{caminho}/"


def _criar_arquivo_cookies_temporario() -> str | None:
    """Cria cookies.txt temporário a partir das variáveis do Railway."""
    conteudo: str | None = None

    cookies_b64 = os.environ.get("INSTAGRAM_COOKIES_B64", "").strip()
    cookies_raw = os.environ.get("INSTAGRAM_COOKIES", "").strip()

    if cookies_b64:
        try:
            conteudo = base64.b64decode(cookies_b64, validate=True).decode("utf-8")
        except Exception as exc:
            raise InstagramImportError(
                "A variável INSTAGRAM_COOKIES_B64 está inválida. Gere novamente o Base64 do cookies.txt."
            ) from exc
    elif cookies_raw:
        # Railway pode salvar quebras de linha como os caracteres literais \n.
        conteudo = cookies_raw.replace("\\n", "\n")

    if not conteudo:
        return None

    if "# Netscape HTTP Cookie File" not in conteudo and "\t.instagram.com\t" not in conteudo:
        raise InstagramImportError(
            "Os cookies do Instagram não parecem estar no formato Netscape cookies.txt."
        )

    arquivo = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix="_instagram_cookies.txt",
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
    texto = mensagem.lower()
    sinais = (
        "empty media response",
        "login",
        "log in",
        "cookies",
        "authentication",
        "main webpage is locked behind the login page",
        "requested content is not available",
    )
    return any(sinal in texto for sinal in sinais)


def _opcoes_yt_dlp(
    template: str,
    limite_bytes: int,
    limite_mb: int,
    cookiefile: str | None,
) -> dict:
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

    if os.environ.get("INSTAGRAM_FORCE_IPV4", "").strip() == "1":
        opcoes["source_address"] = "0.0.0.0"

    return opcoes


def _executar_download(
    url: str,
    template: str,
    limite_bytes: int,
    limite_mb: int,
    cookiefile: str | None,
) -> dict:
    opcoes = _opcoes_yt_dlp(
        template=template,
        limite_bytes=limite_bytes,
        limite_mb=limite_mb,
        cookiefile=cookiefile,
    )
    with yt_dlp.YoutubeDL(opcoes) as ydl:
        return ydl.extract_info(url, download=True)


def baixar_video_instagram(
    url: str,
    pasta_destino: str,
    identificador: str,
    limite_mb: int = 200,
) -> tuple[str, str]:
    """Baixa um Reels e retorna (caminho_do_arquivo, nome_para_download)."""
    url_limpa = normalizar_url_instagram(url)
    pasta = Path(pasta_destino)
    pasta.mkdir(parents=True, exist_ok=True)

    prefixo = pasta / f"{identificador}_instagram"
    template = str(prefixo) + ".%(ext)s"
    limite_bytes = int(limite_mb) * 1024 * 1024
    cookiefile: str | None = None
    info: dict | None = None

    try:
        # Primeiro tenta anonimamente, evitando usar cookies sem necessidade.
        try:
            info = _executar_download(
                url_limpa, template, limite_bytes, limite_mb, cookiefile=None
            )
        except yt_dlp.utils.DownloadError as primeiro_erro:
            mensagem = str(primeiro_erro)
            if "limite" in mensagem.lower() or "too large" in mensagem.lower():
                raise InstagramImportError(
                    f"O vídeo ultrapassa o limite de {limite_mb} MB."
                ) from primeiro_erro

            if not _erro_pede_autenticacao(mensagem):
                raise InstagramImportError(
                    "Não foi possível importar esse Reels. Confirme o link e tente novamente."
                ) from primeiro_erro

            # IPs de nuvem podem receber resposta vazia mesmo para posts públicos.
            cookiefile = _criar_arquivo_cookies_temporario()
            if not cookiefile:
                raise InstagramImportError(
                    "O Instagram bloqueou a importação pelo servidor do Railway. "
                    "Configure a variável INSTAGRAM_COOKIES_B64 e tente novamente."
                ) from primeiro_erro

            try:
                info = _executar_download(
                    url_limpa, template, limite_bytes, limite_mb, cookiefile=cookiefile
                )
            except yt_dlp.utils.DownloadError as segundo_erro:
                mensagem2 = str(segundo_erro)
                if "limite" in mensagem2.lower() or "too large" in mensagem2.lower():
                    raise InstagramImportError(
                        f"O vídeo ultrapassa o limite de {limite_mb} MB."
                    ) from segundo_erro
                if _erro_pede_autenticacao(mensagem2):
                    raise InstagramImportError(
                        "Os cookies do Instagram expiraram ou foram recusados. "
                        "Exporte um cookies.txt novo e atualize INSTAGRAM_COOKIES_B64 no Railway."
                    ) from segundo_erro
                raise InstagramImportError(
                    "O Instagram recusou esse download mesmo com autenticação. Tente novamente mais tarde."
                ) from segundo_erro

        titulo = ((info or {}).get("title") or (info or {}).get("id") or "reel_instagram").strip()

    finally:
        if cookiefile:
            try:
                os.remove(cookiefile)
            except OSError:
                pass

    candidatos = sorted(pasta.glob(f"{identificador}_instagram.*"))
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
        raise InstagramImportError(
            "O download terminou, mas o arquivo de vídeo não foi encontrado."
        )

    if arquivo.stat().st_size > limite_bytes:
        try:
            arquivo.unlink()
        except OSError:
            pass
        raise InstagramImportError(
            f"O vídeo ultrapassa o limite de {limite_mb} MB."
        )

    nome = re.sub(r"[^A-Za-z0-9._-]+", "_", titulo).strip("._")[:80]
    nome = nome or "reel_instagram"
    return str(arquivo), f"{nome}.mp4"

# Compatibilidade com os nomes esperados pelo app.py.
VideoImportError = InstagramImportError


def baixar_video(
    url: str,
    pasta_destino: str,
    identificador: str,
    limite_mb: int = 200,
) -> tuple[str, str]:
    """Alias compatível para a importação do Instagram."""
    return baixar_video_instagram(
        url=url,
        pasta_destino=pasta_destino,
        identificador=identificador,
        limite_mb=limite_mb,
    )
