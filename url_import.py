#!/usr/bin/env python3
"""
Importação genérica de vídeos por URL usando yt-dlp.

Aceita URLs de Instagram, TikTok, YouTube, Twitter/X e outros sites.
Tenta primeiro sem autenticação e repete com cookies quando disponíveis.

Variáveis aceitas (os dois padrões são reconhecidos):
- COOKIES_B64 ou INSTAGRAM_COOKIES_B64
- COOKIES ou INSTAGRAM_COOKIES
- FORCE_IPV4 ou INSTAGRAM_FORCE_IPV4: "1" para forçar IPv4
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


_DOMINIOS_INSTAGRAM = {
    "instagram.com",
    "www.instagram.com",
    "m.instagram.com",
}


def _primeira_variavel_preenchida(*nomes: str) -> str:
    for nome in nomes:
        valor = os.environ.get(nome, "").strip()
        if valor:
            return valor
    return ""


def _url_eh_instagram(url: str) -> bool:
    try:
        return (urlparse(url).hostname or "").lower() in _DOMINIOS_INSTAGRAM
    except Exception:
        return False


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

    # Remove parâmetros como ?igsh= dos links do Instagram. Eles não são
    # necessários para o download e às vezes atrapalham a identificação.
    if host in _DOMINIOS_INSTAGRAM:
        caminho = parsed.path.rstrip("/")
        return f"https://www.instagram.com{caminho}/"

    return url


def _criar_arquivo_cookies_temporario() -> str | None:
    """Cria cookies.txt temporário a partir das variáveis de ambiente."""
    conteudo: str | None = None

    # Compatibilidade com os nomes usados no projeto antigo e no .env atual.
    cookies_b64 = _primeira_variavel_preenchida(
        "COOKIES_B64",
        "INSTAGRAM_COOKIES_B64",
    )
    cookies_raw = _primeira_variavel_preenchida(
        "COOKIES",
        "INSTAGRAM_COOKIES",
    )

    if cookies_b64:
        try:
            conteudo = base64.b64decode(cookies_b64, validate=True).decode("utf-8")
        except Exception as exc:
            raise VideoImportError(
                "A variável de cookies em Base64 está inválida. "
                "Exporte novamente o cookies.txt e gere um novo Base64."
            ) from exc
    elif cookies_raw:
        # Railway pode armazenar as quebras de linha como os caracteres \n.
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


def _erro_de_limite(mensagem: str) -> bool:
    texto = mensagem.lower()
    return "limite" in texto or "too large" in texto or "larger than" in texto


def _erro_pede_autenticacao_ou_fallback(mensagem: str) -> bool:
    """Detecta bloqueios e falhas de extração que podem ser resolvidas com cookies."""
    texto = mensagem.lower()
    sinais = (
        "empty media response",
        "unable to extract video url",
        "unable to extract username",
        "no video formats found",
        "login",
        "log in",
        "cookies",
        "authentication",
        "main webpage is locked",
        "requested content is not available",
        "not available in your country",
        "access denied",
        "forbidden",
        "http error 401",
        "http error 403",
        "http error 429",
        "too many requests",
    )
    return any(sinal in texto for sinal in sinais)


def _forcar_ipv4() -> bool:
    valor = _primeira_variavel_preenchida(
        "FORCE_IPV4",
        "INSTAGRAM_FORCE_IPV4",
    )
    return valor == "1"


def _opcoes_yt_dlp(
    url: str,
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

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/150.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    if _url_eh_instagram(url):
        headers["Referer"] = "https://www.instagram.com/"

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
        "extractor_retries": 3,
        "match_filter": rejeitar_grande,
        "restrictfilenames": True,
        "cachedir": False,
        "http_headers": headers,
    }

    if cookiefile:
        opcoes["cookiefile"] = cookiefile

    if _forcar_ipv4():
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
        url=url,
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

    Suporta Instagram, TikTok, YouTube, Twitter/X e muitos outros sites.
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
        try:
            # Primeira tentativa sem autenticação.
            info = _executar_download(
                url_limpa,
                template,
                limite_bytes,
                limite_mb,
                cookiefile=None,
            )
        except yt_dlp.utils.DownloadError as primeiro_erro:
            mensagem = str(primeiro_erro)

            if _erro_de_limite(mensagem):
                raise VideoImportError(
                    f"O vídeo ultrapassa o limite de {limite_mb} MB."
                ) from primeiro_erro

            # Tenta cookies sempre que estiverem configurados. Isso cobre também
            # mudanças recentes do Instagram cuja mensagem ainda não está mapeada.
            cookiefile = _criar_arquivo_cookies_temporario()
            if not cookiefile:
                if _url_eh_instagram(url_limpa) or _erro_pede_autenticacao_ou_fallback(mensagem):
                    raise VideoImportError(
                        "O Instagram recusou a extração anônima. Configure "
                        "INSTAGRAM_COOKIES_B64 no Railway com um cookies.txt novo "
                        "e faça um novo deploy."
                    ) from primeiro_erro
                raise VideoImportError(
                    "Não foi possível importar esse vídeo. Confirme se o link é "
                    "público e se o yt-dlp está atualizado."
                ) from primeiro_erro

            try:
                info = _executar_download(
                    url_limpa,
                    template,
                    limite_bytes,
                    limite_mb,
                    cookiefile=cookiefile,
                )
            except yt_dlp.utils.DownloadError as segundo_erro:
                mensagem2 = str(segundo_erro)
                if _erro_de_limite(mensagem2):
                    raise VideoImportError(
                        f"O vídeo ultrapassa o limite de {limite_mb} MB."
                    ) from segundo_erro

                if _erro_pede_autenticacao_ou_fallback(mensagem2):
                    raise VideoImportError(
                        "O Instagram recusou o download mesmo com cookies. "
                        "Exporte um cookies.txt novo da conta logada, atualize "
                        "INSTAGRAM_COOKIES_B64 e confirme que o Reels está acessível."
                    ) from segundo_erro

                raise VideoImportError(
                    "O servidor recusou esse download mesmo com autenticação. "
                    "Confirme se o vídeo ainda existe e se é acessível pela conta "
                    "usada para exportar os cookies."
                ) from segundo_erro

        titulo = ((info or {}).get("title") or (info or {}).get("id") or "video").strip()

    finally:
        if cookiefile:
            try:
                os.remove(cookiefile)
            except OSError:
                pass

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
        raise VideoImportError(f"O vídeo ultrapassa o limite de {limite_mb} MB.")

    nome = re.sub(r"[^A-Za-z0-9._-]+", "_", titulo).strip("._")[:80]
    nome = nome or "video_importado"
    return str(arquivo), f"{nome}.mp4"
