#!/usr/bin/env python3
"""Importa Reels do Instagram preservando vídeo e áudio quando disponíveis.

Falhas conhecidas do Instagram/yt-dlp podem produzir MP4 somente com vídeo
(especialmente ao acessar o Instagram via IPv6). O importador prioriza IPv4,
valida cada download com ffprobe e nunca declara sucesso para um arquivo mudo.

Variáveis opcionais:
- INSTAGRAM_FORCE_IPV4: 1 (padrão) para IPv4; 0 tenta rede automática primeiro,
  mas ainda tenta IPv4 se o vídeo chegar sem áudio.
- INSTAGRAM_COOKIES_B64: cookies.txt Netscape em Base64, usado após falhas.
- INSTAGRAM_COOKIES: cookies.txt Netscape em texto, usado após falhas.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp

from .media import MediaError, has_audible_audio, probe_video

logger = logging.getLogger(__name__)


class InstagramImportError(RuntimeError):
    """Mensagem segura para exibir ao usuário."""


_DOMINIOS = {"instagram.com", "www.instagram.com", "m.instagram.com"}
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
    return f"https://www.instagram.com{caminho}/"


def _ha_cookies_configurados() -> bool:
    return bool(os.getenv("INSTAGRAM_COOKIES_B64", "").strip() or os.getenv("INSTAGRAM_COOKIES", "").strip())


def _criar_arquivo_cookies_temporario() -> str | None:
    """Cria cookies temporários somente quando um segundo download é necessário."""
    conteudo: str | None = None
    cookies_b64 = os.environ.get("INSTAGRAM_COOKIES_B64", "").strip()
    cookies_raw = os.environ.get("INSTAGRAM_COOKIES", "").strip()
    if cookies_b64:
        try:
            conteudo = base64.b64decode(cookies_b64, validate=True).decode("utf-8")
        except Exception as exc:
            raise InstagramImportError(
                "INSTAGRAM_COOKIES_B64 está inválida. Exporte novamente os cookies."
            ) from exc
    elif cookies_raw:
        conteudo = cookies_raw.replace("\\n", "\n")
    if not conteudo:
        return None
    if "# Netscape HTTP Cookie File" not in conteudo and "\t.instagram.com\t" not in conteudo:
        raise InstagramImportError("Os cookies não estão no formato Netscape cookies.txt.")

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix="_instagram_cookies.txt", delete=False
    ) as arquivo:
        arquivo.write(conteudo)
        if not conteudo.endswith("\n"):
            arquivo.write("\n")
        return arquivo.name


def _erro_pede_autenticacao(mensagem: str) -> bool:
    return any(s in mensagem.lower() for s in (
        "empty media response", "login", "log in", "cookies", "authentication",
        "main webpage is locked behind the login page", "requested content is not available",
    ))


def _usar_ipv4_primeiro() -> bool:
    return os.getenv("INSTAGRAM_FORCE_IPV4", "1").strip().lower() not in {"0", "false", "no", "off"}


def _opcoes_yt_dlp(
    template: str,
    limite_bytes: int,
    limite_mb: int,
    cookiefile: str | None,
    *,
    forcar_ipv4: bool | None = None,
) -> dict:
    def rejeitar_grande(info, *, incomplete=False):
        del incomplete
        tamanho = info.get("filesize") or info.get("filesize_approx")
        if tamanho and tamanho > limite_bytes:
            return f"O vídeo ultrapassa o limite de {limite_mb} MB."
        return None

    opcoes = {
        "outtmpl": template,
        # Nunca usar fallback explícito para vídeo-only. O ffprobe verificará
        # a faixa real do arquivo; metadados do extractor podem estar incorretos.
        "format": "bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/b[ext=mp4]/b",
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
    if forcar_ipv4 is None:
        forcar_ipv4 = _usar_ipv4_primeiro()
    if forcar_ipv4:
        opcoes["source_address"] = "0.0.0.0"
    return opcoes


def _executar_download(
    url: str,
    template: str,
    limite_bytes: int,
    limite_mb: int,
    cookiefile: str | None,
    *,
    forcar_ipv4: bool | None = None,
) -> dict:
    opcoes = _opcoes_yt_dlp(
        template=template, limite_bytes=limite_bytes, limite_mb=limite_mb,
        cookiefile=cookiefile, forcar_ipv4=forcar_ipv4,
    )
    with yt_dlp.YoutubeDL(opcoes) as ydl:
        return ydl.extract_info(url, download=True)


def _arquivos_tentativa(pasta: Path, identificador: str) -> list[Path]:
    return sorted(
        (p for p in pasta.glob(f"{identificador}_instagram.*") if p.is_file()),
        key=lambda p: p.name,
    )


def _limpar_tentativa(pasta: Path, identificador: str) -> None:
    for arquivo in _arquivos_tentativa(pasta, identificador):
        try:
            arquivo.unlink()
        except OSError:
            logger.warning("Não foi possível remover arquivo temporário do Instagram: %s", arquivo.name)


def _arquivo_baixado(pasta: Path, identificador: str) -> Path | None:
    candidatos = _arquivos_tentativa(pasta, identificador)
    return next((p for p in candidatos if p.suffix.lower() == ".mp4"), None)


def _validar_arquivo_com_audio(arquivo: Path | None, limite_bytes: int, limite_mb: int, *, aceitar_sem_audio: bool = False) -> None:
    if arquivo is None:
        raise InstagramImportError("O download terminou, mas não foi encontrado um arquivo MP4.")
    if arquivo.stat().st_size > limite_bytes:
        raise InstagramImportError(f"O vídeo ultrapassa o limite de {limite_mb} MB.")
    try:
        info = probe_video(arquivo)
    except (MediaError, ValueError, OSError) as exc:
        raise InstagramImportError("O Instagram retornou um arquivo de vídeo inválido.") from exc
    # Faixa presente porém muda (comum em falhas do Instagram/yt-dlp) = sem áudio.
    if not aceitar_sem_audio and not (info.has_audio and has_audible_audio(arquivo)):
        raise InstagramImportError(
            "O Instagram retornou apenas a imagem do Reel, sem a faixa de áudio."
        )


def baixar_video_instagram(
    url: str,
    pasta_destino: str,
    identificador: str,
    limite_mb: int = 200,
    permitir_sem_audio: bool = False,
) -> tuple[str, str]:
    """Baixa vídeo + áudio e verifica a faixa real antes de aceitar a importação.

    Normalmente faz um único download com IPv4 (bom para custo do Railway).
    Se receber arquivo sem áudio, repete pela rota IPv4 ou com cookies, quando
    disponíveis. Nunca devolve silenciosamente um arquivo sem faixa de áudio.
    """
    url_limpa = normalizar_url_instagram(url)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", identificador):
        raise InstagramImportError("Identificador de importação inválido.")
    pasta = Path(pasta_destino)
    pasta.mkdir(parents=True, exist_ok=True)
    limite_bytes = int(limite_mb) * 1024 * 1024
    template = str(pasta / f"{identificador}_instagram") + ".%(ext)s"

    primeiro_ipv4 = _usar_ipv4_primeiro()
    # Nenhum download duplicado se o primeiro tem áudio. Alternativas são usadas
    # somente após erro ou arquivo efetivamente sem faixa de áudio.
    tentativas = [("padrão", primeiro_ipv4, False)]
    if not primeiro_ipv4:
        tentativas.append(("IPv4", True, False))
    if _ha_cookies_configurados():
        tentativas.append(("IPv4 com cookies", True, True))

    cookiefile: str | None = None
    titulo = "reel_instagram"
    sem_audio = False
    erro_login = False
    ultimo_erro = ""
    sucesso = False
    try:
        for indice, (metodo, forcar_ipv4, usa_cookies) in enumerate(tentativas, start=1):
            _limpar_tentativa(pasta, identificador)
            if usa_cookies and cookiefile is None:
                cookiefile = _criar_arquivo_cookies_temporario()
            try:
                info = _executar_download(
                    url_limpa, template, limite_bytes, limite_mb,
                    cookiefile=cookiefile if usa_cookies else None,
                    forcar_ipv4=forcar_ipv4,
                )
                titulo = ((info or {}).get("title") or (info or {}).get("id") or titulo).strip()
                arquivo = _arquivo_baixado(pasta, identificador)
                _validar_arquivo_com_audio(arquivo, limite_bytes, limite_mb, aceitar_sem_audio=permitir_sem_audio)
                nome = re.sub(r"[^A-Za-z0-9._-]+", "_", titulo).strip("._")[:80]
                if permitir_sem_audio and not probe_video(arquivo).has_audio:
                    logger.warning("Instagram importado sem áudio por escolha explícita do usuário")
                else:
                    logger.info("Instagram importado com áudio, tentativa %s (%s)", indice, metodo)
                sucesso = True
                return str(arquivo), f"{nome or 'reel_instagram'}.mp4"
            except yt_dlp.utils.DownloadError as exc:
                ultimo_erro = str(exc)
                erro_login = erro_login or _erro_pede_autenticacao(ultimo_erro)
                logger.warning("Importação Instagram falhou na tentativa %s (%s)", indice, metodo)
                if "limite" in ultimo_erro.lower() or "too large" in ultimo_erro.lower():
                    raise InstagramImportError(f"O vídeo ultrapassa o limite de {limite_mb} MB.") from exc
            except InstagramImportError as exc:
                ultimo_erro = str(exc)
                if "limite" in ultimo_erro.lower():
                    raise
                sem_audio = sem_audio or "sem a faixa de áudio" in ultimo_erro
                logger.warning("Importação Instagram rejeitada na tentativa %s (%s): %s", indice, metodo, ultimo_erro)

        if sem_audio:
            raise InstagramImportError(
                "O Reel tem som no Instagram, mas o Instagram não forneceu a faixa de áudio "
                "ao servidor. A importação foi interrompida para não gerar um vídeo mudo. "
                "Envie um MP4 com áudio, ou marque a opção de áudio externo "
                "para importar o vídeo silencioso e anexar uma faixa separada. "
                "Se tiver cookies do Instagram, configure INSTAGRAM_COOKIES_B64 no Railway."
            )
        if erro_login:
            raise InstagramImportError(
                "O Instagram bloqueou a importação pelo servidor. "
                "Configure ou atualize INSTAGRAM_COOKIES_B64 no Railway e tente novamente."
            )
        raise InstagramImportError(
            "Não foi possível importar o Reel. Confirme o link e tente novamente. "
            "Se persistir, envie um MP4 com áudio pela aba de arquivo."
        )
    finally:
        if cookiefile:
            try:
                os.remove(cookiefile)
            except OSError:
                pass
        if not sucesso:
            _limpar_tentativa(pasta, identificador)
