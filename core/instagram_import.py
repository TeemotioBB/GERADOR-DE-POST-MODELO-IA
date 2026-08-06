from __future__ import annotations

import base64
import os
import re
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import yt_dlp


class InstagramImportError(RuntimeError):
    """Erro de importação que pode ser exibido diretamente na interface."""


ProgressCallback = Callable[[float, str], None]

_DOMAINS = {
    "instagram.com",
    "www.instagram.com",
    "m.instagram.com",
}
_REEL_PATH = re.compile(r"^/(?:reel|reels|p)/[A-Za-z0-9_-]+/?$")


def normalize_instagram_url(url: str) -> str:
    """Valida o link e remove parâmetros como ``igsh`` e UTMs."""
    value = (url or "").strip()
    if not value:
        raise InstagramImportError("Cole o link do Reels do Instagram.")

    try:
        parsed = urlparse(value)
    except Exception as exc:
        raise InstagramImportError("Link do Instagram inválido.") from exc

    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in _DOMAINS:
        raise InstagramImportError("Use um link válido do Instagram.")

    path = parsed.path.rstrip("/")
    if not _REEL_PATH.match(path):
        raise InstagramImportError(
            "Esse endereço não parece ser um Reels ou uma publicação do Instagram."
        )

    return f"https://www.instagram.com{path}/"


def _create_cookie_file() -> str | None:
    """Cria um cookies.txt temporário a partir das variáveis do Railway."""
    content: str | None = None
    cookies_b64 = os.getenv("INSTAGRAM_COOKIES_B64", "").strip()
    cookies_raw = os.getenv("INSTAGRAM_COOKIES", "").strip()

    if cookies_b64:
        try:
            content = base64.b64decode(cookies_b64, validate=True).decode("utf-8")
        except Exception as exc:
            raise InstagramImportError(
                "A variável INSTAGRAM_COOKIES_B64 está inválida. Gere novamente o Base64 do cookies.txt."
            ) from exc
    elif cookies_raw:
        # Alguns painéis salvam quebras de linha como os caracteres literais \n.
        content = cookies_raw.replace("\\n", "\n")

    if not content:
        return None

    if "# Netscape HTTP Cookie File" not in content and "\t.instagram.com\t" not in content:
        raise InstagramImportError(
            "Os cookies do Instagram não parecem estar no formato Netscape cookies.txt."
        )

    cookie_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix="_instagram_cookies.txt",
        delete=False,
    )
    try:
        cookie_file.write(content)
        if not content.endswith("\n"):
            cookie_file.write("\n")
        return cookie_file.name
    finally:
        cookie_file.close()


def _looks_like_auth_error(message: str) -> bool:
    text = message.lower()
    signals = (
        "empty media response",
        "login",
        "log in",
        "cookies",
        "authentication",
        "main webpage is locked behind the login page",
        "requested content is not available",
        "rate-limit",
        "rate limit",
        "403",
        "forbidden",
        "429",
        "no video formats",
        "challenge",
    )
    return any(signal in text for signal in signals)


def _download_options(
    *,
    template: str,
    max_bytes: int,
    max_mb: int,
    cookiefile: str | None,
    progress_callback: ProgressCallback | None,
) -> dict:
    def reject_large(info, *, incomplete=False):
        del incomplete
        size = info.get("filesize") or info.get("filesize_approx")
        if size and size > max_bytes:
            return f"O vídeo ultrapassa o limite de {max_mb} MB."
        return None

    def progress_hook(status: dict) -> None:
        if progress_callback is None:
            return
        state = status.get("status")
        if state == "downloading":
            downloaded = float(status.get("downloaded_bytes") or 0)
            total = float(status.get("total_bytes") or status.get("total_bytes_estimate") or 0)
            if total > 0:
                fraction = max(0.0, min(downloaded / total, 1.0))
                progress_callback(fraction, f"Baixando o Reels... {fraction:.0%}")
            else:
                progress_callback(0.15, "Baixando o Reels do Instagram...")
        elif state == "finished":
            progress_callback(1.0, "Download concluído. Validando o vídeo...")

    options = {
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
        "match_filter": reject_large,
        "restrictfilenames": True,
        "cachedir": False,
        "overwrites": True,
        "progress_hooks": [progress_hook],
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
        options["cookiefile"] = cookiefile

    if os.getenv("INSTAGRAM_FORCE_IPV4", "").strip() == "1":
        options["source_address"] = "0.0.0.0"

    return options


def _execute_download(
    *,
    url: str,
    template: str,
    max_bytes: int,
    max_mb: int,
    cookiefile: str | None,
    progress_callback: ProgressCallback | None,
) -> dict:
    options = _download_options(
        template=template,
        max_bytes=max_bytes,
        max_mb=max_mb,
        cookiefile=cookiefile,
        progress_callback=progress_callback,
    )
    with yt_dlp.YoutubeDL(options) as ydl:
        return ydl.extract_info(url, download=True)


def download_instagram_video(
    url: str,
    destination_dir: str | Path,
    identifier: str,
    *,
    max_mb: int = 500,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, str]:
    """Baixa um Reels e retorna ``(caminho, nome_amigável)``."""
    clean_url = normalize_instagram_url(url)
    folder = Path(destination_dir)
    folder.mkdir(parents=True, exist_ok=True)

    prefix = folder / f"{identifier}_instagram"
    template = str(prefix) + ".%(ext)s"
    max_bytes = int(max_mb) * 1024 * 1024
    cookiefile: str | None = None
    info: dict | None = None

    try:
        try:
            info = _execute_download(
                url=clean_url,
                template=template,
                max_bytes=max_bytes,
                max_mb=max_mb,
                cookiefile=None,
                progress_callback=progress_callback,
            )
        except yt_dlp.utils.DownloadError as first_error:
            message = str(first_error)
            if "limite" in message.lower() or "too large" in message.lower():
                raise InstagramImportError(
                    f"O vídeo ultrapassa o limite de {max_mb} MB."
                ) from first_error

            if not _looks_like_auth_error(message):
                raise InstagramImportError(
                    "Não foi possível importar esse Reels. Confirme se o link é público e tente novamente."
                ) from first_error

            # IPs de datacenter podem ser bloqueados mesmo em Reels públicos.
            cookiefile = _create_cookie_file()
            if not cookiefile:
                raise InstagramImportError(
                    "O Instagram bloqueou o download pelo servidor. Configure "
                    "INSTAGRAM_COOKIES_B64 no Railway e tente novamente."
                ) from first_error

            try:
                info = _execute_download(
                    url=clean_url,
                    template=template,
                    max_bytes=max_bytes,
                    max_mb=max_mb,
                    cookiefile=cookiefile,
                    progress_callback=progress_callback,
                )
            except yt_dlp.utils.DownloadError as second_error:
                second_message = str(second_error)
                if "limite" in second_message.lower() or "too large" in second_message.lower():
                    raise InstagramImportError(
                        f"O vídeo ultrapassa o limite de {max_mb} MB."
                    ) from second_error
                if _looks_like_auth_error(second_message):
                    raise InstagramImportError(
                        "Os cookies do Instagram expiraram ou foram recusados. "
                        "Exporte um cookies.txt novo e atualize INSTAGRAM_COOKIES_B64 no Railway."
                    ) from second_error
                raise InstagramImportError(
                    "O Instagram recusou o download mesmo com autenticação. Tente novamente mais tarde."
                ) from second_error
    finally:
        if cookiefile:
            try:
                os.remove(cookiefile)
            except OSError:
                pass

    candidates = sorted(folder.glob(f"{identifier}_instagram.*"))
    downloaded = next((item for item in candidates if item.suffix.lower() == ".mp4"), None)
    if downloaded is None:
        downloaded = next(
            (
                item
                for item in candidates
                if item.is_file() and item.suffix.lower() not in {".part", ".ytdl"}
            ),
            None,
        )

    if downloaded is None or not downloaded.exists():
        raise InstagramImportError(
            "O download terminou, mas o arquivo de vídeo não foi encontrado."
        )

    if downloaded.stat().st_size > max_bytes:
        try:
            downloaded.unlink()
        except OSError:
            pass
        raise InstagramImportError(f"O vídeo ultrapassa o limite de {max_mb} MB.")

    title = ((info or {}).get("title") or (info or {}).get("id") or "reel_instagram").strip()
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._")[:70]
    safe_name = safe_name or "reel_instagram"
    final_path = folder / f"{safe_name}{downloaded.suffix.lower()}"
    if final_path != downloaded:
        if final_path.exists():
            final_path = folder / f"{safe_name}_{identifier[:8]}{downloaded.suffix.lower()}"
        downloaded.replace(final_path)
        downloaded = final_path

    return str(downloaded), downloaded.name
