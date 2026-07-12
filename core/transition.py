from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import TRANSITION_MAX_SCAN_SECONDS
from .media import MediaError, probe_video


@dataclass(frozen=True)
class TransitionResult:
    seconds: float | None
    confidence: float
    message: str


def _normalize_frame(frame: np.ndarray, target_width: int = 224) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = target_width / max(width, 1)
    target_height = max(96, int(round(height * scale)))
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _gray_hist(frame: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([frame], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def _difference_features(reference: np.ndarray, current: np.ndarray) -> tuple[float, float, float]:
    """Retorna mudança global, intensidade média e cobertura espacial."""
    if reference.shape != current.shape:
        current = cv2.resize(current, (reference.shape[1], reference.shape[0]))

    diff = cv2.absdiff(reference, current)
    changed = diff >= 24
    global_ratio = float(np.mean(changed))
    mean_diff = float(np.mean(diff) / 255.0)

    rows, cols = 8, 6
    block_scores: list[float] = []
    h, w = changed.shape
    for row in range(rows):
        y0 = row * h // rows
        y1 = (row + 1) * h // rows
        for col in range(cols):
            x0 = col * w // cols
            x1 = (col + 1) * w // cols
            block = changed[y0:y1, x0:x1]
            if block.size:
                block_scores.append(float(np.mean(block)))

    active_blocks = sum(score >= 0.10 for score in block_scores)
    spatial_coverage = active_blocks / max(len(block_scores), 1)
    return global_ratio, mean_diff, spatial_coverage


def _hist_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(cv2.compareHist(first, second, cv2.HISTCMP_BHATTACHARYYA))


def _load_samples(
    video_path: str | Path,
    *,
    sample_rate: float,
    scan_until: float,
) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise MediaError("Não foi possível abrir o vídeo para detectar a transição.")

    times: list[float] = []
    frames: list[np.ndarray] = []
    hists: list[np.ndarray] = []
    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(source_fps / max(sample_rate, 0.5))))
        frame_index = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if frame_index % step != 0:
                frame_index += 1
                continue

            ok, frame = cap.retrieve()
            if not ok:
                break
            time_sec = frame_index / source_fps
            frame_index += 1
            if time_sec > scan_until:
                break

            normalized = _normalize_frame(frame)
            times.append(time_sec)
            frames.append(normalized)
            hists.append(_gray_hist(normalized))
    finally:
        cap.release()

    return times, frames, hists


def _detect_static_take_end(
    times: list[float],
    frames: list[np.ndarray],
    *,
    min_time: float,
) -> TransitionResult | None:
    initial = [frame for time_sec, frame in zip(times, frames) if time_sec <= 0.55]
    if not initial:
        return None
    reference = np.median(np.stack(initial), axis=0).astype(np.uint8)

    consecutive = 0
    first_candidate: float | None = None
    strongest = 0.0
    for time_sec, current in zip(times, frames):
        if time_sec < min_time:
            continue
        global_ratio, mean_diff, spatial = _difference_features(reference, current)
        score = 0.45 * spatial + 0.35 * global_ratio + 0.20 * min(mean_diff * 3.0, 1.0)
        strongest = max(strongest, score)
        changed_take = (
            spatial >= 0.30
            or global_ratio >= 0.30
            or (spatial >= 0.20 and global_ratio >= 0.16 and mean_diff >= 0.09)
        )
        if changed_take:
            if consecutive == 0:
                first_candidate = time_sec
            consecutive += 1
            if consecutive >= 2:
                detected = max(min_time, first_candidate or time_sec)
                return TransitionResult(
                    round(detected, 3),
                    min(0.99, 0.55 + score),
                    f"Troca do take estático detectada em {detected:.2f}s.",
                )
        else:
            consecutive = 0
            first_candidate = None
    return TransitionResult(None, max(0.0, min(strongest, 0.49)), "")


def _intro_looks_static(times: list[float], frames: list[np.ndarray]) -> bool:
    initial_indices = [index for index, time_sec in enumerate(times) if 0.10 <= time_sec <= 0.65]
    if not initial_indices or not frames:
        return True
    reference = frames[0]
    global_values: list[float] = []
    spatial_values: list[float] = []
    mean_values: list[float] = []
    for index in initial_indices:
        global_ratio, mean_diff, spatial = _difference_features(reference, frames[index])
        global_values.append(global_ratio)
        spatial_values.append(spatial)
        mean_values.append(mean_diff)
    return (
        float(np.median(global_values)) < 0.10
        and float(np.median(spatial_values)) < 0.20
        and float(np.median(mean_values)) < 0.055
    )


def _detect_moving_take_end(
    times: list[float],
    frames: list[np.ndarray],
    hists: list[np.ndarray],
    *,
    min_time: float,
) -> TransitionResult | None:
    if len(frames) < 4:
        return None

    scores: list[float] = []
    feature_rows: list[tuple[float, float, float, float]] = []
    for index in range(1, len(frames)):
        global_ratio, mean_diff, spatial = _difference_features(frames[index - 1], frames[index])
        hist_distance = _hist_distance(hists[index - 1], hists[index])
        score = (
            0.38 * spatial
            + 0.25 * global_ratio
            + 0.17 * min(mean_diff * 4.0, 1.0)
            + 0.20 * min(hist_distance * 1.5, 1.0)
        )
        scores.append(score)
        feature_rows.append((global_ratio, mean_diff, spatial, hist_distance))

    strongest = 0.0
    for offset, score in enumerate(scores, start=1):
        time_sec = times[offset]
        if time_sec < min_time:
            continue
        strongest = max(strongest, score)
        global_ratio, mean_diff, spatial, hist_distance = feature_rows[offset - 1]

        history = scores[max(0, offset - 13) : max(1, offset - 1)]
        baseline = float(np.median(history)) if history else 0.0
        mad = float(np.median(np.abs(np.asarray(history) - baseline))) if history else 0.0
        adaptive_limit = max(0.32, baseline + max(0.14, 4.0 * mad))

        hard_cut = (
            (spatial >= 0.46 and global_ratio >= 0.27 and (mean_diff >= 0.085 or hist_distance >= 0.20))
            or (hist_distance >= 0.45 and global_ratio >= 0.18 and spatial >= 0.28)
        )
        adaptive_cut = score >= adaptive_limit and spatial >= 0.30 and global_ratio >= 0.18
        if not (hard_cut or adaptive_cut):
            continue

        # Confirma que não foi somente um flash/quadro isolado: os próximos quadros
        # precisam continuar diferentes do quadro imediatamente anterior ao corte.
        before = frames[offset - 1]
        before_hist = hists[offset - 1]
        confirmations = 0
        checked = 0
        for future in range(offset, min(len(frames), offset + 3)):
            checked += 1
            future_global, future_mean, future_spatial = _difference_features(before, frames[future])
            future_hist = _hist_distance(before_hist, hists[future])
            if (
                future_spatial >= 0.34
                or future_global >= 0.28
                or (future_hist >= 0.34 and future_global >= 0.15 and future_mean >= 0.05)
            ):
                confirmations += 1
        if confirmations < min(2, checked):
            continue

        confidence = min(0.99, 0.50 + score + min(0.18, max(0.0, score - baseline)))
        return TransitionResult(
            round(time_sec, 3),
            confidence,
            f"Corte entre takes detectado em {time_sec:.2f}s, mesmo com movimento no primeiro take.",
        )

    return TransitionResult(None, max(0.0, min(strongest, 0.49)), "")


def detect_intro_end(
    video_path: str | Path,
    *,
    sample_rate: float = 8.0,
    min_time: float = 0.7,
    max_scan_seconds: float | None = None,
) -> TransitionResult:
    """Detecta a primeira troca de take, com intro estática ou em movimento."""

    info = probe_video(video_path)
    scan_limit = max_scan_seconds if max_scan_seconds is not None else TRANSITION_MAX_SCAN_SECONDS
    scan_until = min(info.duration, scan_limit)
    if scan_until <= min_time + 0.25:
        return TransitionResult(None, 0.0, "Vídeo curto demais para detectar uma transição.")

    times, frames, hists = _load_samples(
        video_path,
        sample_rate=sample_rate,
        scan_until=scan_until,
    )
    if len(frames) < 3:
        raise MediaError("Não foi possível ler quadros suficientes para analisar o vídeo.")

    looks_static = _intro_looks_static(times, frames)
    static_result = _detect_static_take_end(times, frames, min_time=min_time) if looks_static else None
    moving_result = _detect_moving_take_end(times, frames, hists, min_time=min_time)

    candidates = [result for result in (static_result, moving_result) if result and result.seconds is not None]
    if candidates:
        return min(candidates, key=lambda result: float(result.seconds or 0.0))

    confidence = max(
        static_result.confidence if static_result else 0.0,
        moving_result.confidence if moving_result else 0.0,
    )
    scanned_note = "" if scan_until >= info.duration - 0.05 else f" nos primeiros {scan_until:.0f}s"
    return TransitionResult(
        None,
        confidence,
        f"Nenhuma troca de take confiável foi encontrada{scanned_note}.",
    )


# Compatibilidade com chamadas da versão anterior.
def detect_static_intro_end(
    video_path: str | Path,
    *,
    sample_rate: float = 8.0,
    min_time: float = 0.7,
    max_scan_seconds: float | None = None,
) -> TransitionResult:
    return detect_intro_end(
        video_path,
        sample_rate=sample_rate,
        min_time=min_time,
        max_scan_seconds=max_scan_seconds,
    )
