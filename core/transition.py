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
    """Retorna proporção alterada, diferença média e cobertura espacial."""
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


def _feature_score(
    first: np.ndarray,
    second: np.ndarray,
    first_hist: np.ndarray | None = None,
    second_hist: np.ndarray | None = None,
) -> tuple[float, float, float, float, float]:
    global_ratio, mean_diff, spatial = _difference_features(first, second)
    hist_distance = _hist_distance(
        first_hist if first_hist is not None else _gray_hist(first),
        second_hist if second_hist is not None else _gray_hist(second),
    )
    score = (
        0.38 * spatial
        + 0.25 * global_ratio
        + 0.17 * min(mean_diff * 4.0, 1.0)
        + 0.20 * min(hist_distance * 1.5, 1.0)
    )
    return score, global_ratio, mean_diff, spatial, hist_distance


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
    hists: list[np.ndarray],
    *,
    min_time: float,
) -> TransitionResult | None:
    """Encontra o fim do primeiro take quando ele é (quase) estático.

    A versão anterior aceitava qualquer pico de diferença que durasse ~0.2s.
    Um gesto de mão, o rosto se aproximando ou o celular balançando disparavam
    a "troca" ainda dentro do primeiro take — e a modelo original reaparecia
    no vídeo final. A regra nova: um corte de verdade muda a cena inteira e a
    cena antiga NUNCA volta. Um gesto vai e volta. Então cada candidato só é
    aceito se os quadros seguintes permanecerem diferentes da referência até
    o fim da janela analisada.
    """
    initial = [frame for time_sec, frame in zip(times, frames) if time_sec <= 0.55]
    if not initial:
        return None
    reference = np.median(np.stack(initial), axis=0).astype(np.uint8)
    reference_hist = _gray_hist(reference)

    def features(index: int) -> tuple[float, float, float, float]:
        global_ratio, mean_diff, spatial = _difference_features(reference, frames[index])
        hist_distance = _hist_distance(reference_hist, hists[index])
        return global_ratio, mean_diff, spatial, hist_distance

    def is_scene_change(global_ratio: float, mean_diff: float, spatial: float, hist_distance: float) -> bool:
        # Mudança de cena inteira, não um objeto passando na frente.
        return (
            (spatial >= 0.55 and global_ratio >= 0.30)
            or (hist_distance >= 0.45 and global_ratio >= 0.22 and spatial >= 0.35)
            or (spatial >= 0.45 and hist_distance >= 0.32 and global_ratio >= 0.25 and mean_diff >= 0.06)
        )

    def still_changed(global_ratio: float, spatial: float, hist_distance: float) -> bool:
        # Depois do corte a cena pode ter movimento, então o critério de
        # permanência é um pouco mais tolerante do que o de disparo.
        return spatial >= 0.32 and (global_ratio >= 0.20 or hist_distance >= 0.30)

    def returned_to_intro(global_ratio: float, spatial: float, hist_distance: float) -> bool:
        return global_ratio < 0.08 and spatial < 0.12 and hist_distance < 0.10

    strongest = 0.0
    total = len(frames)
    for index in range(total):
        time_sec = times[index]
        if time_sec < min_time:
            continue
        global_ratio, mean_diff, spatial, hist_distance = features(index)
        score = (
            0.40 * spatial
            + 0.30 * global_ratio
            + 0.15 * min(mean_diff * 3.0, 1.0)
            + 0.15 * min(hist_distance * 1.5, 1.0)
        )
        strongest = max(strongest, score)
        if not is_scene_change(global_ratio, mean_diff, spatial, hist_distance):
            continue

        # Verificação de permanência: olha até 1.6s à frente (ou até o fim).
        lookahead_end = time_sec + 1.6
        checked = 0
        confirmed = 0
        came_back = False
        for future_index in range(index + 1, total):
            if times[future_index] > lookahead_end:
                break
            f_global, _f_mean, f_spatial, f_hist = features(future_index)
            checked += 1
            if still_changed(f_global, f_spatial, f_hist):
                confirmed += 1
            if returned_to_intro(f_global, f_spatial, f_hist):
                came_back = True
                break

        if came_back:
            continue
        # Perto do fim da janela pode haver poucos quadros; exige o que existir.
        if not checked:
            continue
        required = max(2, int(np.ceil(checked * 0.85)))
        if confirmed < required:
            continue

        # Refinamento: a divergência pode começar ANTES do corte de verdade
        # (a pessoa levanta um produto, se aproxima da câmera etc. e logo em
        # seguida vem o corte). Se existir um salto quadro-a-quadro dominante
        # logo à frente, a transição é encaixada nele — senão sobra ~1s da
        # cena original no vídeo final.
        refine_end = index
        refine_limit = time_sec + 2.2
        while refine_end + 1 < total and times[refine_end + 1] <= refine_limit:
            refine_end += 1
        boundaries: list[tuple[int, float]] = []
        for j in range(max(1, index), refine_end + 1):
            jump = _feature_score(frames[j - 1], frames[j], hists[j - 1], hists[j])[0]
            boundaries.append((j, jump))
        detected_time = time_sec
        if boundaries:
            max_jump = max(jump for _j, jump in boundaries)
            if max_jump >= 0.42:
                cutoff = max(0.42, 0.62 * max_jump)
                for j, jump in boundaries:
                    if jump >= cutoff:
                        detected_time = times[j]
                        break

        detected = max(min_time, detected_time)
        return TransitionResult(
            round(detected, 3),
            min(0.99, 0.55 + score),
            f"Troca do take estático detectada em {detected:.2f}s.",
        )

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


def _median_frame(frames: list[np.ndarray], start: int, end: int) -> np.ndarray:
    selected = frames[max(0, start):max(start + 1, end)]
    if not selected:
        selected = [frames[max(0, min(start, len(frames) - 1))]]
    return np.median(np.stack(selected), axis=0).astype(np.uint8)


def _detect_moving_take_end(
    times: list[float],
    frames: list[np.ndarray],
    hists: list[np.ndarray],
    *,
    min_time: float,
    sample_rate: float,
) -> TransitionResult | None:
    """Detecta corte real quando o primeiro take possui movimento.

    A versão anterior aceitava uma diferença grande entre dois quadros consecutivos.
    Movimento do rosto, mão ou celular podia ser interpretado como troca e fazia o
    vídeo original reaparecer ainda no primeiro take. Agora cada candidato precisa:

    1. ser um pico local de mudança;
    2. separar duas janelas de quadros visualmente diferentes;
    3. permanecer diferente nos quadros posteriores.
    """
    if len(frames) < 7:
        return None

    boundary_scores: list[float] = [0.0]
    boundary_features: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)]
    for index in range(1, len(frames)):
        score, global_ratio, mean_diff, spatial, hist_distance = _feature_score(
            frames[index - 1], frames[index], hists[index - 1], hists[index]
        )
        boundary_scores.append(score)
        boundary_features.append((global_ratio, mean_diff, spatial, hist_distance))

    window = max(3, int(round(sample_rate * 0.42)))
    strongest = 0.0
    candidates: list[tuple[float, int, float]] = []

    for index in range(window, len(frames) - window):
        time_sec = times[index]
        if time_sec < min_time:
            continue

        adjacent_score = boundary_scores[index]
        strongest = max(strongest, adjacent_score)

        neighborhood = boundary_scores[max(1, index - window):min(len(boundary_scores), index + window + 1)]
        if adjacent_score + 1e-9 < max(neighborhood):
            continue

        history = boundary_scores[max(1, index - 2 * window):index]
        future_motion = boundary_scores[index + 1:min(len(boundary_scores), index + 1 + window)]
        baseline_values = history + future_motion
        baseline = float(np.median(baseline_values)) if baseline_values else 0.0
        mad = float(np.median(np.abs(np.asarray(baseline_values) - baseline))) if baseline_values else 0.0
        peak_limit = max(0.30, baseline + max(0.12, 4.5 * mad), baseline * 1.75)
        if adjacent_score < peak_limit:
            continue

        pre_scene = _median_frame(frames, index - window, index)
        post_scene = _median_frame(frames, index, index + window)
        scene_score, scene_global, scene_mean, scene_spatial, scene_hist = _feature_score(
            pre_scene, post_scene
        )

        # Uma troca verdadeira altera a identidade visual da cena inteira. Um gesto
        # ou movimento de câmera costuma gerar pico entre quadros, mas as medianas
        # das janelas anterior e posterior continuam relativamente parecidas.
        persistent_change = (
            (scene_spatial >= 0.50 and scene_global >= 0.30 and scene_mean >= 0.075)
            or (scene_hist >= 0.43 and scene_global >= 0.22 and scene_spatial >= 0.36)
            or (scene_score >= 0.50 and scene_spatial >= 0.42 and scene_global >= 0.25)
        )
        if not persistent_change:
            continue

        before_scene = pre_scene
        before_hist = _gray_hist(before_scene)
        confirmations = 0
        checked = 0
        for future_index in range(index, min(len(frames), index + window)):
            checked += 1
            _, future_global, future_mean, future_spatial, future_hist = _feature_score(
                before_scene,
                frames[future_index],
                before_hist,
                hists[future_index],
            )
            if (
                future_spatial >= 0.40
                and future_global >= 0.23
                and (future_mean >= 0.06 or future_hist >= 0.30)
            ):
                confirmations += 1

        required = max(2, int(np.ceil(checked * 0.65)))
        if confirmations < required:
            continue

        combined = 0.42 * adjacent_score + 0.58 * scene_score
        confidence = min(0.99, 0.48 + combined + min(0.12, max(0.0, adjacent_score - baseline)))
        candidates.append((time_sec, index, confidence))

    if not candidates:
        return TransitionResult(None, max(0.0, min(strongest, 0.49)), "")

    # Entre cortes válidos, usa o primeiro. Agora ele já foi confirmado pelas
    # janelas anterior/posterior, então não é apenas o primeiro movimento forte.
    detected, _, confidence = min(candidates, key=lambda item: item[0])
    return TransitionResult(
        round(detected, 3),
        confidence,
        f"Troca real entre cenas detectada em {detected:.2f}s.",
    )


def detect_intro_end(
    video_path: str | Path,
    *,
    sample_rate: float = 10.0,
    min_time: float = 0.7,
    max_scan_seconds: float | None = None,
) -> TransitionResult:
    """Detecta o ponto em que termina o primeiro take do vídeo original."""

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
    if looks_static:
        result = _detect_static_take_end(times, frames, hists, min_time=min_time)
        # Se o caminho estático não achou nada, tenta o detector de movimento:
        # cobre casos em que o começo parece parado mas o take tem movimento depois.
        if result is None or result.seconds is None:
            moving = _detect_moving_take_end(
                times,
                frames,
                hists,
                min_time=min_time,
                sample_rate=sample_rate,
            )
            if moving and moving.seconds is not None:
                result = moving
    else:
        result = _detect_moving_take_end(
            times,
            frames,
            hists,
            min_time=min_time,
            sample_rate=sample_rate,
        )

    if result and result.seconds is not None:
        return result

    confidence = result.confidence if result else 0.0
    scanned_note = "" if scan_until >= info.duration - 0.05 else f" nos primeiros {scan_until:.0f}s"
    return TransitionResult(
        None,
        confidence,
        f"Nenhuma troca real de take foi encontrada{scanned_note}.",
    )


# Compatibilidade com chamadas da versão anterior.
def detect_static_intro_end(
    video_path: str | Path,
    *,
    sample_rate: float = 10.0,
    min_time: float = 0.7,
    max_scan_seconds: float | None = None,
) -> TransitionResult:
    return detect_intro_end(
        video_path,
        sample_rate=sample_rate,
        min_time=min_time,
        max_scan_seconds=max_scan_seconds,
    )
