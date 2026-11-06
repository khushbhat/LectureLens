"""Stage 5 — Keyframes / slide-change detection (fully local).

FFmpeg samples one frame every `sample_every_s` seconds at reduced width.
A new slide starts where consecutive frames differ by perceptual hash
(Hamming distance > threshold) or SSIM (< threshold). Runs shorter than
`min_slide_duration_s` (transitions, animations, presenter movement) are
merged into a neighbour, and A->B->A flicker is collapsed. For every slide
the sharpest frame from its second half (bullets fully revealed) is
re-extracted at native resolution for OCR.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .audio import check_ffmpeg
from .utils import read_json, write_json

log = logging.getLogger(__name__)


def sample_frames(video: str | Path, out_dir: Path, every_s: float, width: int) -> list[tuple[float, Path]]:
    check_ffmpeg()
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("f_*.jpg"))
    if not existing:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-an",
               "-vf", f"fps=1/{every_s},scale={width}:-2", "-q:v", "4", str(out_dir / "f_%06d.jpg")]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg frame sampling failed: {proc.stderr[-400:]}")
        existing = sorted(out_dir.glob("f_*.jpg"))
    return [(i * every_s, p) for i, p in enumerate(existing)]


def _gray_small(img: np.ndarray, w: int = 160) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h = max(1, int(g.shape[0] * w / g.shape[1]))
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global-window SSIM on small grayscale images (fast, no extra deps)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (7, 7), 1.5)
    mu_b = cv2.GaussianBlur(b, (7, 7), 1.5)
    s_aa = cv2.GaussianBlur(a * a, (7, 7), 1.5) - mu_a ** 2
    s_bb = cv2.GaussianBlur(b * b, (7, 7), 1.5) - mu_b ** 2
    s_ab = cv2.GaussianBlur(a * b, (7, 7), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return float(m.mean())


def sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def pixel_change_ratio(a: np.ndarray, b: np.ndarray, delta: int = 40) -> float:
    """Fraction of pixels whose grey level changed by > delta (sensitive to text edits)."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    d = cv2.absdiff(cv2.GaussianBlur(a, (3, 3), 0), cv2.GaussianBlur(b, (3, 3), 0))
    return float((d > delta).mean())


class FrameComparer:
    """Combines pHash distance, SSIM and changed-pixel ratio into one decision."""

    def __init__(self, phash_threshold: int, ssim_threshold: float, pixel_threshold: float):
        self.ph, self.ss, self.px = phash_threshold, ssim_threshold, pixel_threshold

    def features(self, img: np.ndarray):
        import imagehash

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
        return imagehash.phash(Image.fromarray(rgb)), _gray_small(img, 160), _gray_small(img, 320)

    def changed(self, fa, fb) -> bool:
        return (fa[0] - fb[0] > self.ph or ssim(fa[1], fb[1]) < self.ss
                or pixel_change_ratio(fa[2], fb[2]) > self.px)

    def distance(self, fa, fb) -> float:
        """Scalar dissimilarity used to pick which neighbouring slides to merge first."""
        return pixel_change_ratio(fa[2], fb[2]) * 10 + (1 - ssim(fa[1], fb[1])) + (fa[0] - fb[0]) / 64


def detect_changes(feats: list, cmp: "FrameComparer") -> list[int]:
    """Indices where a new slide begins (always includes 0)."""
    if not feats:
        return []
    return [0] + [i for i in range(1, len(feats)) if cmp.changed(feats[i], feats[i - 1])]


def build_runs(starts: list[int], n: int, every_s: float, min_dur_s: float) -> list[list[int]]:
    """Turn change points into runs of frame indices, merging runs shorter than min_dur."""
    bounds = starts + [n]
    runs = [list(range(bounds[k], bounds[k + 1])) for k in range(len(starts)) if bounds[k] < bounds[k + 1]]
    min_frames = max(1, int(round(min_dur_s / every_s)))
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for k, r in enumerate(runs):
            if len(r) < min_frames:
                if k + 1 < len(runs):   # transition frames belong to the slide that follows
                    runs[k + 1] = r + runs[k + 1]
                else:
                    runs[k - 1] = runs[k - 1] + r
                runs.pop(k)
                changed = True
                break
    return runs


def _representative(run: list[int], frames: list[np.ndarray]) -> int:
    half = run[len(run) // 2:] or run
    return max(half, key=lambda i: sharpness(frames[i]))


def extract_frame(video: str | Path, t: float, dst: Path) -> Path:
    if not dst.exists():
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", str(dst)]
        subprocess.run(cmd, capture_output=True)
    return dst


def detect_slides(run_dir: Path, video_path: str | None, duration: float, cfg, progress=None) -> dict:
    out_f = run_dir / "keyframes.json"
    cached = read_json(out_f)
    if cached is not None:
        return cached
    if not video_path:
        result = {"keyframes": [], "reason": "no video stream (audio/captions only)"}
        write_json(out_f, result)
        return result

    every = float(cfg.get("slides.sample_every_s", 2))
    width = int(cfg.get("slides.frame_width", 640))
    frames_dir = run_dir / "frames_tmp"
    if progress:
        progress("Sampling frames with FFmpeg")
    sampled = sample_frames(video_path, frames_dir, every, width)
    if not sampled:
        result = {"keyframes": [], "reason": "no frames decoded"}
        write_json(out_f, result)
        return result
    frames = [cv2.imread(str(p)) for _, p in sampled]
    if progress:
        progress(f"Detecting slide changes in {len(frames)} frames")
    cmp = FrameComparer(int(cfg.get("slides.phash_threshold", 12)), float(cfg.get("slides.ssim_threshold", 0.8)),
                        float(cfg.get("slides.pixel_diff_threshold", 0.006)))
    feats = [cmp.features(f) for f in frames]
    starts = detect_changes(feats, cmp)
    runs = build_runs(starts, len(frames), every, float(cfg.get("slides.min_slide_duration_s", 6)))
    reps = [_representative(r, frames) for r in runs]

    # collapse A -> A' neighbours (e.g. after a short cut to the presenter) with the full comparison
    merged_runs, merged_reps = [runs[0]], [reps[0]]
    for r, rep in zip(runs[1:], reps[1:]):
        if not cmp.changed(feats[merged_reps[-1]], feats[rep]):
            merged_runs[-1] = merged_runs[-1] + r
            merged_reps[-1] = _representative(merged_runs[-1], frames)
        else:
            merged_runs.append(r)
            merged_reps.append(rep)

    # bound OCR cost on very dynamic videos (talking head, screen recordings)
    cap = max(1, int((duration or len(frames) * every) / 60 * float(cfg.get("slides.max_slides_per_min", 4))))
    while len(merged_runs) > cap:
        d = [cmp.distance(feats[merged_reps[k]], feats[merged_reps[k + 1]]) for k in range(len(merged_runs) - 1)]
        k = int(np.argmin(d))
        merged_runs[k: k + 2] = [merged_runs[k] + merged_runs[k + 1]]
        merged_reps[k: k + 2] = [_representative(merged_runs[k], frames)]

    kf_dir = run_dir / "keyframes"
    kf_dir.mkdir(exist_ok=True)
    thumb_w = int(cfg.get("slides.thumbnail_width", 360))
    keyframes = []
    for k, (r, rep) in enumerate(zip(merged_runs, merged_reps)):
        start = r[0] * every
        end = min(duration or (r[-1] + 1) * every, (r[-1] + 1) * every)
        t = rep * every
        frame_p = extract_frame(video_path, t, kf_dir / f"slide_{k:03d}.png")
        if not frame_p.exists():  # fall back to the sampled frame
            shutil.copy(sampled[rep][1], kf_dir / f"slide_{k:03d}.jpg")
            frame_p = kf_dir / f"slide_{k:03d}.jpg"
        thumb = kf_dir / f"thumb_{k:03d}.jpg"
        img = Image.open(frame_p).convert("RGB")
        img.thumbnail((thumb_w, thumb_w))
        img.save(thumb, quality=85)
        keyframes.append({
            "slide_id": k, "start": round(start, 2), "end": round(end, 2), "frame_time": round(t, 2),
            "frame": frame_p.relative_to(run_dir).as_posix(), "thumbnail": thumb.relative_to(run_dir).as_posix(),
            "sharpness": round(sharpness(frames[rep]), 1),
            "pixel_std": round(float(np.std(_gray_small(frames[rep]))), 1),
        })
    if keyframes:
        keyframes[-1]["end"] = round(max(keyframes[-1]["end"], duration or 0), 2)
    shutil.rmtree(frames_dir, ignore_errors=True)
    result = {"keyframes": keyframes, "frames_sampled": len(frames), "raw_changes": len(starts),
              "sample_every_s": every}
    write_json(out_f, result)
    return result
