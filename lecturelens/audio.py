"""Stage 2 — Audio with FFmpeg: 16 kHz mono WAV, low-bitrate MP3 for upload,
chunking with overlap, and merging chunk transcripts (overlap de-dup +
timestamp offsets)."""

from __future__ import annotations

import json
import logging
import platform
import re
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class FFmpegMissing(Exception):
    pass


def ffmpeg_install_hint() -> str:
    sysname = platform.system()
    if sysname == "Windows":
        return "Install FFmpeg:  winget install ffmpeg   (then open a new terminal)"
    if sysname == "Darwin":
        return "Install FFmpeg:  brew install ffmpeg"
    return "Install FFmpeg:  sudo apt install ffmpeg   (or your distro's package manager)"


def check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FFmpegMissing(f"'{tool}' not found on PATH. {ffmpeg_install_hint()}")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    check_ffmpeg()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {proc.stderr.strip()[-600:]}")
    return proc


def probe_duration(path: str | Path) -> float:
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]).stdout
    try:
        return float(json.loads(out)["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        return 0.0


def has_audio_stream(path: str | Path) -> bool:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                "-of", "csv=p=0", str(path)]).stdout
    return bool(out.strip())


def to_wav16k(src: str | Path, dst: str | Path) -> Path:
    dst = Path(dst)
    if not dst.exists():
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
              "-c:a", "pcm_s16le", str(dst)])
    return dst


def to_upload_mp3(src: str | Path, dst: str | Path, bitrate: str = "32k", start: float | None = None,
                  duration: float | None = None) -> Path:
    """Low-bitrate mono MP3 (Groq limit is 25 MB: 1 h at 32 kb/s ~ 14 MB)."""
    dst = Path(dst)
    if dst.exists():
        return dst
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", bitrate, str(dst)]
    _run(cmd)
    return dst


def plan_chunks(total_s: float, chunk_s: float = 600, overlap_s: float = 3) -> list[tuple[float, float]]:
    """[(start, duration)] covering [0, total] with `overlap_s` overlap between neighbours."""
    if total_s <= chunk_s:
        return [(0.0, total_s)]
    chunks, start = [], 0.0
    while start < total_s - 0.5:
        dur = min(chunk_s, total_s - start)
        chunks.append((start, dur))
        if start + dur >= total_s:
            break
        start += chunk_s - overlap_s
    return chunks


def split_audio(src: str | Path, out_dir: str | Path, chunk_s: float, overlap_s: float, bitrate: str = "32k",
                total_s: float | None = None) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = total_s or probe_duration(src)
    pieces = []
    for i, (start, dur) in enumerate(plan_chunks(total, chunk_s, overlap_s)):
        p = to_upload_mp3(src, out / f"chunk_{i:03d}.mp3", bitrate, start, dur)
        pieces.append({"index": i, "path": str(p), "offset": start, "duration": dur})
    return pieces


_NORM = re.compile(r"[^a-z0-9']+")


def _norm_words(text: str) -> list[str]:
    return [w for w in _NORM.sub(" ", text.lower()).split() if w]


def _word_overlap(tail: list[str], head: list[str], max_k: int = 30) -> int:
    """Largest k such that the last k normalized words of `tail` == first k of `head`."""
    best = 0
    for k in range(1, min(max_k, len(tail), len(head)) + 1):
        if tail[-k:] == head[:k]:
            best = k
    return best


def merge_chunk_segments(chunks: list[tuple[float, list[dict]]]) -> list[dict]:
    """Merge per-chunk ASR segments into one timeline.

    `chunks` = [(offset_seconds, [{start, end, text}, ...]), ...] with times
    relative to each chunk. Timestamps are shifted by the offset; segments
    that fall entirely inside audio already covered by the previous chunk are
    dropped, and words repeated across the boundary are removed.
    """
    merged: list[dict] = []
    for offset, segs in chunks:
        shifted = [{"start": s["start"] + offset, "end": s["end"] + offset, "text": s["text"].strip()}
                   for s in segs if s.get("text", "").strip()]
        if merged:
            last_end = merged[-1]["end"]
            shifted = [s for s in shifted if s["end"] > last_end + 0.3]
            if shifted:
                first = shifted[0]
                tail = _norm_words(" ".join(m["text"] for m in merged[-3:]))
                words = first["text"].split()
                head = _norm_words(first["text"])
                k = _word_overlap(tail, head)
                if k and len(head) == len(words):
                    first["text"] = " ".join(words[k:])
                elif k:  # punctuation-only tokens make lengths differ; strip by normalized count
                    kept, seen = [], 0
                    for w in words:
                        if seen < k and _norm_words(w):
                            seen += 1
                            continue
                        if seen < k:
                            continue
                        kept.append(w)
                    first["text"] = " ".join(kept)
                if first["start"] < last_end:
                    first["start"] = last_end
                shifted = [s for s in shifted if s["text"]]
        merged.extend(shifted)
    for i, s in enumerate(merged):
        s["id"] = i
        s["end"] = max(s["end"], s["start"])
    return merged
